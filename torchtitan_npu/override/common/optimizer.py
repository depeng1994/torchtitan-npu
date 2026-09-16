# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides for NPU swap-backed optimizer states and their checkpoints."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from types import MethodType
from typing import Any, Protocol, TypedDict, runtime_checkable

import torch
import torch_npu
from torch.distributed._tensor import DTensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override
from torchtitan.distributed.flex_shard.dist_muon import DistMuon

from torchtitan_npu.extensions.novaswap import swap_api
from torchtitan_npu.extensions.novaswap.swap_engine import SwapEngine, _wait

_ADAMW_SWAP_BUCKET_TIMES = 16


class _CheckpointMetadata(TypedDict):
    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


@runtime_checkable
class _CheckpointableTensor(Protocol):
    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


def make_checkpointable_view(
    tensor_cpu: torch.Tensor,
    *,
    byte_offset: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    global_shape: tuple[int, ...],
    global_offsets: tuple[tuple[int, ...], ...],
    local_offsets: tuple[tuple[int, ...], ...],
    local_sizes: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Expose a logical tensor inside one raw NovaSwap CPU buffer to DCP."""
    if tensor_cpu.device.type != "cpu" or tensor_cpu.dtype != torch.uint8:
        raise TypeError("checkpoint view requires a CPU uint8 swap buffer")

    logical_nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if byte_offset < 0 or byte_offset + logical_nbytes > tensor_cpu.numel():
        raise ValueError(
            f"checkpoint view byte range [{byte_offset}, {byte_offset + logical_nbytes}) "
            f"exceeds the {tensor_cpu.numel()}-byte swap buffer"
        )

    raw_view = tensor_cpu.narrow(0, byte_offset, logical_nbytes)
    view = raw_view.view(dtype).as_strided(shape, stride)
    if not view.is_contiguous():
        raise ValueError("checkpoint view requires a compact contiguous NovaSwap layout")

    setattr(view, "global_shape", global_shape)  # noqa: B010
    setattr(view, "global_offsets", global_offsets)  # noqa: B010
    setattr(view, "local_offsets", local_offsets)  # noqa: B010
    setattr(view, "local_sizes", local_sizes)  # noqa: B010
    if not isinstance(view, _CheckpointableTensor):
        raise TypeError("failed to attach CheckpointableTensor metadata")
    return view


def get_checkpoint_view(
    tensor_name: str,
    tensor: torch.Tensor,
    *,
    byte_offset: int = 0,
) -> torch.Tensor:
    """Build a zero-copy DCP view of one swapped optimizer-state tensor."""
    # The checkpoint adapter intentionally reads NovaSwap's existing registry
    # instead of adding a checkpoint-specific API to the NovaSwap plugin.
    handle = SwapEngine._handles[tensor_name][0]  # pylint: disable=protected-access
    _wait(handle)
    tensor_cpu = handle.tensor_cpu
    if tensor_cpu is None:
        raise RuntimeError(f"NovaSwap CPU buffer is unavailable for {tensor_name!r}")
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return make_checkpointable_view(
        tensor_cpu,
        byte_offset=byte_offset,
        dtype=local.dtype,
        shape=tuple(local.shape),
        stride=tuple(local.stride()),
        **_checkpoint_metadata(tensor, local),
    )


def _checkpoint_metadata(tensor: torch.Tensor, local: torch.Tensor) -> _CheckpointMetadata:
    if isinstance(tensor, DTensor):
        chunks = tensor.__create_chunk_list__()
        if len(chunks) != 1:
            raise RuntimeError(
                f"CheckpointableTensor currently requires exactly one local DTensor chunk, got {len(chunks)}"
            )
        chunk = chunks[0]
        global_offsets = (tuple(chunk.offsets),)
        local_sizes = (tuple(chunk.sizes),)
        global_shape = tuple(tensor.shape)
    else:
        global_offsets = (tuple(0 for _ in local.shape),)
        local_sizes = (tuple(local.shape),)
        global_shape = tuple(local.shape)

    return {
        "global_shape": global_shape,
        "global_offsets": global_offsets,
        "local_offsets": (tuple(0 for _ in local.shape),),
        "local_sizes": local_sizes,
    }


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _replace_local_tensor(tensor: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, DTensor):
        return local
    return DTensor.from_local(
        local,
        tensor.device_mesh,
        tensor.placements,
        shape=tensor.size(),
        stride=tensor.stride(),
        run_check=False,
    )


def make_swap_state_name(*parts: object) -> str:
    return ".".join(map(str, parts))


@dataclass(frozen=True)
class _AdamWSwapBucket:
    group: dict[str, Any]
    parameters: tuple[torch.Tensor, ...]
    state_name: str | None


_SwappedState = tuple[torch.Tensor, str, str, torch.Tensor]


class _NovaSwapAdamW:
    """Run stock AdamW state through NovaSwap with a bucket-level pipeline."""

    _state_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    def __init__(self, optimizer: torch.optim.AdamW) -> None:
        self.optimizer = optimizer
        self._original_step = optimizer.step
        self._buckets: tuple[_AdamWSwapBucket, ...] = ()

        self.checkpoint_locations: dict[tuple[torch.Tensor, str], tuple[str, int]] = {}

    def _parameters(self) -> Iterator[torch.Tensor]:
        for group in self.optimizer.param_groups:
            yield from group["params"]

    @staticmethod
    def _submit(bucket: _AdamWSwapBucket, action: str) -> None:
        if bucket.state_name is not None:
            swap_api.execute(bucket.state_name, action)

    def step(self, closure=None):
        if not self._buckets:
            loss = self._original_step(closure)
            self._buckets = self._build_buckets()
            for bucket in self._buckets:
                self._submit(bucket, "D2H")
            return loss

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._submit(self._buckets[0], "H2D")
        for bucket_index, bucket in enumerate(self._buckets):
            self._submit(bucket, "WAIT_DEVICE")
            if bucket_index + 1 < len(self._buckets):
                self._submit(self._buckets[bucket_index + 1], "H2D")
            self._update_bucket(bucket)
            self._submit(bucket, "D2H")
        return loss

    def ensure_all_state(self) -> None:
        parameters = tuple(self._parameters())
        missing = []
        for parameter in parameters:
            if parameter.requires_grad and not self.optimizer.state.get(parameter):
                missing.append(parameter)
        if missing:
            wrapped_step = self.optimizer.step
            self.optimizer.step = self._original_step  # pyrefly: ignore [bad-override]
            try:
                init_optim_state(self.optimizer)
            finally:
                self.optimizer.step = wrapped_step  # pyrefly: ignore [bad-override]

        unbucketed_ids = set()
        for parameter in parameters:
            state_is_ready = "exp_avg" in self.optimizer.state[parameter]
            if state_is_ready and (parameter, "exp_avg") not in self.checkpoint_locations:
                unbucketed_ids.add(id(parameter))
        new_buckets = self._build_buckets(unbucketed_ids)
        self._buckets += new_buckets
        for bucket in new_buckets:
            self._submit(bucket, "D2H")

    def refresh_bucket_groups(self) -> None:
        group_by_parameter = {
            id(parameter): group for group in self.optimizer.param_groups for parameter in group["params"]
        }
        self._buckets = tuple(
            _AdamWSwapBucket(
                group=group_by_parameter[id(bucket.parameters[0])],
                parameters=bucket.parameters,
                state_name=bucket.state_name,
            )
            for bucket in self._buckets
        )

    def _collect_moments(
        self, parameters: tuple[torch.Tensor, ...]
    ) -> list[tuple[dict[str, Any], str, torch.Tensor, torch.Tensor]]:
        moments = []
        for parameter in parameters:
            state = self.optimizer.state[parameter]
            for state_key in self._state_keys:
                moment = state.get(state_key)
                if moment is None:
                    continue
                local_moment = _local_tensor(moment)
                if local_moment.numel() != 0:
                    moments.append((state, state_key, moment, local_moment))
        return moments

    def _pack_moments(
        self,
        moments: list[tuple[dict[str, Any], str, torch.Tensor, torch.Tensor]],
        bucket_index: int,
    ) -> str | None:
        if not moments:
            return None
        device, dtype = moments[0][3].device, moments[0][3].dtype
        if not all(local.device == device and local.dtype == dtype for _, _, _, local in moments):
            raise RuntimeError("AdamW swap bucket states must share a device and dtype")
        flat = torch.empty(
            sum(local.numel() for _, _, _, local in moments),
            dtype=dtype,
            device=device,
        )
        offset = 0
        for state, state_key, moment, local_moment in moments:
            flat_view = flat.narrow(0, offset, local_moment.numel()).view_as(local_moment)
            flat_view.copy_(local_moment)
            state[state_key] = _replace_local_tensor(moment, flat_view)
            offset += local_moment.numel()
        state_name = make_swap_state_name("adamw", id(self.optimizer), "bucket", bucket_index)
        swap_api.register_tensor(flat, state_name)
        return state_name

    def _append_bucket(
        self,
        buckets: list[_AdamWSwapBucket],
        group: dict[str, Any] | None,
        parameters: list[torch.Tensor],
    ) -> None:
        if not parameters:
            return
        if group is None:
            raise RuntimeError("AdamW swap bucket has no parameter group")
        bucket_parameters = tuple(parameters)
        moments = self._collect_moments(bucket_parameters)
        state_name = self._pack_moments(moments, len(self._buckets) + len(buckets))
        bucket = _AdamWSwapBucket(group=group, parameters=bucket_parameters, state_name=state_name)
        buckets.append(bucket)
        self._record_checkpoint_locations(bucket)

    def _build_buckets(self, parameter_ids: set[int] | None = None) -> tuple[_AdamWSwapBucket, ...]:
        state_parameters = []
        for parameter in self._parameters():
            if "exp_avg" in self.optimizer.state[parameter]:
                state_parameters.append(parameter)
        total_numel = sum(_local_tensor(parameter).numel() for parameter in state_parameters)
        bucket_numel_limit = max(total_numel // _ADAMW_SWAP_BUCKET_TIMES, 1)
        buckets: list[_AdamWSwapBucket] = []
        current_group: dict[str, Any] | None = None
        current_parameters: list[torch.Tensor] = []
        current_numel = 0
        current_signature: tuple[torch.device, torch.dtype] | None = None
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if "exp_avg" not in self.optimizer.state[parameter]:
                    continue
                if parameter_ids is not None and id(parameter) not in parameter_ids:
                    continue
                parameter_numel = _local_tensor(parameter).numel()
                local_exp_avg = _local_tensor(self.optimizer.state[parameter]["exp_avg"])
                parameter_signature = (local_exp_avg.device, local_exp_avg.dtype)
                starts_new_bucket = current_parameters and (
                    group is not current_group
                    or current_numel + parameter_numel > bucket_numel_limit
                    or parameter_signature != current_signature
                )
                if starts_new_bucket:
                    self._append_bucket(buckets, current_group, current_parameters)
                    current_parameters = []
                    current_numel = 0
                current_group = group
                current_signature = parameter_signature
                current_parameters.append(parameter)
                current_numel += parameter_numel
        self._append_bucket(buckets, current_group, current_parameters)
        return tuple(buckets)

    def _record_checkpoint_locations(self, bucket: _AdamWSwapBucket) -> None:
        if bucket.state_name is None:
            return
        for parameter in bucket.parameters:
            state = self.optimizer.state[parameter]
            for state_key in self._state_keys:
                moment = state.get(state_key)
                if moment is None:
                    continue
                local_moment = _local_tensor(moment)
                if local_moment.numel() == 0:
                    continue
                self.checkpoint_locations[(parameter, state_key)] = (
                    bucket.state_name,
                    int(local_moment.storage_offset()) * local_moment.element_size(),
                )

    def _update_bucket(self, bucket: _AdamWSwapBucket) -> None:
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [{**bucket.group, "params": bucket.parameters}]
        try:
            # The saved method is the unwrapped upstream AdamW.step.  Calling
            # it with exactly one temporary param group preserves all current
            # AdamW behavior while leaving this wrapper responsible only for
            # bucket scheduling.
            self._original_step()
        finally:
            self.optimizer.param_groups = original_param_groups


def _make_swap(t: torch.Tensor) -> torch.Tensor:
    local = t.to_local() if isinstance(t, DTensor) else t
    # A DTensor may have p.numel() > 0 globally but local.numel() == 0 on ranks
    # that own an empty shard. Swap-memory allocation rejects zero-sized tensors,
    # so preserve such local shards with a regular empty tensor instead.
    out = (
        torch.empty_like(local)
        if local.numel() == 0
        else torch_npu.empty_with_swapped_memory(local.size(), dtype=local.dtype, device=local.device)
    )
    return _replace_local_tensor(t, out)


def _swap_state_init_hook(optimizer, args, kwargs):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=p.device,
                )
                state["exp_avg"] = _make_swap(p).zero_()
                state["exp_avg_sq"] = _make_swap(p).zero_()


class VirtualOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts):
        super().__init__(config=config, model_parts=model_parts)
        for opt in self.optimizers:
            opt.register_step_pre_hook(_swap_state_init_hook)


@override(
    target=OptimizersContainer.Config,
    description="Allocate Adam/AdamW states in swap memory (host-offload)",
)
def virtual(
    cfg: OptimizersContainer.Config,
) -> VirtualOptimizersContainer.Config:
    return derive(cfg, VirtualOptimizersContainer.Config)


class OptimizerStateSwapContainer(OptimizersContainer):
    supports_async_with_pinned_mem = False

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts: list[Any]) -> None:
        super().__init__(config=config, model_parts=model_parts)
        parameter_to_part = {
            id(parameter): part_index
            for part_index, model_part in enumerate(model_parts)
            for parameter in model_part.parameters()
        }
        for optimizer in self.optimizers:
            if isinstance(optimizer, DistMuon):
                part_indices = {
                    parameter_to_part[id(parameter)]
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                if len(part_indices) != 1:
                    raise RuntimeError(
                        "DistMuon optimizer must own parameters from exactly "
                        f"one model part, got part indices {sorted(part_indices)}"
                    )
                self._swap_muon(optimizer, next(iter(part_indices)))
            elif isinstance(optimizer, torch.optim.AdamW):
                self._swap_adamw(optimizer)

    @staticmethod
    def _swap_muon(optimizer: DistMuon, model_part: int) -> None:
        checkpoint_locations: dict[tuple[torch.Tensor, str], tuple[str, int]] = {}
        setattr(optimizer, "_torchtitan_npu_checkpoint_locations", checkpoint_locations)  # noqa: B010
        original_momentum = optimizer._momentum
        original_prepare_local = optimizer._prepare_local
        runtime = optimizer._redistribution_runtime
        original_enqueue_storage_to_compute = runtime._enqueue_storage_to_compute
        deferred_d2h_names: list[str] | None = None

        def momentum(optimizer_self, compute_layout, grad):
            state = optimizer_self.state[compute_layout.param]
            created = "momentum_buffer" not in state
            result = original_momentum(compute_layout, grad)
            if not created:
                return result
            local = result.to_local()
            if local.numel() == 0:
                return result
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            swap_api.register_tensor(local, name)
            checkpoint_locations[(compute_layout.param, "momentum_buffer")] = (
                name,
                0,
            )
            swap_api.execute(name, "D2H")
            return result

        def submit_h2d(compute_layout) -> None:
            state = optimizer.state.get(compute_layout.param)
            if state is None or "momentum_buffer" not in state:
                return
            momentum = state["momentum_buffer"]
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.numel() == 0:
                return
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            if swap_api.get_handle_phase(name) != "H2D":
                swap_api.execute(name, "H2D")

        def prefetch_plan(plan) -> None:
            """Submit a plan's H2D work before FlexShard enters its transfer stream."""
            redistributed = getattr(plan, "redistributed_items", None)
            items = plan.items if redistributed is None else (*redistributed, *plan.unredistributed_items)
            for compute_layout in items:
                submit_h2d(compute_layout)

        def enqueue_storage_to_compute(plan, slot, context, *, prepare):
            """Align swap prefetch/offload with one FlexShard redistribution bucket."""
            nonlocal deferred_d2h_names
            if deferred_d2h_names is not None:
                raise RuntimeError("Nested FlexShard storage-to-compute enqueue is unsupported")

            prefetch_plan(plan)
            deferred_d2h_names = []
            try:
                work = original_enqueue_storage_to_compute(
                    plan,
                    slot,
                    context,
                    prepare=prepare,
                )
            except Exception:
                deferred_d2h_names = None
                raise

            completed_d2h_names = deferred_d2h_names
            deferred_d2h_names = None

            with torch_npu.npu.stream(context.transfer_stream):
                for name in completed_d2h_names:
                    swap_api.execute(name, "D2H")
            return work

        def prepare_local(optimizer_self, compute_layout, out) -> None:
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            momentum = optimizer_self.state[compute_layout.param]["momentum_buffer"]
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.numel() == 0:
                original_prepare_local(compute_layout, out)
                return
            submit_h2d(compute_layout)
            swap_api.execute(name, "WAIT_DEVICE")
            original_prepare_local(compute_layout, out)

            if deferred_d2h_names is None:
                swap_api.execute(name, "D2H")
            else:
                deferred_d2h_names.append(name)

        optimizer._momentum = MethodType(momentum, optimizer)
        optimizer._prepare_local = MethodType(prepare_local, optimizer)
        runtime._enqueue_storage_to_compute = enqueue_storage_to_compute

    @staticmethod
    def _swap_adamw(optimizer: torch.optim.AdamW) -> None:
        swap = _NovaSwapAdamW(optimizer)
        setattr(optimizer, "_torchtitan_npu_checkpoint_locations", swap.checkpoint_locations)  # noqa: B010
        setattr(optimizer, "_torchtitan_npu_swap_adapter", swap)  # noqa: B010

        @Optimizer.profile_hook_step
        @_use_grad_for_differentiable
        def step(_optimizer, closure=None):
            return swap.step(closure)

        # Install the intentional instance-level AdamW bucket-swap wrapper.
        optimizer.step = MethodType(step, optimizer)  # pyrefly: ignore [bad-override]

    @staticmethod
    def _replace_swapped_states_with_checkpoint_views(
        optimizer: torch.optim.Optimizer,
        flat_state: dict[str, Any],
    ) -> None:
        locations = getattr(optimizer, "_torchtitan_npu_checkpoint_locations")  # noqa: B009
        for parameter, fqn, state_name, tensor in _swapped_state_tensors(optimizer):
            local = _local_tensor(tensor)
            if local.numel() == 0:
                continue
            location = locations.get((parameter, state_name))
            if location is None:
                raise RuntimeError(f"missing NovaSwap checkpoint location for optimizer state state.{fqn}.{state_name}")
            tensor_name, byte_offset = location
            flat_state[f"state.{fqn}.{state_name}"] = get_checkpoint_view(
                tensor_name,
                tensor,
                byte_offset=byte_offset,
            )

    @staticmethod
    def _state_for_optimizer(
        state_dict: dict[str, Any],
        checkpoint_views: dict[str, Any],
        originals: list[_SwappedState],
    ) -> dict[str, Any]:
        state_for_optimizer = dict(state_dict)
        for _parameter, fqn, state_name, tensor in originals:
            key = f"state.{fqn}.{state_name}"
            incoming = state_dict.get(key)
            target_view = checkpoint_views[key]
            if incoming is not None and _local_tensor(tensor).numel() != 0:
                if not isinstance(incoming, torch.Tensor):
                    raise TypeError(f"optimizer state {key} must be a Tensor")
                # DCP has already populated the view when both references share an address.
                if incoming.data_ptr() != target_view.data_ptr():
                    target_view.copy_(incoming)
            # Keep the optimizer bound to its original NovaSwap NPU/DTensor object.
            state_for_optimizer[key] = tensor
        return state_for_optimizer

    @staticmethod
    def _restore_optimizer_references(
        optimizer: torch.optim.Optimizer,
        originals: list[_SwappedState],
        param_names: list[Any],
    ) -> None:
        for parameter, _fqn, state_name, tensor in originals:
            optimizer.state[parameter][state_name] = tensor
        for group, names in zip(optimizer.param_groups, param_names, strict=True):
            if names is not None:
                group["param_names"] = names

    def _load_optimizer_state(self, optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]) -> None:
        _ensure_all_optim_state(optimizer)
        originals = list(_swapped_state_tensors(optimizer))
        # DCP writes these target CPU views in place; direct load copies into them below.
        checkpoint_views = get_flat_optim_state_dict(optimizer)
        self._replace_swapped_states_with_checkpoint_views(optimizer, checkpoint_views)
        state_for_optimizer = self._state_for_optimizer(state_dict, checkpoint_views, originals)
        param_names = [group.get("param_names") for group in optimizer.param_groups]
        try:
            load_flat_optim_state_dict(optimizer, state_for_optimizer)
        finally:
            self._restore_optimizer_references(optimizer, originals, param_names)

        swap = getattr(optimizer, "_torchtitan_npu_swap_adapter", None)
        if isinstance(swap, _NovaSwapAdamW):
            swap.refresh_bucket_groups()

    def state_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for optimizer in self.optimizers:
            _ensure_all_optim_state(optimizer)
            flat_state = get_flat_optim_state_dict(optimizer)
            self._replace_swapped_states_with_checkpoint_views(optimizer, flat_state)
            result.update(flat_state)
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer in self.optimizers:
            self._load_optimizer_state(optimizer, state_dict)


def _ensure_all_optim_state(optimizer: torch.optim.Optimizer) -> None:
    swap = getattr(optimizer, "_torchtitan_npu_swap_adapter", None)
    if isinstance(swap, _NovaSwapAdamW):
        swap.ensure_all_state()
    else:
        init_optim_state(optimizer)


def _named_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> Iterator[tuple[torch.Tensor, str]]:
    for group in optimizer.param_groups:
        yield from zip(group["params"], group["param_names"], strict=True)


def _swapped_state_names(optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    if isinstance(optimizer, DistMuon):
        return ("momentum_buffer",)
    if isinstance(optimizer, torch.optim.AdamW):
        return ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    return ()


def _swapped_state_tensors(
    optimizer: torch.optim.Optimizer,
) -> Iterator[_SwappedState]:
    state_names = _swapped_state_names(optimizer)
    for parameter, fqn in _named_optimizer_parameters(optimizer):
        state = optimizer.state.get(parameter, {})
        for state_name in state_names:
            tensor = state.get(state_name)
            if tensor is not None:
                yield parameter, fqn, state_name, tensor


@override(
    target=OptimizersContainer.Config,
    description="Offload DistMuon and AdamW state by globally unique names",
)
def swap_optimizer(
    cfg: OptimizersContainer.Config,
) -> OptimizerStateSwapContainer.Config:
    return derive(cfg, OptimizerStateSwapContainer.Config)
