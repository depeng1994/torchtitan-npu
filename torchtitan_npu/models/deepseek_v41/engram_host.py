# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-authoritative Engram storage, sparse training and Torch lookup."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torchtitan.tools.logging import logger

from torchtitan_npu.extensions.components.optimizer import register_host_sparse_table

from .engram import EngramTable
from .engram_lookup import HostEngramLookup


class HostEngramTable(EngramTable):
    """CPU table sharded over EP, with a Torch lookup and CPU SparseAdam."""

    uses_host_offload = True
    supports_model_compile = False

    @dataclass(kw_only=True, slots=True)
    class Config(EngramTable.Config):
        pin_memory: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.pin_memory = config.pin_memory
        self._ep_rank = 0
        self._ep_size = 1
        self._pending_sparse_grad: torch.Tensor | None = None
        self._grad_keepalive = torch.zeros((), requires_grad=True)
        self._replica_group: dist.ProcessGroup | None = None
        self._replica_size = 1
        self._mark_host_weight()

    def forward(
        self,
        input_ids_BL: torch.Tensor,
        positions_BL: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # EngramTable flattens the token/head axes with ``view``. Nested FSDP
        # warns that returning a view can lose its pre-backward hook if a
        # caller later performs an in-place op, so end the view chain at this
        # module boundary.
        return super().forward(input_ids_BL, positions_BL, **kwargs).clone()

    def _mark_host_weight(self) -> None:
        # Optimizer construction follows parallelize/to_empty, so preserve the
        # Host marker and checkpoint shard suffix on replacement parameters.
        self.weight._engram_host_offload = True  # type: ignore[attr-defined]
        self.weight._engram_checkpoint_suffix = self._checkpoint_suffix()  # type: ignore[attr-defined]

    def _checkpoint_suffix(self) -> str:
        if self._ep_size <= 1:
            return ""
        return f".ep_shard_{self._ep_rank:05d}_of_{self._ep_size:05d}"

    def parallelize(self, parallel_dims) -> None:
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        ep_size = ep_mesh.size() if ep_mesh is not None else 1
        if self.num_embeddings % ep_size != 0:
            raise ValueError(
                f"Engram table has {self.num_embeddings} physical rows, which is not divisible by EP degree {ep_size}."
            )

        register_host_sparse_table(self)

        # Keep weight out of generic SpmdLayout distribution. Hash metadata is
        # still parallelized by the inherited sharding config.
        global_weight = self._parameters.pop("weight")
        assert isinstance(global_weight, torch.nn.Parameter)
        try:
            super().parallelize(parallel_dims)
        finally:
            local_rows = self.num_embeddings // ep_size
            local_weight = torch.nn.Parameter(
                torch.empty(
                    local_rows,
                    self.embedding_dim,
                    dtype=global_weight.dtype,
                    device=global_weight.device,
                ),
                requires_grad=global_weight.requires_grad,
            )
            self.register_parameter("weight", local_weight)

        self._ep_rank = ep_mesh.get_local_rank() if ep_mesh is not None else 0
        self._ep_size = ep_size
        self._mark_host_weight()
        logger.info(
            "Engram layer %d: %s, CPU shard %s, EP%d, SparseAdam",
            self.layer_id,
            type(self).__name__,
            tuple(self.weight.shape),
            ep_size,
        )

    def wire_sparse_grad_replicas(self, *, edp_mesh, edp_mesh_dims=None) -> None:
        """Record the ranks that hold a copy of this EP shard.

        Rows are partitioned along EP only, so the E-FSDP and DP-replicate axes
        carry replicas of the same shard fed by different tokens. FSDP performs
        the equivalent reduction for the parameters it manages; this table is
        deliberately unmanaged, so it has to do it itself.

        Under the SPMD backends ``edp_mesh`` is the whole sparse storage mesh,
        EP axis included, and ``edp_mesh_dims`` is what names its data-parallel
        axes. Reducing over the EP axis as well would sum shards that own
        disjoint row ranges, so the axes are selected rather than flattened
        wholesale.
        """
        self._replica_group = None
        self._replica_size = 1
        if edp_mesh is None:
            return
        if edp_mesh_dims is not None:
            axes = tuple(axis for axis in (edp_mesh_dims.replicate, edp_mesh_dims.shard) if axis)
            if not axes:
                return
            replica_mesh = edp_mesh[axes]
        else:
            replica_mesh = edp_mesh
        if replica_mesh.size() <= 1:
            return
        # A process group needs one dimension.
        flat_mesh = replica_mesh if replica_mesh.ndim == 1 else replica_mesh._flatten()
        self._replica_group = flat_mesh.get_group()
        self._replica_size = flat_mesh.size()
        logger.info(
            "Engram layer %d reduces its sparse gradient across %d replicas of its EP shard",
            self.layer_id,
            self._replica_size,
        )

    @torch.no_grad()
    def reduce_sparse_gradient_across_replicas(self) -> None:
        """Sum this step's sparse gradient over the replicas of this shard.

        Called from the gradient clipper, which TorchTitan runs unconditionally
        before every optimizer step, so the norm is taken on the reduced
        gradient. Each rank pads its owner-local rows to the group maximum,
        all-gathers them, and coalesces: every replica ends up with the same
        summed gradient, which keeps their weights and SparseAdam state in step
        without ever materializing a dense shard gradient.
        """
        group = self._replica_group
        if group is None:
            return

        device = self.token_id_map.device
        pending = self._pending_sparse_grad
        if pending is None:
            local_ids = torch.empty(0, dtype=torch.int64)
            local_values = torch.empty(0, self.embedding_dim, dtype=torch.float32)
        else:
            local_ids = pending.indices()[0]
            local_values = pending.values()

        # Row counts differ per replica, so the gather has to be padded to the
        # largest one; collecting the counts first also says where to cut.
        counts = torch.zeros(self._replica_size, dtype=torch.int64, device=device)
        counts[dist.get_rank(group)] = local_ids.numel()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        max_rows = int(counts.max())
        if max_rows == 0:
            return

        padded_ids = torch.zeros(max_rows, dtype=torch.int64, device=device)
        padded_ids[: local_ids.numel()] = local_ids.to(device=device)
        padded_values = torch.zeros(max_rows, self.embedding_dim, dtype=torch.float32, device=device)
        padded_values[: local_values.shape[0]] = local_values.to(device=device)

        gathered_ids = [torch.empty_like(padded_ids) for _ in range(self._replica_size)]
        gathered_values = [torch.empty_like(padded_values) for _ in range(self._replica_size)]
        dist.all_gather(gathered_ids, padded_ids, group=group)
        dist.all_gather(gathered_values, padded_values, group=group)

        ids_parts = []
        values_parts = []
        for replica in range(self._replica_size):
            rows = int(counts[replica])
            if rows:
                ids_parts.append(gathered_ids[replica][:rows].to(device="cpu"))
                values_parts.append(gathered_values[replica][:rows].to(device="cpu"))

        self._pending_sparse_grad = torch.sparse_coo_tensor(
            torch.cat(ids_parts).unsqueeze(0),
            torch.cat(values_parts),
            size=self.weight.shape,
            dtype=torch.float32,
            device="cpu",
            check_invariants=False,
        ).coalesce()

    def _apply(self, fn, recurse=True):
        """Apply device transforms to metadata while keeping weight on CPU."""
        weight = self._parameters.pop("weight")
        assert isinstance(weight, torch.nn.Parameter)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            if weight.is_meta:
                materialized = torch.empty(
                    weight.shape,
                    dtype=weight.dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                weight = torch.nn.Parameter(materialized, requires_grad=weight.requires_grad)
            self.register_parameter("weight", weight)
            self._mark_host_weight()
        return result

    def storage_dtype(self, *, param_dtype: torch.dtype) -> torch.dtype:
        # The CPU shard is passed to FSDP as an ignored parameter, so the
        # mixed-precision policy never casts it.
        return self.weight.dtype

    def _init_self_parameters(self) -> None:
        if self.weight.device.type != "cpu":
            raise RuntimeError(
                f"Host-offload Engram weight must be on CPU during initialization, got {self.weight.device}."
            )
        # Rank-dependent streams avoid repeating the same local shard on every
        # EP owner. Exact cross-EP-degree initialization parity is not promised;
        # checkpoints preserve the initialized values thereafter.
        seed = (torch.initial_seed() + 1000003 * self.layer_id + 9176 * self._ep_rank) % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self._init_param("weight", self.weight)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # The table is CPU-resident, while hash metadata follows input_ids on
        # the accelerator. After to_empty(), token_id_map records that device.
        device = buffer_device if buffer_device is not None else self.token_id_map.device
        EngramTable._init_self_buffers(self, buffer_device=device)
        # The keepalive gradient arrives from the accelerator, so the anchor has
        # to live there too.
        self._grad_keepalive = torch.zeros((), device=device, requires_grad=True)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        suffix = self._checkpoint_suffix()
        if suffix:
            destination[prefix + "weight" + suffix] = destination.pop(prefix + "weight")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        suffix = self._checkpoint_suffix()
        shard_key = prefix + "weight" + suffix
        weight_key = prefix + "weight"
        if suffix and shard_key in state_dict:
            state_dict[weight_key] = state_dict.pop(shard_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def accumulate_sparse_gradient(
        self,
        local_row_ids: torch.Tensor,
        grad_rows: torch.Tensor,
    ) -> None:
        """Accumulate one microbatch's owner-local rows in FP32 on CPU."""
        local_row_ids = local_row_ids.reshape(-1)
        grad_rows = grad_rows.reshape(local_row_ids.numel(), self.embedding_dim)
        ids = local_row_ids.to(device="cpu", dtype=torch.int64)
        values = grad_rows.to(device="cpu", dtype=torch.float32)
        if ids.numel():
            min_id = int(ids.min())
            max_id = int(ids.max())
            if min_id < 0 or max_id >= self.weight.shape[0]:
                raise IndexError(
                    "Engram lookup returned local row outside "
                    f"[0, {self.weight.shape[0]}): got range [{min_id}, {max_id}] "
                    f"across {ids.numel()} sparse entries."
                )
        sparse_grad = torch.sparse_coo_tensor(
            ids.unsqueeze(0),
            values,
            size=self.weight.shape,
            dtype=torch.float32,
            device="cpu",
            check_invariants=False,
        ).coalesce()
        if self._pending_sparse_grad is None:
            self._pending_sparse_grad = sparse_grad
        else:
            self._pending_sparse_grad = (self._pending_sparse_grad + sparse_grad).coalesce()

    def prepare_sparse_optimizer_step(self) -> None:
        if self._pending_sparse_grad is None:
            self.weight.grad = None
            return
        self.weight.grad = self._pending_sparse_grad.to(dtype=self.weight.dtype)

    def clear_sparse_gradient(self) -> None:
        self._pending_sparse_grad = None
        self.weight.grad = None

    def mark_sparse_step_complete(self) -> None:
        self._pending_sparse_grad = None

    def pending_sparse_grad(self) -> torch.Tensor | None:
        """The accumulated CPU sparse gradient, before it reaches SparseAdam."""
        return self._pending_sparse_grad

    def scale_pending_sparse_grad(self, scale: float) -> None:
        """Apply the global gradient-clipping coefficient to the sparse gradient."""
        if self._pending_sparse_grad is None:
            return
        self._pending_sparse_grad = torch.sparse_coo_tensor(
            self._pending_sparse_grad.indices(),
            self._pending_sparse_grad.values() * scale,
            self._pending_sparse_grad.shape,
        ).coalesce()

    def _distributed_lookup(self, row_ids_N: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            raise RuntimeError("Torch Host Engram lookup currently supports eager training only.")
        return HostEngramLookup.apply(self.weight, row_ids_N, self, self._grad_keepalive)


__all__ = ["HostEngramTable"]
