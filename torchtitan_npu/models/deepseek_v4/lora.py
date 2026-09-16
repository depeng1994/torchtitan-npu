# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

__all__ = ["DEEPSEEK_V4_LORA_TARGETS", "DeepSeekV4LoRAConverter", "peft_target_modules"]

import math
from dataclasses import dataclass, fields, replace
from functools import cache
from typing import TYPE_CHECKING, cast

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.compiler import _is_non_strict_tracing
from torch.distributed.tensor import DTensor
from torchtitan.components.lora import LoRAConverter, _lora_adapter_sharding
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import derive
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .model import DeepSeekV4Model

_PEFT_MODULE_SUFFIXES = {
    "lm_head": "lm_head",
    "attention.wq_a": "self_attn.q_a_proj",
    "attention.wq_b": "self_attn.q_b_proj",
    "attention.wkv": "self_attn.kv_proj",
    "attention.wo_a": "self_attn.o_a_proj",
    "attention.wo_b": "self_attn.o_b_proj",
    "attention.compressor.wkv": "self_attn.compressor.kv_proj",
    "attention.compressor.wgate": "self_attn.compressor.gate_proj",
    "moe.router.gate": "mlp.gate",
    "moe.shared_experts.w1": "mlp.shared_experts.gate_proj",
    "moe.shared_experts.w2": "mlp.shared_experts.down_proj",
    "moe.shared_experts.w3": "mlp.shared_experts.up_proj",
    "e_proj": "e_proj",
    "h_proj": "h_proj",
}

DEEPSEEK_V4_LORA_TARGETS = tuple(_PEFT_MODULE_SUFFIXES)


def peft_target_modules(targets: Iterable[str]) -> tuple[str, ...]:
    names = []
    for target in targets:
        suffix = next((name for name in _PEFT_MODULE_SUFFIXES if _matches_suffix(target, name)), None)
        if suffix is None:
            raise ValueError(f"Unsupported PEFT LoRA target: {target}")
        prefix = target.removesuffix(suffix)
        name = f"{prefix}{_PEFT_MODULE_SUFFIXES[suffix]}"
        names.append(f"model.{name}" if prefix else name)
    return tuple(dict.fromkeys(names))


_GROUPED_EXPERTS_SUFFIX = "moe.routed_experts.inner_experts"


@dataclass(kw_only=True)
class _LoRAOptions:
    rank: int
    alpha: float
    chunk_rows: int


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Protocol

    class _LoRAModule(Protocol):
        Config: type

    @dataclass(kw_only=True)
    class _LinearLoRAConfig(_LoRAOptions, Linear.Config):
        pass

    @dataclass(kw_only=True)
    class _BatchedLoRAConfig(_LoRAOptions, BatchedLinear.Config):
        pass

    @dataclass(kw_only=True)
    class _GroupedLoRAConfig(_LoRAOptions, GroupedExperts.Config):
        pass


def _matches_suffix(fqn: str, suffix: str) -> bool:
    return fqn == suffix or fqn.endswith(f".{suffix}")


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _dsv4_lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    base_weight_sharding = base_sharding.state_shardings.get("weight") if base_sharding else None
    replicated_weight = dense_param_placement(tp=spmd.R)
    if base_weight_sharding == replicated_weight:
        replicated = ShardingConfig(state_shardings={"weight": replicated_weight})
        return replicated, replicated
    return _lora_adapter_sharding(base_sharding)


def _make_lora_adapter_config_cls(parent_config_cls: type) -> type:
    @dataclass(kw_only=True, slots=True)
    class Config(_LoRAOptions, parent_config_cls):  # type: ignore[misc]
        pass

    return Config


@cache
def _get_linear_lora_cls(parent_cls: type[Module]) -> type[_LoRAModule]:
    batched = issubclass(parent_cls, BatchedLinear)

    class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
        Config = _make_lora_adapter_config_cls(parent_cls.Config)

        def __init__(self, config: _LinearLoRAConfig | _BatchedLoRAConfig) -> None:
            super().__init__(config)
            self._lora_scaling = config.alpha / config.rank
            self._lora_chunk_rows = config.chunk_rows
            lora_a_sharding, lora_b_sharding = _dsv4_lora_adapter_sharding(config.sharding_config)
            self.lora_a = Linear.Config(
                in_features=config.in_features,
                out_features=config.rank,
                bias=False,
                sharding_config=lora_a_sharding,
                param_init={"weight": lambda weight: nn.init.kaiming_uniform_(weight, a=math.sqrt(5))},
            ).build()
            if batched:
                self.lora_b = BatchedLinear.Config(
                    n_heads=cast("_BatchedLoRAConfig", config).n_heads,
                    in_features=config.rank,
                    out_features=config.out_features,
                    sharding_config=lora_b_sharding,
                    param_init={"weight": nn.init.zeros_},
                ).build()
            else:
                self.lora_b = Linear.Config(
                    in_features=config.rank,
                    out_features=config.out_features,
                    bias=False,
                    sharding_config=lora_b_sharding,
                    param_init={"weight": nn.init.zeros_},
                ).build()

        def _lora_delta(self, rows: torch.Tensor) -> torch.Tensor:
            return self.lora_b(self.lora_a(rows) if batched else rows)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = super().forward(x)
            if batched:
                adapter_rows = x.reshape(-1, x.shape[-2], x.shape[-1])
                output_rows = output.reshape(-1, output.shape[-2], output.shape[-1])
            else:
                hidden = self.lora_a(x)
                adapter_rows = hidden.reshape(-1, hidden.shape[-1])
                output_rows = output.reshape(-1, output.shape[-1])
            if torch.compiler.is_compiling() or _is_non_strict_tracing():
                output_rows.add_(self._lora_delta(adapter_rows), alpha=self._lora_scaling)
                return output
            for start in range(0, output_rows.shape[0], self._lora_chunk_rows):
                end = min(start + self._lora_chunk_rows, output_rows.shape[0])
                output_rows[start:end].add_(self._lora_delta(adapter_rows[start:end]), alpha=self._lora_scaling)
            return output

    LoRALinear.__name__ = f"DeepSeekV4LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = LoRALinear.__name__
    return LoRALinear


@cache
def _get_grouped_lora_cls(parent_cls: type[Module]) -> type[_LoRAModule]:
    class GroupedLoRAExperts(parent_cls):  # type: ignore[valid-type, misc]
        Config = _make_lora_adapter_config_cls(parent_cls.Config)

        w13_lora_a: nn.Parameter
        w13_lora_b: nn.Parameter
        w2_lora_a: nn.Parameter
        w2_lora_b: nn.Parameter
        _grouped_mm: Callable[..., torch.Tensor]

        def __init__(self, config: _GroupedLoRAConfig) -> None:
            super().__init__(config)
            self._lora_scaling = config.alpha / config.rank
            self._lora_chunk_rows = config.chunk_rows
            adapter_shapes = {
                "w13_lora_a": (config.num_experts, config.rank, config.dim),
                "w13_lora_b": (config.num_experts, config.hidden_dim, 2, config.rank),
                "w2_lora_a": (config.num_experts, config.rank, config.hidden_dim),
                "w2_lora_b": (config.num_experts, config.dim, config.rank),
            }
            for name, shape in adapter_shapes.items():
                self.register_parameter(name, nn.Parameter(torch.empty(shape)))

        def forward(self, input_tokens: torch.Tensor, token_counts: torch.Tensor, **dimensioned_kwargs) -> torch.Tensor:
            unexpected = dimensioned_kwargs.keys() - {"routed_scores_R"}
            if unexpected:
                raise TypeError(f"Unexpected GroupedExperts arguments: {sorted(unexpected)}")
            routed_scores = dimensioned_kwargs.get("routed_scores_R")
            w1 = _local_tensor(cast("torch.Tensor", self.w1_EFD))
            w2 = _local_tensor(cast("torch.Tensor", self.w2_EDF))
            w3 = _local_tensor(cast("torch.Tensor", self.w3_EFD))
            offsets = torch.cumsum(token_counts, dim=0, dtype=torch.int32)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
                for axis in ("dp", "cp"):
                    spmd.mutate_type(offsets, axis, src=spmd.P, dst=spmd.V)

            gate = self._grouped_mm(A=input_tokens.bfloat16(), B_t=w1.bfloat16().transpose(-2, -1), offs=offsets)
            up = self._grouped_mm(A=input_tokens.bfloat16(), B_t=w3.bfloat16().transpose(-2, -1), offs=offsets)
            gate, up = self._add_grouped_w13_delta(gate=gate, up=up, x=input_tokens, offsets=offsets)
            swiglu_limit = getattr(self, "swiglu_limit", 0.0)
            if swiglu_limit > 0:
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
                gate = torch.clamp(gate, max=swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up
            if routed_scores is not None:
                hidden = (hidden.float() * routed_scores.float().reshape(-1, 1)).to(hidden.dtype)
            output = self._grouped_mm(A=hidden, B_t=w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(input_tokens)
            return self._add_grouped_w2_delta(output=output, x=hidden, offsets=offsets)

        def _iter_grouped_row_chunks(self, num_rows: int, offsets: torch.Tensor):
            if torch.compiler.is_compiling() or _is_non_strict_tracing():
                yield 0, num_rows, offsets.to(torch.int32)
                return
            group_ends = offsets.to(torch.int64)
            group_starts = torch.cat((group_ends.new_zeros(1), group_ends[:-1]))
            for start in range(0, num_rows, self._lora_chunk_rows):
                end = min(start + self._lora_chunk_rows, num_rows)
                chunk_counts = group_ends.clamp(min=start, max=end) - group_starts.clamp(min=start, max=end)
                yield start, end, torch.cumsum(chunk_counts, dim=0, dtype=torch.int32)

        def _add_grouped_w13_delta(
            self, *, gate: torch.Tensor, up: torch.Tensor, x: torch.Tensor, offsets: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            a = _local_tensor(self.w13_lora_a)
            b = _local_tensor(self.w13_lora_b)
            a_t = a.bfloat16().transpose(-2, -1)
            if b.ndim != 4 or b.shape[2] != 2:
                raise ValueError("w13 LoRA B must have shape [E, F, 2, R]")
            gate_b_t = b[:, :, 0, :].bfloat16().transpose(-2, -1)
            up_b_t = b[:, :, 1, :].bfloat16().transpose(-2, -1)
            gate_width = gate.shape[-1]
            if b.shape[1] != gate_width or b.shape[1] != up.shape[-1]:
                raise ValueError("w13 LoRA output width does not match gate/up")
            for start, end, chunk_offsets in self._iter_grouped_row_chunks(x.shape[0], offsets):
                hidden = self._grouped_mm(A=x[start:end].bfloat16(), B_t=a_t, offs=chunk_offsets)
                gate_delta = self._grouped_mm(A=hidden, B_t=gate_b_t, offs=chunk_offsets)
                up_delta = self._grouped_mm(A=hidden, B_t=up_b_t, offs=chunk_offsets)
                gate[start:end].add_(gate_delta.to(gate.dtype), alpha=self._lora_scaling)
                up[start:end].add_(up_delta.to(up.dtype), alpha=self._lora_scaling)
            return gate, up

        def _add_grouped_w2_delta(
            self, *, output: torch.Tensor, x: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            a = _local_tensor(self.w2_lora_a)
            b = _local_tensor(self.w2_lora_b)
            a_t = a.bfloat16().transpose(-2, -1)
            b_t = b.bfloat16().transpose(-2, -1)
            for start, end, chunk_offsets in self._iter_grouped_row_chunks(x.shape[0], offsets):
                hidden = self._grouped_mm(A=x[start:end].bfloat16(), B_t=a_t, offs=chunk_offsets)
                delta = self._grouped_mm(A=hidden, B_t=b_t, offs=chunk_offsets)
                output[start:end].add_(delta.to(output.dtype), alpha=self._lora_scaling)
            return output

    GroupedLoRAExperts.__name__ = f"DeepSeekV4LoRA{parent_cls.__name__}"
    GroupedLoRAExperts.__qualname__ = GroupedLoRAExperts.__name__
    return GroupedLoRAExperts


class DeepSeekV4LoRAConverter(LoRAConverter):
    @dataclass(kw_only=True, slots=True)
    class Config(LoRAConverter.Config):
        rank: int = 64
        alpha: float = 128.0
        rank_experts: int = 64
        dense_chunk_rows: int = 1024
        grouped_chunk_rows: int = 4096
        adapt_routed_experts: bool = True
        include_mtp: bool = True
        strict: bool = True

    def __init__(self, config: Config, **kwargs) -> None:
        if config.rank <= 0 or config.rank_experts <= 0:
            raise ValueError("LoRA ranks must be positive")
        if not math.isfinite(config.alpha) or config.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if config.dense_chunk_rows <= 0 or config.grouped_chunk_rows <= 0:
            raise ValueError("LoRA chunk sizes must be positive")
        self.config = config
        self.targets = tuple(config.target_modules if config.target_modules is not None else DEEPSEEK_V4_LORA_TARGETS)

    def convert(self, model_config: Module.Config) -> Module.Config:
        config = cast("DeepSeekV4LoRAConverter.Config", self.config)
        converted_root = model_config
        targets = self.targets
        if config.target_modules is None and (not config.include_mtp or not getattr(model_config, "mtp_layers", None)):
            targets = tuple(target for target in targets if target not in ("e_proj", "h_proj"))
        matched = dict.fromkeys(targets, 0)
        grouped_matches = 0
        configs = list(model_config.traverse(Module.Config, recurse=True))
        if has_quantization(model_config) or any(
            getattr(cfg, "_torchao_npu_config", None) is not None for _, cfg, _, _ in configs
        ):
            raise NotImplementedError("DeepSeek-V4 LoRA requires an unquantized base model")

        for fqn, cfg, parent, attr in reversed(configs):
            if not isinstance(cfg, Module.Config):
                raise TypeError(f"Expected a Module.Config at {fqn!r}, got {type(cfg).__name__}")
            if not config.include_mtp and (fqn == "mtp_layers" or fqn.startswith("mtp_layers.")):
                continue
            dense_target = next((target for target in targets if _matches_suffix(fqn, target)), None)
            if dense_target is not None and isinstance(cfg, (BatchedLinear.Config, Linear.Config)):
                new_cfg = self._make_lora_config(cfg)
                matched[dense_target] += 1
            elif (
                config.adapt_routed_experts
                and _matches_suffix(fqn, _GROUPED_EXPERTS_SUFFIX)
                and isinstance(cfg, GroupedExperts.Config)
            ):
                new_cfg = self._make_lora_config(cfg)
                grouped_matches += 1
            else:
                continue

            if parent is None:
                converted_root = new_cfg
            elif isinstance(parent, list):
                if not isinstance(attr, int):
                    raise TypeError(f"List parent at {fqn!r} requires an integer index, got {type(attr).__name__}")
                parent[attr] = new_cfg
            else:
                if not isinstance(attr, str):
                    raise TypeError(f"Config parent at {fqn!r} requires a string attribute, got {type(attr).__name__}")
                setattr(parent, attr, new_cfg)

        missing = [target for target, count in matched.items() if not count]
        if missing:
            message = f"DeepSeek-V4 LoRA targets did not match: {missing}"
            if config.strict:
                raise RuntimeError(message)
            logger.warning(message)
        if config.strict and config.adapt_routed_experts and not grouped_matches:
            raise RuntimeError("DeepSeek-V4 LoRA found no routed GroupedExperts")
        logger.info(
            "DeepSeek-V4 LoRA: dense=%d grouped=%d rank=%d expert_rank=%d",
            sum(matched.values()),
            grouped_matches,
            config.rank,
            config.rank_experts,
        )
        return derive(converted_root, _LoRAModelConfig, lora=replace(config, target_modules=list(targets)))

    def _make_lora_config(self, cfg: Module.Config):
        config = cast("DeepSeekV4LoRAConverter.Config", self.config)
        if isinstance(cfg, (BatchedLinear.Config, Linear.Config)):
            factory = _get_linear_lora_cls
        else:
            factory = _get_grouped_lora_cls
        owner = getattr(cfg, "_owner", None)
        if owner is None:
            raise ValueError(f"{type(cfg).__name__} has no owner class to wrap with LoRA")
        lora_cls = factory(owner)
        values = {field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init}
        grouped = isinstance(cfg, GroupedExperts.Config)
        if grouped:
            values["param_init"] = {
                **(cfg.param_init or {}),
                "w13_lora_a": lambda value: nn.init.kaiming_uniform_(value, a=math.sqrt(5)),
                "w13_lora_b": nn.init.zeros_,
                "w2_lora_a": lambda value: nn.init.kaiming_uniform_(value, a=math.sqrt(5)),
                "w2_lora_b": nn.init.zeros_,
            }
        return lora_cls.Config(
            **values,
            rank=config.rank_experts if grouped else config.rank,
            alpha=config.alpha,
            chunk_rows=config.grouped_chunk_rows if grouped else config.dense_chunk_rows,
        )


@dataclass(kw_only=True, slots=True)
class _LoRAModelConfig(DeepSeekV4Model.Config):
    lora: DeepSeekV4LoRAConverter.Config
