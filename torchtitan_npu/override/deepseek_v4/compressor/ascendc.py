# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CANN compressor using the operator's native autograd implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan_npu.models.deepseek_v4.compressor import Compressor, CompressorImplementation


def _state_inputs(metadata: Any, x: torch.Tensor, *, ratio: int, head_dim: int):
    varlen = metadata.varlen
    cu_seqlens = varlen.cu_seq_q.to(device=x.device, dtype=torch.int32).contiguous()
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    block_size = 8 if ratio == 4 else 16
    coff = 2 if ratio == 4 else 1
    # CANN tiling requires state_block_table despite its optional schema.
    # In cache_mode=1, block ID 0 skips cache writes for full-document training.
    state_block_table = torch.zeros(
        (lengths.numel(), max((varlen.max_k + block_size - 1) // block_size, 1)),
        dtype=torch.int32,
        device=x.device,
    )
    state_cache = torch.zeros((1, block_size, 2 * coff * head_dim), dtype=torch.float32, device=x.device)
    return state_block_table, state_cache, cu_seqlens, lengths


class AscCompressor(CompressorImplementation):
    @dataclass(kw_only=True, slots=True)
    class Config(CompressorImplementation.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Register the optional kernel only when the fused component is built.
        importlib.import_module("cann_ops_transformer.ops.compressor")

    def forward(self, compressor: Compressor, x: torch.Tensor, attention_masks: Any) -> torch.Tensor:
        plan = attention_masks.plans[compressor.compress_ratio]
        if getattr(attention_masks, "window", None) is not None or plan.exchange is not None:
            raise ValueError("CANN compressor requires a full document stream without context parallelism")
        if plan.gather_indices is None or plan.block_positions is None:
            raise ValueError("CANN compressor requires complete compression metadata")
        if plan.gather_indices.numel() == 0:
            return compressor._forward(x, attention_masks)

        state_block_table, state_cache, cu_seqlens, lengths = _state_inputs(
            attention_masks, x, ratio=compressor.compress_ratio, head_dim=compressor.head_dim
        )
        pooled = torch.ops.cann_ops_transformer.compressor.default(
            x.reshape(-1, x.shape[-1]).contiguous(),
            compressor.wkv.weight,
            compressor.wgate.weight,
            state_cache,
            compressor.ape,
            cmp_ratio=compressor.compress_ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=lengths,
            coff=1 + int(compressor.overlap),
            cache_mode=1,
        )
        pooled = compressor.norm(pooled[: plan.gather_indices.numel() // compressor.compress_ratio].to(x.dtype))
        return (
            compressor.rope(pooled.unsqueeze(0).unsqueeze(2), positions=plan.block_positions.unsqueeze(0))
            .squeeze(0)
            .squeeze(1)
        )
