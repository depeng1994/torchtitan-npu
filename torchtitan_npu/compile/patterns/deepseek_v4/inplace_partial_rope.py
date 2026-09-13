# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replace DeepSeek-V4 split/RoPE/cat regions before AOTAutograd.

This module exports a ``PATTERNS`` dict.  It does NOT register itself at
import time — use ``pattern_manager.setup_patterns()`` for registration.
"""

from __future__ import annotations

import torch
import torch_npu

from torchtitan_npu.compile.pattern_replacement import PatternReplacement
from torchtitan_npu.ops.ascendc.inplace_partial_rotary_mul import (
    inplace_partial_rotary_mul,
)

if torch_npu.npu.is_available():
    import torch_npu._inductor


torch.fx.wrap("inplace_partial_rotary_mul")


def _rotate_interleaved(x):
    """Interleave rotation used by the deepseek-v4 partial-rope fragment."""
    return torch.stack(
        (-x[..., 1::2], x[..., ::2]),
        dim=-1,
    ).flatten(-2)


def _make_partial_rope_pattern(
    *,
    inverse: bool,
    unsqueeze_dims: tuple[int, ...],
    squeeze_dims: tuple[int, ...],
) -> PatternReplacement:
    """Build a shape-agnostic interleaved partial-RoPE pattern."""

    def search_fn(x, cos, sin):
        # Shape literals are placeholders generalized by ignore_literals=True.
        prefix, rotary = torch.split(x, [2, 2], dim=-1)
        rotary_u = rotary
        for dim in unsqueeze_dims:
            rotary_u = rotary_u.unsqueeze(dim)
        rotary_float = rotary_u.float()
        rotated = _rotate_interleaved(rotary_float)
        if inverse:
            sin = -sin
        rotated = rotary_float * cos + rotated * sin
        rotated = rotated.type_as(rotary_u)
        for dim in squeeze_dims:
            rotated = rotated.squeeze(dim)
        return torch.cat([prefix, rotated], dim=-1)

    def replacement_fn(x, cos, sin):
        if inverse:
            sin = -sin
        end = x.shape[-1]
        output = x.clone()
        for dim in unsqueeze_dims:
            output = output.unsqueeze(dim)
        inplace_partial_rotary_mul(
            output,
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[end - cos.shape[-1], end],
        )
        for dim in squeeze_dims:
            output = output.squeeze(dim)
        return output

    return PatternReplacement(
        search_fn=search_fn,
        replacement_fn=replacement_fn,
        ignore_literals=True,
    )


def _make_parent_rope_pattern(
    *,
    inverse: bool,
) -> PatternReplacement:
    return _make_partial_rope_pattern(
        inverse=inverse,
        unsqueeze_dims=(),
        squeeze_dims=(),
    )


def _make_kv_rope_pattern() -> PatternReplacement:
    return _make_partial_rope_pattern(
        inverse=False,
        unsqueeze_dims=(2,),
        squeeze_dims=(2,),
    )


def _make_compressor_rope_pattern() -> PatternReplacement:
    """Match compressor RoPE after its input shape is already materialized.

    The broadcast-cache helper also reads the dynamic query shape. Keep that
    producer-side metadata outside the matched subgraph so extra size users do
    not make the pattern fail containment checks.
    """

    def search_fn(prefix, rotary_u, cos, sin):
        rotary_float = rotary_u.float()
        rotated = _rotate_interleaved(rotary_float)
        rotated = (rotary_float * cos + rotated * sin).type_as(rotary_u)
        rotated = rotated.squeeze(0).squeeze(1)
        return torch.cat([prefix, rotated], dim=-1)

    def replacement_fn(prefix, rotary_u, cos, sin):
        output = rotary_u.clone()
        inplace_partial_rotary_mul(
            output,
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[0, cos.shape[-1]],
        )
        output = output.squeeze(0).squeeze(1)
        return torch.cat([prefix, output], dim=-1)

    return PatternReplacement(
        search_fn=search_fn,
        replacement_fn=replacement_fn,
    )


PATTERNS: dict[str, PatternReplacement] = {
    "dsv4_partial_rope_wo_squeeze_inverse": _make_parent_rope_pattern(inverse=True),
    "dsv4_partial_rope_wo_squeeze_forward": _make_parent_rope_pattern(inverse=False),
    "dsv4_partial_rope_attention_kv_forward": _make_kv_rope_pattern(),
    "dsv4_partial_rope_compressor_kv_forward": _make_compressor_rope_pattern(),
}
