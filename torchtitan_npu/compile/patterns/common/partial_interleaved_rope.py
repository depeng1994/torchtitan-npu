# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic interleaved partial-RoPE patterns (model-agnostic).

Matches the canonical ``split -> decomposed interleaved RoPE -> cat`` fragment
that appears in any model using partial RoPE (e.g. DeepSeek-V3 Q and
DeepSeek-V4), and replaces it with ``inplace_partial_rotary_mul``.

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


def rotate_interleaved(x):
    """Interleave rotation used by partial-RoPE fragments."""
    return torch.stack(
        (-x[..., 1::2], x[..., ::2]),
        dim=-1,
    ).flatten(-2)


def make_partial_rope_pattern(
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
        rotated = rotate_interleaved(rotary_float)
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


PATTERNS: dict[str, PatternReplacement] = {
    "partial_rope_wo_squeeze_forward": make_partial_rope_pattern(
        inverse=False,
        unsqueeze_dims=(),
        squeeze_dims=(),
    ),
    "partial_rope_wo_squeeze_inverse": make_partial_rope_pattern(
        inverse=True,
        unsqueeze_dims=(),
        squeeze_dims=(),
    ),
}
