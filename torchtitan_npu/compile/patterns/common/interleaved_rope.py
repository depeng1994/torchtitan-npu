# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic interleaved RoPE pattern for decomposed RoPE arithmetic.

Matches the decomposed (real-valued interleaved) RoPE subgraph and replaces it
with ``torch_npu.npu_rotary_mul(..., rotary_mode="interleave")``.

This pattern runs *after* the model-specific DSV4 partial-RoPE patterns so that
fragments already consumed by ``inplace_partial_rotary_mul`` are not matched
again.
"""

from __future__ import annotations

import torch
import torch_npu

from torchtitan_npu.compile.pattern_replacement import PatternReplacement

if torch_npu.npu.is_available():
    import torch_npu._inductor


def _search_interleaved_rope(x, cos, sin):
    """Match the canonical decomposed interleaved RoPE fragment."""
    x_float = x.float()
    rotated = torch.stack(
        (-x_float[..., 1::2], x_float[..., ::2]),
        dim=-1,
    ).flatten(-2)
    return (x_float * cos + rotated * sin).type_as(x)


def _replace_interleaved_rope(x, cos, sin):
    """Replace with a single npu_rotary_mul call."""
    return torch_npu.npu_rotary_mul(x.float(), cos, sin, rotary_mode="interleave").type_as(x)


PATTERNS: dict[str, PatternReplacement] = {
    "npu_interleaved_rope": PatternReplacement(
        search_fn=_search_interleaved_rope,
        replacement_fn=_replace_interleaved_rope,
    ),
}
