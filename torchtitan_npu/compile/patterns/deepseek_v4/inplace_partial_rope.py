# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replace DeepSeek-V4 split/RoPE/cat regions before AOTAutograd.

This module exports a ``PATTERNS`` dict for model-specific patterns only:
attention-KV and compressor RoPE fragments that require unsqueeze/squeeze
shape handling.  The generic parent ``partial_rope_wo_squeeze_*`` patterns
live in ``torchtitan_npu.compile.patterns.common.partial_interleaved_rope``.
"""

from __future__ import annotations

import torch
import torch_npu

from torchtitan_npu.compile.pattern_replacement import PatternReplacement
from torchtitan_npu.compile.patterns.common.partial_interleaved_rope import (
    make_partial_rope_pattern,
    rotate_interleaved,
)
from torchtitan_npu.ops.ascendc.inplace_partial_rotary_mul import (
    inplace_partial_rotary_mul,
)

if torch_npu.npu.is_available():
    import torch_npu._inductor


torch.fx.wrap("inplace_partial_rotary_mul")


def _make_kv_rope_pattern() -> PatternReplacement:
    return make_partial_rope_pattern(
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
        rotated = rotate_interleaved(rotary_float)
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
    "dsv4_partial_rope_attention_kv_forward": _make_kv_rope_pattern(),
    "dsv4_partial_rope_compressor_kv_forward": _make_compressor_rope_pattern(),
}
