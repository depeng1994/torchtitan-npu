# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end baseline for the DeepSeek-V4 partial-RoPE compile pipeline.

This test deliberately goes through the shared pre-AOT graph pass instead of
calling a PatternReplacement replacement_fn directly.  It locks the behavior
that the compiler-friendly interleaved RoPE representation can be consumed by
the DeepSeek-V4 partial-RoPE pattern without changing BF16 numerics.
"""

import importlib

import pytest
import torch
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan_npu.compile import pattern_replacement
from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE

inplace_partial_rope = importlib.import_module(
    "torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope"
)


class _PartialRoPEModule(torch.nn.Module):
    """Minimal DeepSeek-V4 split -> RoPE -> cat boundary."""

    def __init__(self, *, inverse: bool) -> None:
        super().__init__()
        self.inverse = inverse

    def forward(self, x, cos, sin):
        prefix, rotary = torch.split(x, [8, 8], dim=-1)
        rotary = WorkaroundComplexRoPE.apply_rotary_emb(
            rotary,
            None,
            (cos, sin),
            inverse=self.inverse,
        )
        return torch.cat([prefix, rotary], dim=-1)


def _complex_reference(x, cos, sin, *, inverse: bool):
    prefix, rotary = torch.split(x, [8, 8], dim=-1)
    complex_cache = torch.complex(cos[..., ::2], sin[..., ::2])
    rotary = ComplexRoPE.apply_rotary_emb(
        rotary,
        None,
        complex_cache,
        inverse=inverse,
    )
    return torch.cat([prefix, rotary], dim=-1)


def _fake_inplace_partial_rotary_mul(
    x,
    cos,
    sin,
    *,
    rotary_mode,
    partial_slice,
):
    assert rotary_mode == "interleave"
    start, end = partial_slice
    rotary = x[..., start:end]
    rotary_float = rotary.float()
    rotated = torch.stack(
        (-rotary_float[..., 1::2], rotary_float[..., ::2]),
        dim=-1,
    ).flatten(-2)
    rotary.copy_((rotary_float * cos + rotated * sin).type_as(rotary))


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_partial_rope_shared_pre_aot_pass_matches_complex_reference(
    monkeypatch,
    inverse,
):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 16, dtype=torch.bfloat16)
    angles = torch.randn(2, 4, 1, 4)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)

    expected = _complex_reference(x, cos, sin, inverse=inverse)

    module = _PartialRoPEModule(inverse=inverse)
    graph_module = torch.fx.symbolic_trace(module)

    # First lock the compiler-friendly decomposed/interleaved representation
    # itself against upstream ComplexRoPE semantics.
    before_rewrite = graph_module(x, cos, sin)
    torch.testing.assert_close(before_rewrite, expected)

    calls = 0

    def fake_op(x, cos, sin, *, rotary_mode, partial_slice):
        nonlocal calls
        calls += 1
        _fake_inplace_partial_rotary_mul(
            x,
            cos,
            sin,
            rotary_mode=rotary_mode,
            partial_slice=partial_slice,
        )

    monkeypatch.setattr(
        inplace_partial_rope,
        "inplace_partial_rotary_mul",
        fake_op,
    )

    # Execute the real shared pass used by Inductor pre_grad_custom_pass.  A
    # zero-hit rewrite leaves the graph numerically correct, so the fake fused
    # op call count is also required as the structural oracle.
    pattern_replacement._PRE_AOT_PATTERN_PASS(graph_module.graph)
    graph_module.recompile()

    actual = graph_module(x, cos, sin)

    assert calls == 1
    torch.testing.assert_close(actual, expected)
