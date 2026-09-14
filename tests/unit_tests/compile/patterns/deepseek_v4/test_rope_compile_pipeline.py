# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end baseline for the DeepSeek-V4 partial-RoPE compile pipeline.

This test deliberately goes through the shared pre-AOT graph pass via the real
``pattern_manager.setup_patterns()`` instead of calling a PatternReplacement
replacement_fn directly.  It locks the behavior that the compiler-friendly
interleaved RoPE representation can be consumed by the DeepSeek-V4 partial-RoPE
pattern without changing BF16 numerics.
"""

import importlib

import pytest
import torch
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan_npu.compile.pattern_manager import setup_patterns
from torchtitan_npu.compile.pattern_replacement import (
    _PRE_AOT_PATTERN_PASS,
)
from torchtitan_npu.override.common.rope import DecomposedComplexRoPE

partial_interleaved_rope = importlib.import_module(
    "torchtitan_npu.compile.patterns.common.partial_interleaved_rope"
)
interleaved_rope = importlib.import_module(
    "torchtitan_npu.compile.patterns.common.interleaved_rope"
)


def _patch_inplace_rotary(monkeypatch, fake_op):
    """Patch ``inplace_partial_rotary_mul`` in the enclosing module."""
    monkeypatch.setattr(partial_interleaved_rope, "inplace_partial_rotary_mul", fake_op)


class _PartialRoPEModule(torch.nn.Module):
    """Minimal DeepSeek-V4 split -> RoPE -> cat boundary."""

    def __init__(self, *, inverse: bool) -> None:
        super().__init__()
        self.inverse = inverse

    def forward(self, x, cos, sin):
        prefix, rotary = torch.split(x, [8, 8], dim=-1)
        rotary = DecomposedComplexRoPE.apply_rotary_emb(
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
    x, cos, sin, *, rotary_mode, partial_slice
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


def _decomposed_rope_arith(x, cos, sin):
    """Canonical decomposed interleaved RoPE (used by generic pattern)."""
    x_float = x.float()
    rotated = torch.stack(
        (-x_float[..., 1::2], x_float[..., ::2]),
        dim=-1,
    ).flatten(-2)
    return (x_float * cos + rotated * sin).type_as(x)


@pytest.fixture(autouse=True)
def reset_pass(monkeypatch):
    """Clean the shared pass before and after each test."""
    saved = dict(_PRE_AOT_PATTERN_PASS._patterns)
    _PRE_AOT_PATTERN_PASS._patterns.clear()
    yield
    _PRE_AOT_PATTERN_PASS._patterns.clear()
    _PRE_AOT_PATTERN_PASS._patterns.update(saved)


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_partial_rope_via_real_manager(monkeypatch, inverse):
    """Baseline: DSV4 partial pattern via real setup_patterns()."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 16, dtype=torch.bfloat16)
    angles = torch.randn(2, 4, 1, 4)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)

    expected = _complex_reference(x, cos, sin, inverse=inverse)

    module = _PartialRoPEModule(inverse=inverse)
    graph_module = torch.fx.symbolic_trace(module)

    # Phase A: canonicalization matches upstream ComplexRoPE
    before_rewrite = graph_module(x, cos, sin)
    torch.testing.assert_close(before_rewrite, expected)

    calls = 0

    def fake_op(x, cos, sin, *, rotary_mode, partial_slice):
        nonlocal calls
        calls += 1
        _fake_inplace_partial_rotary_mul(
            x, cos, sin, rotary_mode=rotary_mode, partial_slice=partial_slice
        )

    _patch_inplace_rotary(monkeypatch, fake_op)

    # Phase B: Register through the real manager (not register_pre_aot_patterns)
    setup_patterns(enable_patterns=True)

    # Phase C: Execute shared pass and verify replacement
    _PRE_AOT_PATTERN_PASS(graph_module.graph)
    graph_module.recompile()

    actual = graph_module(x, cos, sin)

    assert calls == 1, "DSV4 partial pattern must be matched exactly once"
    torch.testing.assert_close(actual, expected)


def test_partial_blacklisted_generic_takes_over(monkeypatch):
    """Blacklist DSV4 partial patterns; generic interleaved must consume the graph."""
    partial_calls = 0
    generic_calls = 0

    def fake_inplace_partial_rotary_mul(x, cos, sin, *, rotary_mode, partial_slice):
        nonlocal partial_calls
        partial_calls += 1

    def fake_npu_rotary_mul(x, cos, sin, *, rotary_mode):
        nonlocal generic_calls
        generic_calls += 1
        assert rotary_mode == "interleave"
        return _decomposed_rope_arith(x, cos, sin)

    _patch_inplace_rotary(monkeypatch, fake_inplace_partial_rotary_mul)
    monkeypatch.setattr(
        interleaved_rope.torch_npu,
        "npu_rotary_mul",
        fake_npu_rotary_mul,
    )

    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 16, dtype=torch.bfloat16)
    angles = torch.randn(2, 4, 1, 4)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    expected = _complex_reference(x, cos, sin, inverse=False)

    # Register ONLY the generic pattern (blacklist all DSV4 patterns)
    setup_patterns(
        enable_patterns=True,
        pattern_blacklist=(
            "partial_rope_wo_squeeze_forward",
            "partial_rope_wo_squeeze_inverse",
            "partial_rope_attention_kv_forward",
            "partial_rope_compressor_kv_forward",
        ),
    )

    # Same DSV4 split -> rope -> cat graph as the specific-pattern test
    graph_module = torch.fx.symbolic_trace(_PartialRoPEModule(inverse=False))
    _PRE_AOT_PATTERN_PASS(graph_module.graph)
    graph_module.recompile()

    actual = graph_module(x, cos, sin)

    assert partial_calls == 0, "DSV4 partial patterns must be blacklisted"
    assert generic_calls == 1, "generic pattern must take over the rotary fragment"
    torch.testing.assert_close(actual, expected)


def test_all_patterns_disabled_keeps_decomposed_graph(monkeypatch):
    """Disable all patterns; graph must remain unchanged decomposed Torch."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 16, dtype=torch.bfloat16)
    angles = torch.randn(2, 4, 1, 8)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    expected = _decomposed_rope_arith(x, cos, sin)

    calls = 0

    def never_called(*args, **kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        interleaved_rope.torch_npu, "npu_rotary_mul", never_called,
    )
    _patch_inplace_rotary(monkeypatch, never_called)

    setup_patterns(enable_patterns=False)
    graph_module = torch.fx.symbolic_trace(_decomposed_rope_arith)
    _PRE_AOT_PATTERN_PASS(graph_module.graph)
    graph_module.recompile()

    actual = graph_module(x, cos, sin)

    assert calls == 0, "no replacement should occur when patterns are disabled"
    torch.testing.assert_close(actual, expected)
