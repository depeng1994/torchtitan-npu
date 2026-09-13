# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
from torch.fx.subgraph_rewriter import replace_pattern_with_filters

from torchtitan_npu.compile.pattern_manager import setup_patterns
from torchtitan_npu.compile.pattern_replacement import _PRE_AOT_PATTERN_PASS

interleaved_rope = importlib.import_module(
    "torchtitan_npu.compile.patterns.common.interleaved_rope"
)


def _decomposed_rope_arith(x, cos, sin):
    """Canonical decomposed interleaved RoPE fragment."""
    x_float = x.float()
    rotated = torch.stack(
        (-x_float[..., 1::2], x_float[..., ::2]),
        dim=-1,
    ).flatten(-2)
    return (x_float * cos + rotated * sin).type_as(x)


def _complex_reference(x, cos, sin):
    complex_cache = torch.complex(cos[..., ::2], sin[..., ::2])
    rotary = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(rotary * complex_cache).flatten(-2)
    return rotated.type_as(x)


def test_generic_pattern_matches_decomposed_arith():
    pattern = interleaved_rope.PATTERNS["npu_interleaved_rope"]
    graph_module = torch.fx.symbolic_trace(_decomposed_rope_arith)
    matches = replace_pattern_with_filters(
        graph_module,
        pattern.search_fn,
        pattern.replacement_fn,
        ignore_literals=pattern.ignore_literals,
    )
    assert len(matches) == 1


def test_generic_replacement_numerics_match_decomposed(monkeypatch):
    """replacement_fn (npu_rotary_mul path) must stay numerically equivalent."""
    calls = []

    def fake_npu_rotary_mul(x, cos, sin, *, rotary_mode):
        assert rotary_mode == "interleave"
        calls.append(rotary_mode)
        return _decomposed_rope_arith(x, cos, sin)

    monkeypatch.setattr(
        interleaved_rope.torch_npu,
        "npu_rotary_mul",
        fake_npu_rotary_mul,
    )

    pattern = interleaved_rope.PATTERNS["npu_interleaved_rope"]
    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 16, dtype=torch.bfloat16)
    angles = torch.randn(2, 4, 1, 8)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)

    expected = _complex_reference(x, cos, sin)
    actual = pattern.replacement_fn(x, cos, sin)

    torch.testing.assert_close(actual, expected)
    assert calls == ["interleave"]


def test_generic_pattern_missing_module_skipped(monkeypatch):
    """ImportError in an optional pattern module must not break setup."""
    import torchtitan_npu.compile.pattern_manager as pattern_manager

    monkeypatch.setattr(
        pattern_manager,
        "_BUILTIN_PATTERN_MODULES",
        ("no.such.pattern.module", "torchtitan_npu.compile.patterns.common.interleaved_rope"),
    )
    patterns = pattern_manager._discover_builtin_patterns()
    assert "npu_interleaved_rope" in patterns


def test_full_pass_rewrites_generic_rope_graph(monkeypatch):
    """The shared pre-AOT pass consumes the generic pattern too."""
    calls = []

    def fake_npu_rotary_mul(x, cos, sin, *, rotary_mode):
        calls.append(rotary_mode)
        return _decomposed_rope_arith(x, cos, sin)

    monkeypatch.setattr(
        interleaved_rope.torch_npu,
        "npu_rotary_mul",
        fake_npu_rotary_mul,
    )

    saved_patterns = dict(_PRE_AOT_PATTERN_PASS._patterns)
    _PRE_AOT_PATTERN_PASS._patterns.clear()
    setup_patterns(enable_patterns=True)

    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 16, dtype=torch.bfloat16)
    angles = torch.randn(2, 4, 1, 8)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    expected = _complex_reference(x, cos, sin)

    graph_module = torch.fx.symbolic_trace(_decomposed_rope_arith)
    _PRE_AOT_PATTERN_PASS(graph_module.graph)
    graph_module.recompile()

    actual = graph_module(x, cos, sin)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected)

    _PRE_AOT_PATTERN_PASS._patterns.clear()
    _PRE_AOT_PATTERN_PASS._patterns.update(saved_patterns)
