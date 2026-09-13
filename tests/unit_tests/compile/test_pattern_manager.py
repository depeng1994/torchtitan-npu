# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
from torch._inductor.custom_graph_pass import get_custom_graph_passes

from torchtitan_npu.compile import pattern_manager
from torchtitan_npu.compile.pattern_replacement import (
    PatternReplacement,
    _PRE_AOT_PATTERN_PASS,
)


def _pattern(name):
    return PatternReplacement(
        search_fn=lambda x: torch.relu(x),
        replacement_fn=lambda x: torch.sigmoid(x),
    )


@pytest.fixture(autouse=True)
def isolated_pass(monkeypatch):
    """Give every test an empty shared pass and a clean Inductor config."""
    import torchtitan_npu.compile.pattern_replacement as pr

    pass_instance = type(_PRE_AOT_PATTERN_PASS)()
    monkeypatch.setattr(pr, "_PRE_AOT_PATTERN_PASS", pass_instance)
    with torch._inductor.config.patch(pre_grad_custom_pass=None):
        yield pass_instance
    torch._dynamo.reset()


def test_setup_patterns_registers_builtin_patterns(isolated_pass):
    pattern_manager.setup_patterns(enable_patterns=True)

    assert "partial_rope_wo_squeeze_forward" in isolated_pass._patterns
    assert "npu_interleaved_rope" in isolated_pass._patterns
    installed = get_custom_graph_passes(torch._inductor.config.pre_grad_custom_pass)
    assert isolated_pass in installed


def test_setup_patterns_disabled_registers_nothing(isolated_pass):
    pattern_manager.setup_patterns(enable_patterns=False)

    assert isolated_pass._patterns == {}


def test_setup_patterns_blacklist_skips_named_pattern(isolated_pass):
    pattern_manager.setup_patterns(
        enable_patterns=True,
        pattern_blacklist=("partial_rope_wo_squeeze_forward",),
    )

    assert "partial_rope_wo_squeeze_forward" not in isolated_pass._patterns
    assert "npu_interleaved_rope" in isolated_pass._patterns


def test_setup_patterns_blacklist_all_keeps_decomposed_graph(isolated_pass):
    pattern_manager.setup_patterns(
        enable_patterns=True,
        pattern_blacklist=(
            "partial_rope_wo_squeeze_forward",
            "partial_rope_wo_squeeze_inverse",
            "dsv4_partial_rope_attention_kv_forward",
            "dsv4_partial_rope_compressor_kv_forward",
            "npu_interleaved_rope",
        ),
    )

    assert isolated_pass._patterns == {}


def test_setup_patterns_idempotent_repeated_call(isolated_pass):
    pattern_manager.setup_patterns(enable_patterns=True)
    first_patterns = dict(isolated_pass._patterns)
    installed_after_first = get_custom_graph_passes(torch._inductor.config.pre_grad_custom_pass)

    pattern_manager.setup_patterns(enable_patterns=True)

    assert isolated_pass._patterns == first_patterns
    installed_after_second = get_custom_graph_passes(torch._inductor.config.pre_grad_custom_pass)
    assert installed_after_second.count(isolated_pass) == installed_after_first.count(isolated_pass)


def test_setup_patterns_policy_transitions_replace_patterns(isolated_pass):
    """Policy change must fully replace the pattern set (no additive residue)."""
    all_names = {
        "partial_rope_wo_squeeze_forward",
        "partial_rope_wo_squeeze_inverse",
        "dsv4_partial_rope_attention_kv_forward",
        "dsv4_partial_rope_compressor_kv_forward",
        "npu_interleaved_rope",
    }
    dsv4_names = all_names - {"npu_interleaved_rope"}

    # all -> disabled
    pattern_manager.setup_patterns(enable_patterns=True)
    assert set(isolated_pass._patterns) == all_names
    pattern_manager.setup_patterns(enable_patterns=False)
    assert isolated_pass._patterns == {}

    # disabled -> blacklist (DSV4 patterns only)
    pattern_manager.setup_patterns(
        enable_patterns=True,
        pattern_blacklist=tuple(sorted(dsv4_names)),
    )
    assert set(isolated_pass._patterns) == {"npu_interleaved_rope"}

    # blacklist -> all
    pattern_manager.setup_patterns(enable_patterns=True)
    assert set(isolated_pass._patterns) == all_names


def test_discover_skips_missing_module(monkeypatch, isolated_pass):
    monkeypatch.setattr(
        pattern_manager,
        "_BUILTIN_PATTERN_MODULES",
        ("torchtitan_npu.compile.patterns.common.interleaved_rope", "no.such.module"),
    )

    patterns = pattern_manager._discover_builtin_patterns()

    assert "npu_interleaved_rope" in patterns


def test_discover_rejects_duplicate_pattern_names(monkeypatch, isolated_pass):
    """Two modules exporting the same pattern name must raise, not silently shadow."""
    dup_module = importlib.import_module(
        "torchtitan_npu.compile.patterns.common.interleaved_rope"
    )
    monkeypatch.setattr(
        pattern_manager,
        "_BUILTIN_PATTERN_MODULES",
        (
            "torchtitan_npu.compile.patterns.common.interleaved_rope",
            "torchtitan_npu.compile.patterns.common.interleaved_rope",
        ),
    )

    with pytest.raises(ValueError, match="Duplicate NPU pattern name"):
        pattern_manager._discover_builtin_patterns()

    assert dup_module  # keep reference to avoid unused-import lint


def test_common_pattern_module_exports_named_patterns():
    module = importlib.import_module("torchtitan_npu.compile.patterns.common.interleaved_rope")

    assert "npu_interleaved_rope" in module.PATTERNS
    assert module.PATTERNS["npu_interleaved_rope"].replacement_fn is not None


def test_common_partial_interleaved_module_exports_named_patterns():
    module = importlib.import_module(
        "torchtitan_npu.compile.patterns.common.partial_interleaved_rope"
    )

    assert set(module.PATTERNS) == {
        "partial_rope_wo_squeeze_forward",
        "partial_rope_wo_squeeze_inverse",
    }


def test_dsv4_pattern_module_exports_named_patterns():
    module = importlib.import_module(
        "torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope"
    )

    assert set(module.PATTERNS) == {
        "dsv4_partial_rope_attention_kv_forward",
        "dsv4_partial_rope_compressor_kv_forward",
    }
