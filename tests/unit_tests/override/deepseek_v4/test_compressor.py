# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU oracle at the CANN boundary, with real configs and document metadata."""

import copy
import importlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4 import _debugmodel
from torchtitan_npu.models.deepseek_v4.compressor import Compressor, CompressorImplementation
from torchtitan_npu.models.deepseek_v4.metadata import build_compressed_varlen_metadata
from torchtitan_npu.override.deepseek_v4.compressor import ascendc


@pytest.mark.parametrize("reverse", [False, True], ids=["compressor-first", "compressor-last"])
def test_compressor_override_composes_with_norm_and_rope(monkeypatch, reverse):
    from torchtitan_npu.override.common.rms_norm import AscRMSNorm
    from torchtitan_npu.override.common.rope import AscComplexRoPE

    registry = importlib.import_module("torchtitan.config.override")
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
    cfg = _debugmodel()
    imports = [
        "torchtitan_npu.override.deepseek_v4.compressor.asc",
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.common.rope.asc_complex",
    ]
    apply_overrides(OverrideConfig(imports=imports[::-1] if reverse else imports), cfg)
    compressors = [node for _, node, _, _ in cfg.traverse(Compressor.Config)]
    assert compressors
    for node in compressors:
        assert type(node) is Compressor.Config
        assert isinstance(node.implementation, ascendc.AscCompressor.Config)
        assert isinstance(node.norm, AscRMSNorm.Config)
        assert isinstance(node.rope, AscComplexRoPE.Config)
    assert all(
        type(node.implementation) is CompressorImplementation.Config
        for _, node, _, _ in _debugmodel().traverse(Compressor.Config)
    )


def _document_pool(x, wkv, wgate, ape, cu, ratio):
    outputs = []
    width = wkv.shape[0] // (2 if ratio == 4 else 1)
    for start, end in zip(cu[:-1], cu[1:]):
        values = F.linear(x[start:end].float(), wkv.float())
        scores = F.linear(x[start:end].float(), wgate.float())
        for offset in range(0, (end - start) // ratio * ratio, ratio):
            current_values = values[offset:offset + ratio, -width:]
            current_scores = scores[offset:offset + ratio, -width:] + ape[:, -width:]
            if ratio == 4 and offset:
                current_values = torch.cat((values[offset - ratio:offset, :width], current_values))
                current_scores = torch.cat((scores[offset - ratio:offset, :width] + ape[:, :width], current_scores))
            outputs.append((current_values * current_scores.softmax(0)).sum(0))
    return torch.stack(outputs)


def _postprocess(module, pooled, positions):
    normalized = F.rms_norm(pooled, (module.head_dim,), module.norm.weight, module.norm.eps)
    return module.rope(normalized[None, :, None, :], positions=positions[None, :])[0, :, 0, :]


@pytest.fixture
def compressor_factory(monkeypatch):
    registry = importlib.import_module("torchtitan.config.override")
    with monkeypatch.context() as scoped, torch.random.fork_rng():
        scoped.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
        scoped.setitem(sys.modules, "cann_ops_transformer.ops.compressor", types.ModuleType("compressor"))
        torch.manual_seed(93)

        def build(ratio, indexer=False):
            cfg = _debugmodel()
            apply_overrides(OverrideConfig(imports=["torchtitan_npu.override.deepseek_v4.compressor.asc"]), cfg)
            attention = cfg.layers[2 if ratio == 4 else 3].attention
            compressor_cfg = attention.indexer.compressor if indexer else attention.compressor
            compressor_cfg.wkv.in_features = 1024
            compressor_cfg.wgate.in_features = 1024
            module = compressor_cfg.build()
            assert isinstance(module.implementation, ascendc.AscCompressor)
            module.init_states(buffer_device=torch.device("cpu"))
            module.wkv.to(torch.float16)
            module.wgate.to(torch.float16)
            module.norm.to(torch.float16)
            return module

        yield build


@pytest.mark.parametrize("ratio,indexer", [(4, False), (4, True), (128, False)], ids=["c4-attention", "c4-indexer", "c128"])
def test_compressor_cann_schema_document_outputs_and_gradients(compressor_factory, monkeypatch, ratio, indexer):
    module = compressor_factory(ratio, indexer)
    module.wgate.weight.requires_grad_(not indexer)
    reference = copy.deepcopy(module)
    lengths = [2 * ratio + 1, ratio - 1, ratio + 2]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    boundaries = torch.tensor(cu, dtype=torch.int32)
    metadata = build_compressed_varlen_metadata(
        VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=max(lengths), max_k=max(lengths)), (ratio,)
    )
    generator = torch.Generator().manual_seed(21)
    x = (torch.randn(1, cu[-1], module.wkv.in_features, generator=generator) * 0.1).half().requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    calls = []
    perturbation = torch.linspace(-0.02, 0.03, module.head_dim)

    def fake_cann(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *, state_block_table=None,
                  cu_seqlens=None, seqused=None, start_pos=None, coff=1, cache_mode=1):
        assert torch.is_grad_enabled()
        calls.append(cmp_ratio)
        assert x.dtype == wkv.dtype == wgate.dtype == torch.float16
        assert ape.dtype == state_cache.dtype == torch.float32
        assert x.shape[1] == wkv.shape[1] == wgate.shape[1] == 1024
        assert wkv is module.wkv.weight and wgate is module.wgate.weight
        assert ape is module.ape
        block_size = 8 if cmp_ratio == 4 else 16
        assert state_cache.shape == (1, block_size, 2 * coff * module.head_dim)
        assert state_block_table.shape == (len(lengths), (max(lengths) + block_size - 1) // block_size)
        assert not state_block_table.any() and start_pos is None and cache_mode == 1
        torch.testing.assert_close(cu_seqlens, boundaries)
        torch.testing.assert_close(seqused, torch.tensor(lengths, dtype=torch.int32))
        pooled = (2 * _document_pool(x, wkv, wgate, ape, cu, cmp_ratio) + perturbation).to(x.dtype)
        # Poison unused TND capacity so consuming it fails output comparisons.
        return torch.cat((pooled, pooled.new_full((len(lengths), module.head_dim), float("nan"))))

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer, "compressor", types.SimpleNamespace(default=fake_cann), raising=False
    )

    actual = module(x, metadata)
    raw = _document_pool(reference_x[0], reference.wkv.weight, reference.wgate.weight, reference.ape, cu, ratio)
    expected = _postprocess(reference, (2 * raw + perturbation).half(), metadata.plans[ratio].block_positions)
    upstream_grad = torch.randn(actual.shape, generator=generator).half()
    actual.backward(upstream_grad)
    expected.backward(upstream_grad)

    assert calls == [ratio]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0.006, atol=0.004)
    for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
        actual_grad = module.get_parameter(name).grad
        expected_grad = reference.get_parameter(name).grad
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.006, atol=0.004)


@pytest.mark.parametrize("reason", ["empty", "cp"], ids=str)
def test_compressor_empty_plan_and_cp(compressor_factory, reason):
    module = compressor_factory(4)
    length = 3 if reason == "empty" else 8
    boundaries = torch.tensor([0, length], dtype=torch.int32)
    metadata = build_compressed_varlen_metadata(
        VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=length, max_k=length), (4,)
    )
    if reason == "cp":
        # CP may have no exchange on a rank; window presence still forces fallback.
        metadata.window = object()
    x = torch.ones(1, length, module.wkv.in_features, dtype=torch.float16, requires_grad=True)
    if reason == "cp":
        with pytest.raises(ValueError, match="without context parallelism"):
            module(x, metadata)
        return
    expected = module._forward(x, metadata)
    actual = module(x, metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (0, module.head_dim)
    actual.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)
