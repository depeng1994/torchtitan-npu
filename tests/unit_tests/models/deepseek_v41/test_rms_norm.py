# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides
from torchtitan.models.common.nn_modules import RMSNorm

from torchtitan_npu.models.deepseek_v41.model_registry import model_registry
from torchtitan_npu.models.deepseek_v41.rms_norm import V41RMSNorm
from torchtitan_npu.models.deepseek_v41.vision import DeepSeekV41VisionEncoder
from torchtitan_npu.override.deepseek_v41.rms_norm import AscV41RMSNorm

OVERRIDE = "torchtitan_npu.override.deepseek_v41.rms_norm.ascendc"


@pytest.mark.parametrize("fp32", [False, True], ids=["native", "reference-fp32"])
def test_reference_forward_backward_preserves_arithmetic(fp32):
    cfg = V41RMSNorm.Config(normalized_shape=8, eps=1e-6, reference_fp32=fp32)
    norm = cfg.build().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
    x = torch.linspace(-2, 3, 24).reshape(3, 8).to(torch.bfloat16).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    ref_w = norm.weight.detach().clone().requires_grad_()
    if fp32:
        values = ref_x.float()
        expected = (ref_w * (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6))).to(ref_x.dtype)
    else:
        expected = torch.nn.functional.rms_norm(ref_x, (8,), ref_w, 1e-6)
    actual = norm(x)
    grad = torch.linspace(-1, 1, 24).reshape(3, 8).to(x.dtype)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in [(actual, expected), (x.grad, ref_x.grad), (norm.weight.grad, ref_w.grad)]:
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_override_reaches_vision_and_preserves_parameter_names():
    cfg = DeepSeekV41VisionEncoder.Config(dim=8, num_layers=2, num_heads=2, inter_dim=16, text_dim=8)
    ref = cfg.build()
    fused_cfg = copy.deepcopy(cfg)
    apply_overrides(OverrideConfig(imports=[OVERRIDE]), fused_cfg)
    fused = fused_cfg.build()
    norms = [m for m in fused.modules() if isinstance(m, V41RMSNorm)]
    assert len(norms) == 5
    assert all(isinstance(m, AscV41RMSNorm) and m.reference_fp32 for m in norms)
    assert ref.state_dict().keys() == fused.state_dict().keys()
    fused.load_state_dict(ref.state_dict(), strict=True)
    common = RMSNorm.Config(normalized_shape=8)
    apply_overrides(OverrideConfig(imports=[OVERRIDE]), common)
    assert type(common) is RMSNorm.Config


def test_model_config_selects_both_rms_contracts():
    cfg = model_registry("deepseek_v41_debugmodel").model
    apply_overrides(OverrideConfig(imports=[OVERRIDE]), cfg)
    assert isinstance(cfg.norm, AscV41RMSNorm.Config)
    assert not cfg.layers[0].attention.q_norm.reference_fp32
    source = cfg.layers[2].attention
    assert isinstance(source.compressor.norm, AscV41RMSNorm.Config)
    assert source.compressor.norm.reference_fp32
    assert isinstance(source.indexer.k_norm, AscV41RMSNorm.Config)
    assert not source.indexer.k_norm.reference_fp32
    assert not cfg.layers[20].attention.compressor.norm.reference_fp32
