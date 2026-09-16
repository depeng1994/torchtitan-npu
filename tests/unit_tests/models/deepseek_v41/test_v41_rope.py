# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides

from torchtitan_npu.models.deepseek_v41.compressor import _apply_complex_rope
from torchtitan_npu.models.deepseek_v41.model_registry import model_registry
from torchtitan_npu.models.deepseek_v41.rope import V41RoPERotation
from torchtitan_npu.models.deepseek_v41.vision import DeepSeekV41VisionEncoder
from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE
from torchtitan_npu.override.deepseek_v41.rope import AscV41RoPERotation


@pytest.mark.parametrize("mode", ["interleave", "complex", "half"])
@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_quarter_turn_and_gradient_on_partial_noncontiguous_input(mode, inverse, dtype):
    full = torch.arange(48, dtype=dtype).reshape(2, 3, 1, 8).requires_grad_()
    x = full[..., 4:]
    assert not x.is_contiguous()
    width = 2 if mode == "half" else 4
    cos = torch.zeros(2, 3, 1, width)
    # Different rotation direction in each batch must be preserved.
    sin = torch.ones_like(cos)
    sin[1] = -1
    actual = V41RoPERotation.Config(mode=mode).build()(x, cos, sin, inverse=inverse)
    real_ids, imag_ids = ([0, 1], [2, 3]) if mode == "half" else ([0, 2], [1, 3])
    expected = torch.empty_like(x)
    sign = torch.tensor([1, -1], dtype=dtype).view(2, 1, 1, 1) * (-1 if inverse else 1)
    expected[..., real_ids] = -x[..., imag_ids] * sign
    expected[..., imag_ids] = x[..., real_ids] * sign
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.arange(1, 25, dtype=dtype).reshape_as(x)
    actual.backward(grad)
    expected_grad = torch.zeros_like(full)
    expected_grad[..., [4 + i for i in real_ids]] = grad[..., imag_ids] * sign
    expected_grad[..., [4 + i for i in imag_ids]] = -grad[..., real_ids] * sign
    torch.testing.assert_close(full.grad, expected_grad, rtol=0, atol=0)


def test_compressor_preserves_per_batch_positions():
    rope = WorkaroundComplexRoPE.Config(dim=4, max_seq_len=8).build()
    rotary = V41RoPERotation.Config(mode="complex").build()
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 4)
    positions = torch.tensor([[0, 1, 2], [3, 1, 0]])
    actual = _apply_complex_rope(rope, rotary, x, positions)
    expected = torch.cat([_apply_complex_rope(rope, rotary, x[i : i + 1], positions[i : i + 1]) for i in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["interleave", "complex", "half"])
@pytest.mark.parametrize("unbatched", [False, True], ids=["batched", "unbatched"])
def test_fused_adapter_preserves_fp32_cache_and_batches(monkeypatch, mode, unbatched):
    import torchtitan_npu.override.deepseek_v41.rope as fused

    calls = []

    def kernel(x, cos, sin, *, rotary_mode):
        calls.append((x, cos, sin, rotary_mode))
        return x + 2

    monkeypatch.setattr(fused.torch_npu, "npu_rotary_mul", kernel)
    x = torch.ones(2, 3, 1, 4, dtype=torch.bfloat16)
    width = 2 if mode == "half" else 4
    cos = torch.linspace(0.1, 0.9, 2 * 3 * width).reshape(2, 3, 1, width)
    sin = cos.flip(1)
    if unbatched:
        x, cos, sin = x[0], cos[0], sin[0]
    actual = AscV41RoPERotation.Config(mode=mode).build()(x, cos, sin, inverse=True)
    value, sent_cos, sent_sin, rotary_mode = calls[0]
    assert value.dtype == sent_cos.dtype == sent_sin.dtype == torch.float32
    assert sent_cos.shape[0] == 1
    expected_cos = torch.cat((cos, cos), -1) if mode == "half" else cos
    expected_sin = -torch.cat((sin, sin), -1) if mode == "half" else -sin
    if unbatched:
        expected_cos, expected_sin = expected_cos.unsqueeze(0), expected_sin.unsqueeze(0)
    if not unbatched:
        expected_cos = expected_cos.flatten(0, 1).unsqueeze(0)
        expected_sin = expected_sin.flatten(0, 1).unsqueeze(0)
    torch.testing.assert_close(sent_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sent_sin, expected_sin, rtol=0, atol=0)
    assert rotary_mode == ("half" if mode == "half" else "interleave")
    torch.testing.assert_close(actual, torch.full_like(x, 3), rtol=0, atol=0)


def test_override_reaches_text_compressor_indexer_and_vision_without_new_state():
    overrides = OverrideConfig(imports=["torchtitan_npu.override.deepseek_v41.rope.ascendc"])
    cfg = model_registry("deepseek_v41_debugmodel").model
    apply_overrides(overrides, cfg)
    assert isinstance(cfg.layers[0].attention.rotary, AscV41RoPERotation.Config)
    assert isinstance(cfg.layers[2].attention.compressor.rotary, AscV41RoPERotation.Config)
    assert isinstance(cfg.layers[2].attention.indexer.rotary, AscV41RoPERotation.Config)
    vision = DeepSeekV41VisionEncoder.Config(dim=8, num_layers=2, num_heads=2, inter_dim=16, text_dim=8)
    reference = vision.build()
    adapted = copy.deepcopy(vision)
    apply_overrides(overrides, adapted)
    model = adapted.build()
    assert all(isinstance(b.attn.rotary, AscV41RoPERotation) for b in model.blocks)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert model.state_dict().keys() == reference.state_dict().keys()
