# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
from dataclasses import replace

import torch
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.distributed.activation_checkpoint import FullAC

from torchtitan_npu.models.deepseek_v4_1.vision_data import build_image_token_layout
from torchtitan_npu.models.deepseek_v4_1.vision_loader import _SyntheticVisionDataset

from tests.unit_tests.models.mtp_test_utils import build_cpu_model


def test_synthetic_vision_without_image_files():
    dataset = _SyntheticVisionDataset(vocab_size=64, seq_len=128, patch_count=64, span_start=8)

    sample, labels = next(iter(dataset))
    repeated, repeated_labels = next(iter(dataset))

    assert sample["pixel_values"].shape == (64, 588)
    assert sample["image_grid"].tolist() == [[8, 8]]
    assert sample["image_feature_indices"][sample["image_feature_indices"] >= 0].tolist() == list(range(9))
    assert labels[-1].item() == -100
    assert (labels[:-1][sample["token_types"][1:] >= 0] == -100).all()
    torch.testing.assert_close(sample["pixel_values"], repeated["pixel_values"], rtol=0, atol=0)
    torch.testing.assert_close(labels, repeated_labels, rtol=0, atol=0)


def test_full_ac_preserves_image_routing(monkeypatch):
    # CPU-only checkpoint inputs otherwise inherit the registered NPU backend.
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    monkeypatch.setattr(
        registry,
        "_DEBUG_WIDTHS",
        replace(
            registry._DEBUG_WIDTHS,
            dim=8,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            index_n_heads=2,
            index_head_dim=4,
            moe_inter_dim=16,
            vision_dim=8,
            vision_heads=2,
            vision_inter_dim=16,
        ),
    )
    config = registry.model_registry("deepseek_v4_1_debugmodel").model
    config.vocab_size = 64
    # The trainer's update_from_config fills the aux-loss denominators before the run.
    from torchtitan_npu.models.deepseek_v4_1.indexer import IndexerKLLoss

    for _, loss_cfg, _, _ in config.traverse(IndexerKLLoss.Config):
        loss_cfg.global_batch_size = 1
    config.tok_embeddings.num_embeddings = 64
    config.lm_head.out_features = 64
    # The golden reference arithmetic is the native attention path; no
    # class-swap overrides are needed for the CPU contract test.
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    with torch.no_grad():
        for layer in model.layers.values():
            layer.moe.router.gate.weight.zero_()
            bias = torch.arange(layer.moe.router.num_experts, dtype=torch.float32)
            layer.moe.expert_bias_E.copy_(bias)
            layer.moe.router.bias_vl.copy_(bias.flip(0))
    checkpointed = copy.deepcopy(model)
    FullAC.Config().build().apply(checkpointed)

    # Real image protocol without depending on the synthetic RNG fallback.
    ids, types, feature_ids = build_image_token_layout([(3, 3)], span_start=2, vocab_size=64)
    tokens = torch.arange(128).remainder(32).unsqueeze(0)
    tokens[:, 2:ids.numel()] = ids[2:]
    token_types = torch.full_like(tokens, -1)
    token_types[:, :types.numel()] = types
    indices = torch.full_like(tokens, -1)
    indices[:, :feature_ids.numel()] = feature_ids
    inputs = dict(
        positions=torch.arange(128).unsqueeze(0),
        pixel_values=torch.linspace(-1, 1, 9 * 588).reshape(1, 9, 588),
        image_grid=torch.tensor([[3, 3]]),
        image_feature_indices=indices,
        token_types=token_types,
    )
    expected_mask = token_types >= 0
    route_calls = []

    def capture_route(module, args, kwargs, output):
        mask = kwargs.get("image_mask")
        route_calls.append((None if mask is None else mask.clone(), output[1].detach().clone()))

    handle = checkpointed.layers["0"].moe.router.register_forward_hook(capture_route, with_kwargs=True)
    try:
        _, _, kwargs = model.build_attention_masks(tokens, tokens, dict(inputs))
        expected = model(tokens, **kwargs)
        _, _, kwargs = checkpointed.build_attention_masks(tokens, tokens, dict(inputs))
        actual = checkpointed(tokens, **kwargs)
        assert route_calls[0][0] is not None
        torch.testing.assert_close(route_calls[0][0], expected_mask)
        selected = route_calls[0][1]
        assert selected[expected_mask][0].tolist() == [0, 1, 2, 3, 4, 5]
        assert selected[~expected_mask][0].tolist() == [15, 14, 13, 12, 11, 10]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for mask, _ in route_calls:
            torch.testing.assert_close(mask, expected_mask)
        for expected_parameter, actual_parameter in zip(
            model.parameters(), checkpointed.parameters(), strict=True
        ):
            assert (expected_parameter.grad is None) == (actual_parameter.grad is None)
            if expected_parameter.grad is not None:
                torch.testing.assert_close(
                    actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-7
                )
        route_calls.clear()
        _, _, kwargs = checkpointed.build_attention_masks(tokens, tokens, {"positions": inputs["positions"]})
        checkpointed(tokens, **kwargs)
        assert route_calls[0][0] is None
    finally:
        handle.remove()
