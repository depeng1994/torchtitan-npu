# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 StateDict adapter round-trip test: verify ownership-based partition."""

import torch

from torchtitan_npu.models.deepseek_v4_1 import deepseek_v4_1_debugmodel_config
from torchtitan_npu.models.deepseek_v4_1.state_dict_adapter import DeepSeekV41StateDictAdapter
from torchtitan_npu.models.deepseek_v4_1.vision_state_dict import DeepSeekV41VisionStateDictAdapter


def _build_model_config():
    """A minimal local V4.1 fixture (the debugmodel topology, tiny widths)."""
    config = deepseek_v4_1_debugmodel_config()
    config.vocab_size = 32
    config.tok_embeddings.num_embeddings = 32
    config.lm_head.out_features = 32
    return config


def _vision_hf_dict() -> dict[str, torch.Tensor]:
    return {
        # Vision tower
        "vision.patch_embed.proj.weight": torch.randn(32, 3, 14, 14),
        "vision.norm.weight": torch.randn(32),
        "aligner.w1.weight": torch.randn(64, 32),
        "image_start": torch.randn(32),
        "image_newline": torch.randn(32),
        "image_end": torch.randn(32),
    }


def _base_hf_dict() -> dict[str, torch.Tensor]:
    return {
        # Text backbone (keys match the V4.1 adapter's from_hf_map).
        "embed.weight": torch.randn(512, 32),
        "layers.3.attn.wq_a.weight": torch.randn(8, 32),
        "layers.3.attn.wq_b.weight": torch.randn(16, 8),
        "layers.3.ffn.gate.weight": torch.randn(4, 32),
        "layers.3.ffn.gate.bias": torch.randn(4),
        "layers.3.ffn.gate.bias_vl": torch.randn(4),
        "norm.weight": torch.randn(32),
    }


def _merged_hf_dict() -> dict[str, torch.Tensor]:
    return _base_hf_dict() | _vision_hf_dict()


class TestVisionAdapterOwnership:
    """Unit tests for DeepSeekV41VisionStateDictAdapter ownership helpers."""

    @staticmethod
    def test_owns_hf_key():
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("vision.patch_embed.proj.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("aligner.w1.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_start")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_newline")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_end")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("embed.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("layers.0.attn.wq_a.weight")

    @staticmethod
    def test_owns_local_key():
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("vision_encoder.patch_embed.proj.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("vision_encoder.aligner.w1.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("image_marker_embeddings.image_start")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("image_marker_embeddings.image_end")
        assert not DeepSeekV41VisionStateDictAdapter.owns_local_key("embed.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_local_key("layers.0.attention.wq_a.weight")

    @staticmethod
    def test_vision_adapter_round_trip():
        """Vision-only: HF→local→HF preserves key set and name format."""
        adapter = DeepSeekV41VisionStateDictAdapter()
        hf_in = _vision_hf_dict()
        local = adapter.from_hf(hf_in)
        hf_out = adapter.to_hf(local)

        # Every HF key has a local counterpart (namespaced properly).
        for hf_key in hf_in:
            assert hf_key in hf_out, f"missing {hf_key} in round-trip"
        # No raw HF-style key leaked into local.
        for lk in local:
            assert lk.startswith("vision_encoder.") or lk.startswith("image_marker_embeddings."), (
                f"local key {lk} is not in vision namespace"
            )


class TestComposedAdapterPartition:
    """Ownership partition ensures no key-space pollution."""

    @staticmethod
    def test_partition_prevents_duplicate_keys():
        adapter = DeepSeekV41StateDictAdapter(
            model_config=_build_model_config(),
            hf_assets_path=None,
        )
        hf_in = _merged_hf_dict()
        local = adapter.from_hf(hf_in)

        # Base keys: raw HF name must NOT appear in local (the adapter
        # always transforms them).
        assert "embed.weight" not in local, "raw HF embed.weight leaked"
        assert "layers.0.attn.wq_a.weight" not in local, "raw HF attn key leaked"
        # Vision keys: raw HF name must NOT appear in local.
        assert "vision.patch_embed.proj.weight" not in local, "raw HF vision key leaked"

        # Every key in local has the expected prefix.
        expected_prefixes = ("tok_embeddings", "layers.", "vision_encoder.", "image_marker_embeddings.", "norm.")
        for lk in local:
            assert any(lk.startswith(p) for p in expected_prefixes), f"unexpected local key {lk}"

        # Round-trip: local → HF must reproduce the original HF key set
        # with bit-identical values (strict contract, no tautology).
        hf_out = adapter.to_hf(local)
        assert set(hf_out) == set(hf_in), (
            f"HF round-trip key set mismatch: missing={set(hf_in) - set(hf_out)}, "
            f"extra={set(hf_out) - set(hf_in)}"
        )
        for key in hf_in:
            assert torch.equal(hf_out[key], hf_in[key]), f"HF round-trip value mismatch for {key}"

    @staticmethod
    def test_round_trip_deterministic():
        adapter = DeepSeekV41StateDictAdapter(
            model_config=_build_model_config(),
            hf_assets_path=None,
        )
        hf_in = _merged_hf_dict()
        local1 = adapter.from_hf(hf_in)
        local2 = adapter.from_hf(hf_in)
        assert set(local1.keys()) == set(local2.keys())
        for k in local1:
            assert torch.equal(local1[k], local2[k]), f"key {k} differs"
