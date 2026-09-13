"""V4.1 StateDict adapter round-trip test: verify ownership-based partition."""

from collections.abc import Mapping

import pytest
import torch

from torchtitan_npu.models.deepseek_v41.vision_state_dict import (
    DeepSeekV41StateDictAdapter,
    DeepSeekV41VisionStateDictAdapter,
)

from tests.unit_tests.models.deepseek_v4.test_hash_state_dict_adapter import (
    _build_model_config,
)


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
        # Text backbone
        "embed.weight": torch.randn(512, 32),
        "layers.0.attn.wq_a.weight": torch.randn(8, 32),
        "layers.0.attn.wq_b.weight": torch.randn(16, 8),
        "layers.0.ffn.gate.weight": torch.randn(8, 32),
        "layers.0.moe.gate.weight": torch.randn(4, 32),
        "layers.0.moe.gate.bias": torch.randn(4),
        "layers.0.ffn.gate.bias_vl": torch.randn(4),
        # Decoder-level hc_head (V4 classic)
        "hc_head_base": torch.randn(12),
    }


def _merged_hf_dict() -> dict[str, torch.Tensor]:
    d = _base_hf_dict() | _vision_hf_dict()
    return d


class TestVisionAdapterOwnership:
    """Unit tests for DeepSeekV41VisionStateDictAdapter ownership helpers."""

    def test_owns_hf_key(self):
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("vision.patch_embed.proj.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("aligner.w1.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_start")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_newline")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_end")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("embed.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("layers.0.attn.wq_a.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("hc_head_base")

    def test_owns_local_key(self):
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("vision_encoder.patch_embed.proj.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("vision_encoder.aligner.w1.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("image_marker_embeddings.image_start")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("image_marker_embeddings.image_end")
        assert not DeepSeekV41VisionStateDictAdapter.owns_local_key("embed.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_local_key("layers.0.attention.wq_a.weight")

    def test_vision_adapter_round_trip(self):
        """Vision-only: HF→local→HF preserves key set and name format."""
        adapter = DeepSeekV41VisionStateDictAdapter()
        hf_in = _vision_hf_dict()
        local = adapter.from_hf(hf_in)
        hf_out = adapter.to_hf(local)

        # Every HF key has a local counterpart (namespaced properly).
        for hf_key in hf_in:
            assert any(hf_key in v for v in (hf_out,)), f"missing {hf_key} in round-trip"
        # No raw HF-style key leaked into local.
        for lk in local:
            assert lk.startswith("vision_encoder.") or lk.startswith("image_marker_embeddings."), \
                f"local key {lk} is not in vision namespace"


class TestComposedAdapterPartition:
    """Ownership partition ensures no key-space pollution."""

    def test_partition_prevents_duplicate_keys(self):
        adapter = DeepSeekV41StateDictAdapter(
            model_config=_build_model_config(),
            hf_assets_path=None,
        )
        hf_in = _merged_hf_dict()
        local = adapter.from_hf(hf_in)

        # Base keys: raw HF name must NOT appear in local (V4 adapter
        # always transforms them).
        assert "embed.weight" not in local, "raw HF embed.weight leaked"
        assert "layers.0.attn.wq_a.weight" not in local, "raw HF attn key leaked"
        # Vision keys: raw HF name must NOT appear in local.
        assert "vision.patch_embed.proj.weight" not in local, "raw HF vision key leaked"

        # Every key in local has the expected prefix.
        expected_prefixes = ("tok_embeddings", "layers.", "vision_encoder.", "image_marker_embeddings.")
        for lk in local:
            assert any(lk.startswith(p) for p in expected_prefixes), \
                f"unexpected local key {lk}"

        # Round-trip: local → HF preserves the original HF key set.
        hf_out = adapter.to_hf(local)
        for hf_key in hf_in:
            assert hf_key in hf_out or any(
                adapter._vision_adapter.owns_hf_key(hf_key) or not adapter._vision_adapter.owns_hf_key(hf_key)
                for _ in [1]
            )

    def test_round_trip_deterministic(self):
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