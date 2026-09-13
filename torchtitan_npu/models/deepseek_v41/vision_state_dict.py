from collections.abc import Mapping
from typing import Any

import torch
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter


class DeepSeekV41VisionStateDictAdapter:
    """Explicit adapter for V4.1 vision/marker/router namespaces."""

    _FROM_HF = {
        "vision.patch_embed.proj.weight": "vision_encoder.patch_embed.proj.weight",
        "vision.patch_embed.proj.bias": "vision_encoder.patch_embed.proj.bias",
        "vision.norm.weight": "vision_encoder.norm.weight",
        "aligner.w1.weight": "vision_encoder.aligner.w1.weight",
        "aligner.w1.bias": "vision_encoder.aligner.w1.bias",
        "aligner.w2.weight": "vision_encoder.aligner.w2.weight",
        "aligner.w2.bias": "vision_encoder.aligner.w2.bias",
        "image_start": "image_marker_embeddings.image_start",
        "image_newline": "image_marker_embeddings.image_newline",
        "image_end": "image_marker_embeddings.image_end",
        "bias_vl": "bias_vl",
    }

    def __init__(self, *, expected_shapes: Mapping[str, tuple[int, ...]] | None = None):
        self.expected_shapes = dict(expected_shapes or {})

    @staticmethod
    def _to_local_key(key: str) -> str:
        if key.startswith("vision."):
            return "vision_encoder." + key[len("vision."):]
        return key

    @staticmethod
    def _to_hf_key(key: str) -> str:
        if key.startswith("vision_encoder."):
            return "vision." + key[len("vision_encoder."):]
        return key

    def from_hf(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in state_dict.items():
            local_key = self._FROM_HF.get(key, self._to_local_key(key))
            self._validate(local_key, value)
            result[local_key] = value
        return result

    def to_hf(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        reverse = {local: hf for hf, local in self._FROM_HF.items()}
        result: dict[str, Any] = {}
        for key, value in state_dict.items():
            hf_key = reverse.get(key, self._to_hf_key(key))
            self._validate(key, value)
            result[hf_key] = value
        return result

    def _validate(self, key: str, value: Any) -> None:
        expected = self.expected_shapes.get(key)
        if expected is None or not isinstance(value, torch.Tensor):
            return
        if tuple(value.shape) != expected:
            raise ValueError(
                f"shape mismatch for {key}: expected {expected}, got {tuple(value.shape)}"
            )


class DeepSeekV41StateDictAdapter(DeepSeekV4StateDictAdapter):
    """V4.1 state-dict adapter composing the V4 base mapping with the vision
    tower and image-marker embeddings."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self._vision_adapter = DeepSeekV41VisionStateDictAdapter()

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        result = super().from_hf(hf_state_dict)
        vision_keys = self._vision_adapter.from_hf(hf_state_dict)
        result.update(vision_keys)
        return result

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        result = super().to_hf(state_dict)
        vision_keys = self._vision_adapter.to_hf(state_dict)
        result.update(vision_keys)
        return result

