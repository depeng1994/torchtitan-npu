# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
from typing import Any

import torch


class DeepSeekV41VisionStateDictAdapter:
    """Explicit adapter for V4.1 vision/marker namespaces.

    Owns keys under ``vision.*`` / ``vision_encoder.*`` and the special
    markers ``image_start`` / ``image_newline`` / ``image_end`` /
    ``image_marker_embeddings.*``.  Text-backbone keys are delegated to the
    inherited V4 adapter; V4.1-only text-side mappings are registered by the
    composed adapter below.
    """

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
    }

    _HF_PREFIXES = ("vision.", "aligner.")
    _HF_EXACT = {"image_start", "image_newline", "image_end"}
    _LOCAL_PREFIXES = ("vision_encoder.", "image_marker_embeddings.")

    def __init__(self, *, expected_shapes: Mapping[str, tuple[int, ...]] | None = None):
        self.expected_shapes = dict(expected_shapes or {})

    @classmethod
    def owns_hf_key(cls, key: str) -> bool:
        return key in cls._HF_EXACT or any(key.startswith(p) for p in cls._HF_PREFIXES)

    @classmethod
    def owns_local_key(cls, key: str) -> bool:
        return any(key.startswith(p) for p in cls._LOCAL_PREFIXES)

    @staticmethod
    def _to_local_key(key: str) -> str:
        if key.startswith("vision."):
            return "vision_encoder." + key[len("vision.") :]
        if key.startswith("aligner."):
            return "vision_encoder." + key
        if key in ("image_start", "image_newline", "image_end"):
            return "image_marker_embeddings." + key
        return key

    @staticmethod
    def _to_hf_key(key: str) -> str:
        if key.startswith("vision_encoder.aligner."):
            return "aligner." + key[len("vision_encoder.") :]
        if key.startswith("vision_encoder."):
            return "vision." + key[len("vision_encoder.") :]
        if key.startswith("image_marker_embeddings."):
            return key[len("image_marker_embeddings.") :]
        return key

    def from_hf(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in state_dict.items():
            local_key = self._FROM_HF.get(key, self._to_local_key(key) if self.owns_hf_key(key) else key)
            self._validate(local_key, value)
            result[local_key] = value
        return result

    def to_hf(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        reverse = {local: hf for hf, local in self._FROM_HF.items()}
        result: dict[str, Any] = {}
        for key, value in state_dict.items():
            hf_key = reverse.get(key, self._to_hf_key(key) if self.owns_local_key(key) else key)
            self._validate(key, value)
            result[hf_key] = value
        return result

    def _validate(self, key: str, value: Any) -> None:
        expected = self.expected_shapes.get(key)
        if expected is None or not isinstance(value, torch.Tensor):
            return
        if tuple(value.shape) != expected:
            raise ValueError(f"shape mismatch for {key}: expected {expected}, got {tuple(value.shape)}")
