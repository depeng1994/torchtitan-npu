# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 data entry: the synthetic tokenizer plus the image protocol exports."""

from dataclasses import dataclass

from torchtitan.components.tokenizer import BaseTokenizer

from .vision_data import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    TEXT,
    ImagePatchProcessor,
    VisionBatch,
    build_image_token_layout,
    build_shifted_labels,
    scatter_image_features,
)

__all__ = [
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_START",
    "TEXT",
    "ImagePatchProcessor",
    "SyntheticTokenizer",
    "VisionBatch",
    "build_image_token_layout",
    "build_shifted_labels",
    "scatter_image_features",
]


class SyntheticTokenizer(BaseTokenizer):
    """Deterministic ord-modulo tokenizer (the V4.1 golden dataloader default)."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int = 129280

    def __init__(self, config: Config, *, tokenizer_path: str | None = None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.eos_id = 1

    def encode(self, text: str, **kwargs) -> list[int]:  # pyrefly: ignore [bad-override]
        return [2 + (ord(char) % max(self.vocab_size - 2, 1)) for char in text]

    def decode(self, token_ids, **kwargs) -> str:  # pyrefly: ignore [bad-override]
        return "".join(chr(int(token_id) % 128) for token_id in token_ids)

    def get_vocab_size(self) -> int:
        return self.vocab_size
