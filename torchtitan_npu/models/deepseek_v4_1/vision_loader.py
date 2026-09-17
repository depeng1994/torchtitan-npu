# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader

from torchtitan_npu.models.deepseek_v4_1.vision_data import (
    ImagePatchProcessor,
    build_image_token_layout,
    build_shifted_labels,
)


def _validate_document_alignment(seq_len: int, document_alignment: int) -> None:
    """Check that every emitted document sits on the pooling grid.

    Each row this dataset emits is one document, so the row edge is the document
    edge a compressed-attention pool of k tokens must not straddle: the row
    length has to be a multiple of k.  A row cannot be padded after it is cut,
    so the requirement is checked when the dataset is built.
    """
    if document_alignment < 1:
        raise ValueError("document_alignment must be positive")
    if seq_len % document_alignment != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of document_alignment "
            f"({document_alignment}): every row is one document, so the row edge "
            "is the document edge the compressed-attention pool must not straddle."
        )


class _SyntheticVisionDataset(IterableDataset):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        patch_count: int,
        span_start: int,
        image_paths: tuple[str, ...] = (),
        tokenizer_path: str | None = None,
        text: str = "Describe the image.",
        document_alignment: int = 1,
    ):
        _validate_document_alignment(seq_len, document_alignment)
        self.vocab_size = vocab_size
        self.document_alignment = document_alignment
        self.seq_len = seq_len
        self.patch_count = patch_count
        self.span_start = span_start
        self.image_paths = image_paths
        self.tokenizer_path = tokenizer_path
        self.text = text

    def __iter__(self):
        tokenizer = None
        if self.tokenizer_path:
            from torchtitan.components.tokenizer import HuggingFaceTokenizer

            tokenizer = HuggingFaceTokenizer(tokenizer_path=self.tokenizer_path)
            text_tokens = tokenizer.encode(self.text, add_bos=True, add_eos=True)
            if not text_tokens:
                raise ValueError("configured vision text produced no tokenizer ids")
        step = 0
        while True:
            if tokenizer is None:
                tokens = (torch.arange(self.seq_len, dtype=torch.long) + step) % self.vocab_size
            else:
                repeated = (text_tokens * ((self.seq_len // len(text_tokens)) + 1))[  # pyrefly: ignore [unbound-name]
                    : self.seq_len
                ]  # pyrefly: ignore [unbound-name]
                tokens = torch.tensor(repeated, dtype=torch.long)
            # This loader emits only the index-based image protocol; the
            # model keeps a span-based fusion path for external span batches.
            if self.image_paths:
                pixel_values, image_grid = ImagePatchProcessor().from_path(
                    self.image_paths[step % len(self.image_paths)]
                )
                grid_hw = (int(image_grid[0]), int(image_grid[1]))
            else:
                generator = torch.Generator().manual_seed(1234)
                pixel_values = torch.randn(
                    self.patch_count,
                    3 * 14 * 14,
                    dtype=torch.float32,
                    generator=generator,
                )
                grid_hw = (8, 8)
            layout_ids, layout_types, layout_indices = build_image_token_layout(
                [grid_hw],
                span_start=self.span_start,
                vocab_size=self.vocab_size,
            )
            if layout_ids.numel() > self.seq_len:
                raise ValueError("image protocol exceeds configured sequence length")
            tokens[self.span_start : layout_ids.numel()] = layout_ids[self.span_start :]
            token_types = torch.full((self.seq_len,), -1, dtype=torch.long)
            image_feature_indices = torch.full((self.seq_len,), -1, dtype=torch.long)
            token_types[: layout_types.numel()] = layout_types
            image_feature_indices[: layout_indices.numel()] = layout_indices
            labels = build_shifted_labels(tokens, token_types)
            input_dict = {
                "input": tokens,
                "positions": torch.arange(self.seq_len, dtype=torch.long),
                "pixel_values": pixel_values,
                "image_grid": torch.tensor([grid_hw], dtype=torch.long),
                "token_types": token_types,
                "image_feature_indices": image_feature_indices,
            }
            yield input_dict, labels
            step += 1


class DeepSeekV41SyntheticVisionDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        vocab_size: int = 129280
        patch_count: int = 64
        image_span_start: int = 8
        image_paths: tuple[str, ...] = ()
        tokenizer_path: str | None = None
        text: str = "Describe the image."
        infinite: bool = True
        document_alignment: int = 1
        """Multiple every emitted document (row) must be aligned to.

        Each row of this loader is a single document, so a model that pools k
        consecutive tokens needs ``seq_len`` to be a multiple of k; the value is
        derived from the model's compression ratios by the config registry.
        """

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        if local_batch_size != 1:
            raise ValueError("DeepSeek V4 synthetic vision loader currently requires local_batch_size=1")
        dataset = _SyntheticVisionDataset(
            vocab_size=config.vocab_size,
            seq_len=seq_len,
            patch_count=config.patch_count,
            span_start=config.image_span_start,
            image_paths=config.image_paths,
            tokenizer_path=config.tokenizer_path,
            text=config.text,
            document_alignment=config.document_alignment,
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )
