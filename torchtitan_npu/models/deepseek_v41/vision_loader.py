from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader

from torchtitan_npu.models.deepseek_v41.vision_data import (
    ImagePatchProcessor,
    build_image_token_layout,
    build_shifted_labels,
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
    ):
        self.vocab_size = vocab_size
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
                repeated = (text_tokens * ((self.seq_len // len(text_tokens)) + 1))[: self.seq_len]
                tokens = torch.tensor(repeated, dtype=torch.long)
                # Use the full image protocol for the new index-based path. The
            # legacy contiguous span remains in the batch for old checkpoints.
            if self.image_paths:
                pixel_values, image_grid = ImagePatchProcessor().from_path(
                    self.image_paths[step % len(self.image_paths)]
                )
                grid_hw = (int(image_grid[0]), int(image_grid[1]))
            else:
                pixel_values = torch.randn(self.patch_count, 3 * 14 * 14, dtype=torch.float32)
                grid_hw = (8, 8)
            span_len = ((grid_hw[0] + 2) // 3) * ((grid_hw[1] + 2) // 3)
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
                "image_spans": torch.tensor([[0, self.span_start, span_len]], dtype=torch.long),
                "token_types": token_types,
                "image_feature_indices": image_feature_indices,
            }
            yield input_dict, labels
            step += 1


class DeepSeekV4SyntheticVisionDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        vocab_size: int = 129280
        patch_count: int = 64
        image_span_start: int = 8
        image_paths: tuple[str, ...] = ()
        tokenizer_path: str | None = None
        text: str = "Describe the image."
        infinite: bool = True

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
