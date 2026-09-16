# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CC12M caption data entry for the V4.1 training baseline.

The single real-image-text entry (replaces the LLaVA entry).  One sample
is one image plus one caption:

    BOS + full V4.1 image protocol + caption tokens + EOS

The caption and the EOS are the only supervised targets; BOS, the image
protocol and padding are masked out.  ``assemble_caption_sequence`` is
the one place that builds this sequence — the offline preparation
(``examples/deepseek_v41/prepare_cc12m.py``) and the training dataset
both call it, so length filtering, supervision and statistics can never
drift apart.

DP sharding is the dataset's job (``ParallelAwareDataloader`` only keeps
rank state): each rank iterates ``manifest[rank::world]`` and cycles.
Cycling is intentional and explicit: fixed alignment recipes must keep
``steps * world <= manifest size`` (the digest entry below reports the
consumption budget).

Subcommands:

* ``python -m torchtitan_npu.models.deepseek_v41.cc12m_loader digest ...``
  — A/B preflight: sample ids and hashes of the first batches per rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from .vision_data import TEXT, ImagePatchProcessor, build_image_token_layout

PAD_ID = 0


class SampleBudgetExceededError(Exception):
    """A sample needs more tokens than the configured sequence budget."""


def assemble_caption_sequence(
    *,
    caption_ids: list[int],
    grid_hw: tuple[int, int],
    vocab_size: int,
    bos_id: int,
    eos_id: int,
    seq_len: int,
    downsample_ratio: int = 3,
) -> tuple[list[int], list[int], list[int], list[bool]]:
    """Build the one sample sequence: BOS + image protocol + caption + EOS.

    Returns ``(ids, token_types, image_feature_indices, supervised)``,
    four parallel lists without padding.  ``supervised[j]`` marks token
    j as a supervised target (caption text or the closing EOS).  Raises
    :class:`SampleBudgetExceededError` when the assembled length exceeds
    ``seq_len`` — preparation filters on this exact exception and the
    runtime fails on it, so both sides judge every sample identically.
    """
    if not caption_ids:
        raise ValueError("caption must encode to at least one token")
    layout_ids, layout_types, layout_indices = build_image_token_layout(
        [grid_hw], span_start=0, vocab_size=vocab_size, downsample_ratio=downsample_ratio
    )
    ids = [bos_id, *layout_ids.tolist(), *caption_ids, eos_id]
    n_text = len(caption_ids) + 1  # caption tokens + closing eos
    types = [TEXT, *layout_types.tolist(), *([TEXT] * n_text)]
    indices = [-1, *layout_indices.tolist(), *([-1] * n_text)]
    supervised = [False, *([False] * layout_ids.numel()), *([True] * n_text)]
    if len(ids) > seq_len:
        raise SampleBudgetExceededError(
            f"assembled length {len(ids)} exceeds seq_len={seq_len} "
            f"(protocol {layout_ids.numel()}, caption {len(caption_ids)})"
        )
    return ids, types, indices, supervised


def _pad_sample(
    ids: list[int], types: list[int], indices: list[int], supervised: list[bool], seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad to ``seq_len`` and apply the single next-token shift."""
    pad = seq_len - len(ids)
    tokens = torch.tensor(ids + [PAD_ID] * pad, dtype=torch.long)
    token_types = torch.tensor(types + [TEXT] * pad, dtype=torch.long)
    feature_indices = torch.tensor(indices + [-1] * pad, dtype=torch.long)
    targets = supervised + [False] * pad
    labels = torch.full((seq_len,), -100, dtype=torch.long)
    supervised_next = torch.tensor(targets[1:], dtype=torch.bool)
    labels[:-1] = tokens[1:].masked_fill(~supervised_next, -100)
    return tokens, token_types, feature_indices, labels


class _Cc12mCaptionDataset(IterableDataset):
    """Fixed CC12M manifest: image + caption, assistant-free supervision."""

    def __init__(
        self,
        *,
        manifest_path: str,
        data_dir: str,
        tokenizer_path: str,
        seq_len: int,
        vocab_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ):
        if dp_world_size <= 0:
            raise ValueError(f"dp_world_size must be positive, got {dp_world_size}")
        if not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"dp_rank {dp_rank} out of range for world size {dp_world_size}")
        self._entries = self._read_manifest(manifest_path)
        self._data_dir = Path(data_dir)
        self._seq_len = seq_len
        self._rank = dp_rank
        self._world = dp_world_size
        shard = self._entries[dp_rank::dp_world_size]
        if not shard:
            raise ValueError(
                f"manifest {manifest_path} has {len(self._entries)} samples; "
                f"DP rank {dp_rank}/{dp_world_size} would receive an empty shard"
            )
        self._shard = shard
        self._processor = ImagePatchProcessor()
        self._tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)
        self._vocab_size = vocab_size
        if self._tokenizer.get_vocab_size() != vocab_size:
            raise ValueError(
                f"tokenizer vocab ({self._tokenizer.get_vocab_size()}) does not match "
                f"the configured vocab_size {vocab_size}; the trainer-side tokenizer "
                "config and the data-entry tokenizer must agree"
            )
        self._verify_meta(manifest_path)

        # The tokenizer's own special tokens define the sequence frame.
        # Narrow to non-optional ints before storing.
        bos_id = self._tokenizer.bos_id
        eos_id = self._tokenizer.eos_id
        if bos_id is None or eos_id is None:
            raise ValueError("the tokenizer must define bos_id and eos_id")
        self._bos: int = bos_id
        self._eos: int = eos_id

    @staticmethod
    def _read_manifest(manifest_path: str) -> list[dict[str, Any]]:
        entries = []
        seen_ids: set[str] = set()
        with open(manifest_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                for field in ("id", "image", "caption"):
                    if not isinstance(entry.get(field), str) or not entry[field]:
                        raise ValueError(f"manifest line {line_no}: missing field {field!r}")
                if entry["id"] in seen_ids:
                    raise ValueError(f"manifest line {line_no}: duplicate sample id {entry['id']!r}")
                seen_ids.add(entry["id"])
                entries.append(entry)
        if not entries:
            raise ValueError(f"manifest {manifest_path} contains no samples")
        return entries

    def _verify_meta(self, manifest_path: str) -> None:
        """Cross-check the prepared data-recipe identity when meta.json exists.

        Verifies the manifest SHA256, the tokenizer file hashes and the
        image-processing parameters recorded by ``prepare_cc12m.py`` so a
        stale or edited subset fails loudly instead of training silently.
        Fixture directories without meta.json are allowed (tests, ad-hoc
        manifests).
        """
        meta_path = self._data_dir / "meta.json"
        if not meta_path.is_file():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        if digest != meta.get("manifest_sha256"):
            raise ValueError(
                f"manifest {manifest_path} does not match meta.json (sha256 {digest}); "
                "the subset was modified after preparation — re-run prepare_cc12m.py"
            )
        for name, expected in (meta.get("tokenizer_files") or {}).items():
            actual = hashlib.sha256((Path(self._tokenizer.tokenizer_path) / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"tokenizer file {name} hash does not match meta.json")
        recorded = meta.get("image_processing") or {}
        for field in ("patch_size", "downsample_ratio", "min_pixels", "max_n_token", "max_wh_ratio"):
            if field in recorded and recorded[field] != getattr(self._processor, field):
                raise ValueError(
                    f"image-processing parameter {field}={getattr(self._processor, field)!r} differs from "
                    f"the prepared recipe value {recorded[field]!r}"
                )
        for field in ("mean", "std"):
            if field in recorded and tuple(recorded[field]) != getattr(self._processor, field):
                raise ValueError(
                    f"image-processing parameter {field}={getattr(self._processor, field)!r} differs from "
                    f"the prepared recipe value {recorded[field]!r}"
                )

    def _build_sample(self, entry: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        image_path = self._data_dir / entry["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"manifest image missing on disk: {image_path}")
        pixel_values, image_grid = self._processor.from_path(image_path)
        grid_hw = (int(image_grid[0]), int(image_grid[1]))
        caption_ids = self._tokenizer.encode(entry["caption"], add_bos=False, add_eos=False)
        if not caption_ids:
            raise ValueError(f"sample {entry['id']}: caption encodes to zero tokens")
        try:
            ids, types, indices, supervised = assemble_caption_sequence(
                caption_ids=caption_ids,
                grid_hw=grid_hw,
                vocab_size=self._vocab_size,
                bos_id=self._bos,
                eos_id=self._eos,
                seq_len=self._seq_len,
                downsample_ratio=self._processor.downsample_ratio,
            )
        except SampleBudgetExceededError as exc:
            # The preparation filters on the same exception; reaching here
            # means manifest and config disagree — fail loudly.
            raise ValueError(f"sample {entry['id']}: {exc}") from exc
        tokens, token_types, feature_indices, labels = _pad_sample(ids, types, indices, supervised, self._seq_len)
        input_dict = {
            "input": tokens,
            "positions": torch.arange(self._seq_len, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid": torch.tensor([grid_hw], dtype=torch.long),
            "token_types": token_types,
            "image_feature_indices": feature_indices,
        }
        return input_dict, labels

    def __iter__(self):
        # Explicit cycling semantics: the shard repeats forever; alignment
        # recipes must keep steps * world <= manifest size so no epoch
        # boundary is crossed within a compared run.
        while True:
            for entry in self._shard:
                yield self._build_sample(entry)


class DeepSeekV41Cc12mDataLoader(ParallelAwareDataloader):
    """ParallelAware dataloader over the fixed CC12M manifest."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        manifest_path: str = ""
        data_dir: str = ""
        tokenizer_path: str = ""
        vocab_size: int = 129280

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
            raise ValueError("the CC12M loader requires local_batch_size=1")
        if config.num_workers != 0:
            # Each worker would replay the same rank shard; single-process
            # iteration is the contract for this first version.
            raise ValueError("the CC12M loader requires num_workers=0")
        for field_name in ("manifest_path", "data_dir", "tokenizer_path"):
            if not getattr(config, field_name):
                raise ValueError(f"DeepSeekV41Cc12mDataLoader requires config.{field_name}")
        dataset = _Cc12mCaptionDataset(
            manifest_path=config.manifest_path,
            data_dir=config.data_dir,
            tokenizer_path=config.tokenizer_path,
            seq_len=seq_len,
            vocab_size=config.vocab_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
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


def _sha256_prefix(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def digest_batches(
    *,
    manifest_path: str,
    data_dir: str,
    tokenizer_path: str,
    seq_len: int,
    vocab_size: int,
    world: int,
    num_batches: int,
    global_batch_size: int | None = None,
) -> dict[str, Any]:
    """A/B preflight: per-rank ids and content hashes of the first batches.

    ``num_batches`` counts optimizer steps.  With the standard recipe
    (local_batch_size=1, no gradient accumulation) one step consumes
    ``world`` samples; when ``global_batch_size`` is given and exceeds
    ``world``, the per-step consumption is ``global_batch_size`` and the
    report says so.  Reports, for every rank, the first ``num_batches``
    samples' ids, input/label tensor hashes, image-preprocessing hash
    and supervised token count, plus the consumption check for
    fixed-length runs.
    """
    if world <= 0:
        raise ValueError(f"world must be positive, got {world}")
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}")
    if global_batch_size is not None:
        if global_batch_size <= 0:
            raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")
        if global_batch_size % world != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be a multiple of world ({world}) "
                f"for local_batch_size=1 gradient accumulation"
            )
    manifest = _Cc12mCaptionDataset._read_manifest(manifest_path)
    per_step = world if global_batch_size is None else global_batch_size
    consumed = num_batches * per_step
    if consumed > len(manifest):
        raise ValueError(
            f"consumption check failed: {num_batches} steps x {per_step} samples/step "
            f"(GBS={global_batch_size or world}, world={world}) = {consumed} samples "
            f"> manifest size {len(manifest)}; a fixed-length run would cycle the shard"
        )
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_samples": len(manifest),
        "seq_len": seq_len,
        "world": world,
        "global_batch_size": global_batch_size or world,
        "samples_per_step": per_step,
        "num_batches": num_batches,
        "consumed_samples": consumed,
        "ranks": {},
    }
    per_rank_per_step = per_step // world  # microbatches per rank per step (local_batch_size=1)
    for rank in range(world):
        dataset = _Cc12mCaptionDataset(
            manifest_path=manifest_path,
            data_dir=data_dir,
            tokenizer_path=tokenizer_path,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dp_rank=rank,
            dp_world_size=world,
        )
        iterator = iter(dataset)
        shard = manifest[rank::world]
        samples = []
        for _ in range(num_batches * per_rank_per_step):
            input_dict, labels = next(iterator)
            samples.append(
                {
                    "step": len(samples) // per_rank_per_step,
                    "sample_id": shard[len(samples)]["id"],
                    "input_sha": _sha256_prefix(input_dict["input"]),
                    "labels_sha": _sha256_prefix(labels),
                    "pixel_sha": _sha256_prefix(input_dict["pixel_values"]),
                    "supervised_tokens": int((labels != -100).sum()),
                }
            )
        report["ranks"][f"rank_{rank}"] = samples
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="CC12M loader preflight utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    digest = sub.add_parser("digest", help="per-rank first-batch ids and hashes")
    digest.add_argument("--manifest", required=True)
    digest.add_argument("--data-dir", required=True)
    digest.add_argument("--tokenizer", required=True)
    digest.add_argument("--seq-len", type=int, default=512)
    digest.add_argument("--vocab-size", type=int, default=129280)
    digest.add_argument("--world", type=int, default=8)
    digest.add_argument("--num-batches", type=int, default=3, help="optimizer steps to preflight")
    digest.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="samples consumed per optimizer step (default: world, i.e. local batch 1 without accumulation)",
    )

    args = parser.parse_args()
    if args.command == "digest":
        report = digest_batches(
            manifest_path=args.manifest,
            data_dir=args.data_dir,
            tokenizer_path=args.tokenizer,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            world=args.world,
            num_batches=args.num_batches,
            global_batch_size=args.global_batch_size,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
