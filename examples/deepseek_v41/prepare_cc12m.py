#!/usr/bin/env python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline CC12M subset preparation for the V4.1 training baseline.

Scans local CC12M WebDataset tars (``pixparse/cc12m-wds`` layout: every
sample is a ``{key}.jpg`` + ``{key}.txt`` pair), keeps image-text pairs
whose image fully decodes and whose assembled sequence fits the exact
training-time budget, samples a fixed ``count`` of them with a fixed
seed, and writes:

    <output>/manifest.jsonl   {"id", "image", "caption"} per line
    <output>/meta.json        data-recipe identity (hashes, params, stats)
    <output>/images/*.jpg     only the selected images (hash-verified)

The length/supervision judgment is ``assemble_caption_sequence`` from
``torchtitan_npu.models.deepseek_v41.cc12m_loader`` — the very function
the training dataset runs — so "prepared" and "trainable" can never
disagree (R1).  Image shape decisions use the shared
``ImagePatchProcessor.target_grid`` method (R4).

Usage on the training host:

    python examples/deepseek_v41/prepare_cc12m.py \
        --tars /data/p00465316/fused/datasets/cc12m/tars/cc12m-train-0000.tar \
               /data/p00465316/fused/datasets/cc12m/tars/cc12m-train-0001.tar \
        --revision 796118f2eabdb9984f23f7f15d1e74d388612fc6 \
        --tokenizer /data/p00465316/fused/dsv41_tokenizer \
        --output-dir /data/p00465316/fused/datasets/cc12m/subset_8k \
        --count 8000 --seq-len 512 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path
from typing import Any


def _summary(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(ordered), "min": ordered[0], "p50": ordered[len(ordered) // 2], "max": ordered[-1]}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_tars(paths: list[str]) -> list[Path]:
    """Deduplicate, sort stably and reject ambiguous tar identities.

    The tar file stem is the shard identity used in sample ids and image
    names, so two distinct files with the same stem would silently mix
    (or overwrite) each other's images; refuse that outright.  Sorting
    makes the seed-dependent selection independent of the --tars order.
    """
    # Sort by (stem length, name) so equally-padded shard names sort
    # numerically ("0009" < "0010"); mixed-padding inputs need consistent naming.
    unique = {Path(p).resolve(): None for p in paths}
    tars = sorted(unique, key=lambda p: (len(p.stem), p.name))
    stems = [path.stem for path in tars]
    for path in tars:
        if not path.is_file():
            raise SystemExit(f"tar not found: {path}")
    duplicated = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicated:
        raise SystemExit(
            f"duplicate tar stem(s) {duplicated}: the shard identity is the tar file stem; "
            "distinct files with the same name would silently mix image-text pairs"
        )
    return tars


def _iter_wds_samples(tar: tarfile.TarFile) -> Any:
    """Yield ``(key, {"jpg": bytes, "txt": bytes})`` per WebDataset sample.

    WebDataset tars store each sample's files under one key consecutively;
    a member re-using an earlier key (or a duplicate extension member)
    would break that contract and is rejected loudly instead of silently
    mis-pairing or overwriting.
    """
    seen_keys: set[str] = set()
    current_key: str | None = None
    pending: dict[str, bytes] = {}
    for member in tar:
        key, _, ext = member.name.rpartition(".")
        if key != current_key:
            if pending:
                yield current_key, pending
            if key in seen_keys:
                raise ValueError(f"tar member order violation: key {key!r} re-appears after other samples")
            seen_keys.add(key)
            current_key, pending = key, {}
        if ext in ("jpg", "txt"):
            if ext in pending:
                raise ValueError(f"duplicate {ext} member for tar key {key!r}")
            fileobj = tar.extractfile(member)
            if fileobj is None:
                raise ValueError(f"tar member {member.name!r} is not a regular file")
            pending[ext] = fileobj.read()
    if pending:
        yield current_key, pending


def _decode_image(image_bytes: bytes) -> tuple[int | None, int | None]:
    """Full-pixel decode validation; (width, height), or (None, None) if corrupt.

    A JPEG with an intact header but truncated pixel data still opens
    lazily — only ``load()`` proves the sample is trainable.
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return image.size
    except Exception:
        return None, None


def _scan_tars(
    tar_paths: list[Path],
    *,
    encode,
    bos_id: int,
    eos_id: int,
    seq_len: int,
    vocab_size: int,
    processor,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """First pass: enumerate candidates with the exact training-time judgment."""
    from torchtitan_npu.models.deepseek_v41.cc12m_loader import (
        SampleBudgetExceededError,
        assemble_caption_sequence,
    )

    filtered: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    for tar_path in tar_paths:
        shard = tar_path.stem
        with tarfile.open(tar_path, "r") as tar:
            for key, sample in _iter_wds_samples(tar):
                sample_id = f"{shard}/{key}"
                caption = sample.get("txt", b"").decode("utf-8", errors="replace").strip()
                caption_ids = encode(caption)
                if not caption_ids:
                    filtered["empty_caption"] = filtered.get("empty_caption", 0) + 1
                    continue
                image_bytes = sample.get("jpg")
                if image_bytes is None:
                    filtered["missing_image"] = filtered.get("missing_image", 0) + 1
                    continue
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                width, height = _decode_image(image_bytes)
                if width is None:
                    filtered["bad_image"] = filtered.get("bad_image", 0) + 1
                    continue
                grid_hw = processor.target_grid(height, width)
                try:
                    ids, _, _, supervised = assemble_caption_sequence(
                        caption_ids=caption_ids,
                        grid_hw=grid_hw,
                        vocab_size=vocab_size,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        seq_len=seq_len,
                        downsample_ratio=processor.downsample_ratio,
                    )
                except SampleBudgetExceededError:
                    filtered["overlong"] = filtered.get("overlong", 0) + 1
                    continue
                candidates.append(
                    {
                        "sample_id": sample_id,
                        "shard": shard,
                        "key": key,
                        "caption": caption,
                        "image_hash": image_hash,
                        "total": len(ids),
                        "supervised": sum(supervised),
                        "protocol": len(ids) - len(caption_ids) - 2,
                    }
                )
    return candidates, filtered


def _extract_selected(tar_paths: list[Path], selected: list[dict[str, Any]], images_dir: Path) -> None:
    """Second pass: extract only the selected images from the tars."""
    images_dir.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, dict[str, Path]] = {}
    for entry in selected:
        by_shard.setdefault(entry["shard"], {})[entry["key"]] = images_dir / entry["image_name"]
    for tar_path in tar_paths:
        wanted = by_shard.get(tar_path.stem)
        if not wanted:
            continue
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                key, _, ext = member.name.rpartition(".")
                target = wanted.get(key)
                if target is not None and ext == "jpg":
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        raise ValueError(f"tar member {member.name!r} is not a regular file")
                    target.write_bytes(fileobj.read())


def _verify_extracted(selected: list[dict[str, Any]], images_dir: Path) -> None:
    """Every extracted image must re-read with the scan-stage hash."""
    for entry in selected:
        data = (images_dir / entry["image_name"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["image_hash"]:
            raise SystemExit(f"extracted image hash mismatch for {entry['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tars", nargs="+", required=True, help="local CC12M WebDataset tar files")
    parser.add_argument(
        "--revision",
        default=None,
        help="source revision the tars were downloaded from; recorded verbatim in meta.json "
        "(omit for locally-produced tars — do not guess a revision)",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit(f"--count must be positive, got {args.count}")
    tar_paths = _resolve_tars(args.tars)
    output_dir = Path(args.output_dir)

    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    from torchtitan_npu.models.deepseek_v41.vision_data import ImagePatchProcessor

    tokenizer = HuggingFaceTokenizer(tokenizer_path=args.tokenizer)
    bos_id, eos_id = tokenizer.bos_id, tokenizer.eos_id
    if bos_id is None or eos_id is None:
        raise SystemExit("the tokenizer must define bos_id and eos_id")
    vocab_size = tokenizer.get_vocab_size()
    processor = ImagePatchProcessor()

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_bos=False, add_eos=False)

    candidates, filtered = _scan_tars(
        tar_paths,
        encode=encode,
        bos_id=bos_id,
        eos_id=eos_id,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        processor=processor,
    )

    # Deterministic selection: seeded shuffle, first occurrence per image
    # hash (exact-duplicate removal), until count.
    random.Random(args.seed).shuffle(candidates)
    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for entry in candidates:
        if entry["image_hash"] in seen_hashes:
            filtered["duplicate_image"] = filtered.get("duplicate_image", 0) + 1
            continue
        seen_hashes.add(entry["image_hash"])
        entry["image_name"] = f"{entry['shard']}-{entry['key']}.jpg"
        selected.append(entry)
        if len(selected) >= args.count:
            break
    if len(selected) < args.count:
        raise SystemExit(
            f"not enough qualifying samples: requested {args.count}, "
            f"only {len(selected)} unique-image pairs available "
            f"({len(candidates)} candidates before dedup); add more shards"
        )

    _extract_selected(tar_paths, selected, output_dir / "images")
    _verify_extracted(selected, output_dir / "images")

    manifest_lines = [
        json.dumps(
            {"id": entry["sample_id"], "image": f"images/{entry['image_name']}", "caption": entry["caption"]},
            ensure_ascii=False,
        )
        for entry in selected
    ]
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    tokenizer_files = sorted(Path(args.tokenizer).glob("*.json"))
    meta = {
        "source": "pixparse/cc12m-wds",
        "revision": args.revision,
        "tars": [{"file": path.name, "sha256": _sha256_file(path)} for path in tar_paths],
        "manifest_sha256": _sha256_file(manifest_path),
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_files": {p.name: _sha256_file(p) for p in tokenizer_files},
        "tokenizer_vocab_size": vocab_size,
        "image_processing": {
            "patch_size": processor.patch_size,
            "downsample_ratio": processor.downsample_ratio,
            "min_pixels": processor.min_pixels,
            "max_n_token": processor.max_n_token,
            "max_wh_ratio": processor.max_wh_ratio,
            "mean": list(processor.mean),
            "std": list(processor.std),
        },
        "supervision": "BOS + image protocol + caption + EOS; caption and EOS supervised",
        "seq_len": args.seq_len,
        "seed": args.seed,
        "requested_count": args.count,
        "selected_count": len(selected),
        "unique_images": len(seen_hashes),
        "filtered": filtered,
        "candidates_total": len(candidates),
        "token_stats": {
            "total": _summary([e["total"] for e in selected]),
            "supervised": _summary([e["supervised"] for e in selected]),
            "protocol": _summary([e["protocol"] for e in selected]),
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
