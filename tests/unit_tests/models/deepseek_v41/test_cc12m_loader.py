# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract tests for the CC12M caption data entry.

Portable tests run everywhere against the committed mini tokenizer
(``tests/assets/deepseek_v3``: bos=2002, eos=2003) — no private
tokenizer path is involved.  The real V4.1 tokenizer compatibility
(BOS/EOS ids, real captions) is a separate integration test at the
bottom that skips only when the tokenizer is not deployed.
"""

import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest
import torch

from torchtitan_npu.models.deepseek_v41.cc12m_loader import (
    DeepSeekV41Cc12mDataLoader,
    SampleBudgetExceededError,
    _Cc12mCaptionDataset,
    assemble_caption_sequence,
    digest_batches,
)
from torchtitan_npu.models.deepseek_v41.vision_data import (
    IMAGE_END,
    IMAGE_START,
    TEXT,
    ImagePatchProcessor,
    build_image_token_layout,
)

ASSET_IMAGE = Path(__file__).resolve().parents[3] / "assets" / "dsv4_vit_test.jpeg"
MINI_TOKENIZER = Path(__file__).resolve().parents[3] / "assets" / "deepseek_v3"
REAL_TOKENIZER = Path("/data/p00465316/fused/dsv41_tokenizer")
PREPARE_PATH = Path(__file__).resolve().parents[4] / "examples" / "deepseek_v41" / "prepare_cc12m.py"

BOS, EOS = 2002, 2003
VOCAB = 2004  # mini tokenizer: 2001 base + bos/eos added tokens
SEQ_LEN = 512


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    return HuggingFaceTokenizer(tokenizer_path=str(MINI_TOKENIZER))


@pytest.fixture(scope="module")
def prepare():
    spec = importlib.util.spec_from_file_location("prepare_cc12m", PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A tiny dataset root: one image + a manifest with 8 captions."""
    root = tmp_path_factory.mktemp("cc12m")
    images = root / "images"
    images.mkdir()
    shutil.copy(ASSET_IMAGE, images / "img-0.jpg")
    captions = [f"Description number {i} of the image." for i in range(8)]
    manifest_lines = [
        json.dumps({"id": f"shard-a/{i:04d}", "image": "images/img-0.jpg", "caption": caption})
        for i, caption in enumerate(captions)
    ]
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {"root": root, "manifest": manifest, "captions": captions}


def _dataset(env, *, seq_len: int = SEQ_LEN, rank: int = 0, world: int = 1):
    return _Cc12mCaptionDataset(
        manifest_path=str(env["manifest"]),
        data_dir=str(env["root"]),
        tokenizer_path=str(MINI_TOKENIZER),
        seq_len=seq_len,
        vocab_size=VOCAB,
        dp_rank=rank,
        dp_world_size=world,
    )


# ---------------------------------------------------------------------------
# shared assembly function (used by preparation and runtime alike)
# ---------------------------------------------------------------------------


def test_assemble_sequence_layout_and_supervision(tokenizer):
    caption_ids = tokenizer.encode("A black cat on a wooden table.", add_bos=False, add_eos=False)
    ids, types, indices, supervised = assemble_caption_sequence(
        caption_ids=caption_ids,
        grid_hw=(32, 48),
        vocab_size=VOCAB,
        bos_id=BOS,
        eos_id=EOS,
        seq_len=SEQ_LEN,
    )
    assert ids[0] == BOS
    assert ids[-1] == EOS and supervised[-1]
    # Frame: BOS + protocol + caption + EOS, in that order.
    assert sum(supervised) == len(caption_ids) + 1
    # The protocol block matches the reference layout for this grid.
    layout_ids, layout_types, layout_indices = build_image_token_layout(
        [(32, 48)], span_start=0, vocab_size=VOCAB
    )
    assert ids[1 : 1 + layout_ids.numel()] == layout_ids.tolist()
    assert types[1 : 1 + layout_ids.numel()] == layout_types.tolist()
    assert indices[1 : 1 + layout_ids.numel()] == layout_indices.tolist()


def test_assemble_rejects_empty_and_overlong(tokenizer):
    with pytest.raises(ValueError, match="at least one token"):
        assemble_caption_sequence(
            caption_ids=[],
            grid_hw=(8, 8),
            vocab_size=VOCAB,
            bos_id=BOS,
            eos_id=EOS,
            seq_len=SEQ_LEN,
        )
    with pytest.raises(SampleBudgetExceededError):
        assemble_caption_sequence(
            caption_ids=list(range(600)),
            grid_hw=(32, 48),
            vocab_size=VOCAB,
            bos_id=BOS,
            eos_id=EOS,
            seq_len=SEQ_LEN,
        )


def test_target_grid_matches_full_decode_at_boundaries(tmp_path):
    """The shared size decision agrees with the full pixel decode (R4).

    Covers the min_pixels upscale boundary (tiny images), extreme aspect
    ratios and a regular CC12M-like size.
    """
    from PIL import Image

    processor = ImagePatchProcessor()
    cases = [(100, 100), (60, 1600), (1600, 60), (701, 1024), (555, 416)]
    for i, (width, height) in enumerate(cases):
        path = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (width, height), (128, 128, 128)).save(path, format="JPEG")
        _, image_grid = processor.from_path(path)
        expected = (int(image_grid[0]), int(image_grid[1]))
        assert processor.target_grid(height, width) == expected, (width, height)


# ---------------------------------------------------------------------------
# dataset contract
# ---------------------------------------------------------------------------


def test_caption_and_eos_are_the_only_supervised_targets(env, tokenizer):
    input_dict, labels = next(iter(_dataset(env)))
    tokens = input_dict["input"]
    expected = tokenizer.encode(env["captions"][0], add_bos=False, add_eos=False) + [EOS]
    assert labels[labels != -100].tolist() == expected


def test_labels_shifted_exactly_once(env):
    input_dict, labels = next(iter(_dataset(env)))
    tokens, types = input_dict["input"], input_dict["token_types"]
    mask = labels != -100
    assert not mask[-1].item()  # last position never predicts anything
    assert torch.equal(labels[:-1][mask[:-1]], tokens[1:][mask[:-1]])
    assert (labels[:-1][types[1:] >= 0] == -100).all()  # protocol never supervised


def test_image_protocol_block_is_complete_and_indexed(env):
    input_dict, _ = next(iter(_dataset(env)))
    tokens, types, indices = (
        input_dict["input"],
        input_dict["token_types"],
        input_dict["image_feature_indices"],
    )
    proto = (types >= 0).nonzero().squeeze(-1)
    start, end = int(proto[0]), int(proto[-1])
    assert end - start + 1 == proto.numel()  # one contiguous span
    assert int(types[start]) == IMAGE_START and int(types[end]) == IMAGE_END
    grid_h, grid_w = input_dict["image_grid"][0].tolist()
    expected_ids, expected_types, expected_indices = build_image_token_layout(
        [(grid_h, grid_w)], span_start=0, vocab_size=VOCAB
    )
    assert torch.equal(tokens[start : end + 1], expected_ids)
    assert torch.equal(indices[start : end + 1], expected_indices)
    assert (indices[:start] == -1).all() and (indices[end + 1 :] == -1).all()
    n_features = int((expected_indices >= 0).sum())
    assert n_features == (grid_h + 2) // 3 * ((grid_w + 2) // 3)
    assert input_dict["pixel_values"].shape == (grid_h * grid_w, 3 * 14 * 14)
    assert int(indices.max()) < input_dict["pixel_values"].shape[0]


def test_padding_is_uniform_and_budget_mismatch_fails(env):
    input_dict, labels = next(iter(_dataset(env)))
    tokens, types = input_dict["input"], input_dict["token_types"]
    eos_positions = (tokens == EOS).nonzero().squeeze(-1)
    tail = slice(int(eos_positions[-1]) + 1, SEQ_LEN)
    assert (tokens[tail] == 0).all()
    assert (types[tail] == TEXT).all()
    assert (labels[tail] == -100).all()
    # A manifest/config budget mismatch fails loudly instead of truncating.
    with pytest.raises(ValueError, match="exceeds seq_len"):
        next(iter(_dataset(env, seq_len=64)))


def test_rank_shards_disjoint_reproducible_and_validated(env):
    world = 4
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tok = HuggingFaceTokenizer(tokenizer_path=str(MINI_TOKENIZER))
    expected = [
        tok.encode(env["captions"][i], add_bos=False, add_eos=False) + [EOS] for i in range(8)
    ]

    def rank_samples(rank, steps):
        iterator = iter(_dataset(env, rank=rank, world=world))
        return [next(iterator) for _ in range(steps)]

    per_rank = {rank: rank_samples(rank, 2) for rank in range(world)}
    for step in range(2):
        ids = [s[1][s[1] != -100].tolist() for s in (per_rank[rank][step] for rank in range(world))]
        assert len({tuple(i) for i in ids}) == world  # disjoint at every step
    merged = [s for step in range(2) for s in (per_rank[rank][step] for rank in range(world))]
    assert [s[1][s[1] != -100].tolist() for s in merged] == expected  # manifest order


def test_empty_rank_shard_fails_immediately(env, tmp_path):
    # 7 samples over DP8 -> rank 7 empty -> construct-time error (R2).
    small = tmp_path / "manifest.jsonl"
    lines = env["manifest"].read_text(encoding="utf-8").splitlines()[:7]
    small.write_text("\n".join(lines) + "\n", encoding="utf-8")
    small_env = {"root": env["root"], "manifest": small, "captions": env["captions"]}
    with pytest.raises(ValueError, match="empty shard"):
        _dataset(small_env, rank=7, world=8)


def test_invalid_rank_world_fail_immediately(env):
    with pytest.raises(ValueError, match="dp_world_size"):
        _dataset(env, world=0)
    with pytest.raises(ValueError, match="dp_rank"):
        _dataset(env, rank=8, world=8)
    with pytest.raises(ValueError, match="dp_rank"):
        _dataset(env, rank=-1, world=8)


def test_manifest_rejects_duplicate_ids(env, tmp_path):
    line = env["manifest"].read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "manifest.jsonl"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample id"):
        _Cc12mCaptionDataset._read_manifest(str(path))


def test_bad_manifest_entries_fail_loudly(env, tmp_path):
    bad_lines = [
        json.dumps({"id": "a/1", "image": "images/img-0.jpg"}),  # missing caption
        json.dumps({"id": "", "image": "images/img-0.jpg", "caption": "x"}),  # empty id
        json.dumps({"id": "a/3", "image": "images/missing.jpg", "caption": "lost file"}),  # missing image
    ]
    for line in bad_lines:
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(line + "\n", encoding="utf-8")
        bad_env = {"root": env["root"], "manifest": manifest, "captions": ["x"]}
        with pytest.raises((ValueError, FileNotFoundError)):
            next(iter(_dataset(bad_env)))
    empty = tmp_path / "manifest.jsonl"
    empty.write_text("", encoding="utf-8")
    empty_env = {"root": env["root"], "manifest": empty, "captions": ["x"]}
    with pytest.raises(ValueError, match="no samples"):
        _dataset(empty_env)


def test_dataset_verifies_meta_identity(env):
    """A stale or edited subset fails the meta.json cross-check (R2)."""
    manifest = env["manifest"]
    meta = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "tokenizer_files": {},
        "image_processing": {},
    }
    meta_path = env["root"] / "meta.json"
    original = manifest.read_text(encoding="utf-8")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    try:
        _dataset(env)  # hash matches -> constructs fine
        manifest.write_text("\n".join(original.splitlines()[:-1]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="does not match meta.json"):
            _dataset(env)
    finally:
        meta_path.unlink()
        manifest.write_text(original, encoding="utf-8")


def test_loader_config_guards(env):
    base = dict(
        manifest_path=str(env["manifest"]),
        data_dir=str(env["root"]),
        tokenizer_path=str(MINI_TOKENIZER),
    )
    common = dict(dp_world_size=1, dp_rank=0, seq_len=SEQ_LEN, local_batch_size=1)
    with pytest.raises(ValueError, match="local_batch_size"):
        DeepSeekV41Cc12mDataLoader(
            DeepSeekV41Cc12mDataLoader.Config(**base), dp_world_size=1, dp_rank=0, seq_len=SEQ_LEN, local_batch_size=2
        )
    with pytest.raises(ValueError, match="num_workers"):
        DeepSeekV41Cc12mDataLoader(
            DeepSeekV41Cc12mDataLoader.Config(**base, num_workers=2), **common
        )
    with pytest.raises(ValueError, match="manifest_path"):
        DeepSeekV41Cc12mDataLoader(
            DeepSeekV41Cc12mDataLoader.Config(data_dir=base["data_dir"], tokenizer_path=str(MINI_TOKENIZER)),
            **common,
        )
    with pytest.raises(ValueError, match="vocab"):
        DeepSeekV41Cc12mDataLoader(
            DeepSeekV41Cc12mDataLoader.Config(**base, vocab_size=999), **common
        )


# ---------------------------------------------------------------------------
# digest (A/B preflight)
# ---------------------------------------------------------------------------


def test_digest_consumption_accounting(env):
    common = dict(
        manifest_path=str(env["manifest"]),
        data_dir=str(env["root"]),
        tokenizer_path=str(MINI_TOKENIZER),
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
    )
    # 2 steps x 4 ranks (default GBS=world) = 8 samples: exactly the manifest.
    report = digest_batches(**common, world=4, num_batches=2)
    assert report["consumed_samples"] == 8
    assert report["samples_per_step"] == 4
    assert all(len(samples) == 2 for samples in report["ranks"].values())
    # 3 steps would cycle the shard -> rejected.
    with pytest.raises(ValueError, match="consumption check failed"):
        digest_batches(**common, world=4, num_batches=3)
    # Gradient accumulation: GBS=8 over world=4 consumes 8 samples per step.
    report = digest_batches(**common, world=4, num_batches=1, global_batch_size=8)
    assert report["consumed_samples"] == 8
    assert all(len(samples) == 2 for samples in report["ranks"].values())
    # GBS >= world but not a multiple of it is invalid (guards silent floor division).
    with pytest.raises(ValueError, match="multiple of world"):
        digest_batches(**common, world=4, num_batches=1, global_batch_size=6)
    with pytest.raises(ValueError, match="num_batches must be positive"):
        digest_batches(**common, world=4, num_batches=0)


# ---------------------------------------------------------------------------
# offline preparation (examples/deepseek_v41/prepare_cc12m.py)
# ---------------------------------------------------------------------------


def _jpeg_bytes(width=64, height=48) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_tar(path: Path, samples: dict[str, dict[str, bytes | None]]) -> None:
    import tarfile

    with tarfile.open(path, "w") as tar:
        for key, parts in samples.items():
            for ext, data in parts.items():
                if data is None:
                    continue
                info = tarfile.TarInfo(f"{key}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def _scan(prepare, tar_paths, *, encode=None):
    return prepare._scan_tars(
        [Path(p) for p in tar_paths],
        encode=encode or (lambda text: [1, 2, 3]),
        bos_id=BOS,
        eos_id=EOS,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
        processor=ImagePatchProcessor(),
    )


def test_prepare_full_decode_rejects_truncated_jpeg(tmp_path, prepare):
    good = _jpeg_bytes()
    truncated = good[: len(good) // 2]  # intact header, truncated pixel data
    tar = tmp_path / "shard0.tar"
    _write_tar(
        tar,
        {
            "k1": {"jpg": good, "txt": b"a cat"},
            "k2": {"jpg": truncated, "txt": b"a dog"},
        },
    )
    candidates, filtered = _scan(prepare, [tar])
    assert [c["key"] for c in candidates] == ["k1"]
    assert filtered["bad_image"] == 1


def test_prepare_rejects_duplicate_tar_stems(tmp_path, prepare):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "same.tar").write_bytes(b"")
    (dir_b / "same.tar").write_bytes(b"")
    with pytest.raises(SystemExit, match="duplicate tar stem"):
        prepare._resolve_tars([str(dir_a / "same.tar"), str(dir_b / "same.tar")])


def test_prepare_resolves_dedupes_and_sorts_tars(tmp_path, prepare):
    tar_b = tmp_path / "b.tar"
    tar_a = tmp_path / "a.tar"
    tar_b.write_bytes(b"")
    tar_a.write_bytes(b"")
    resolved = prepare._resolve_tars([str(tar_b), str(tar_a), str(tar_a)])
    assert [p.name for p in resolved] == ["a.tar", "b.tar"]


def test_prepare_rejects_duplicate_tar_member(tmp_path, prepare):
    import tarfile

    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tar:
        for _ in range(2):  # two .jpg members under one key
            data = _jpeg_bytes()
            info = tarfile.TarInfo("k1.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with tarfile.open(tar_path, "r") as tar:
        with pytest.raises(ValueError, match="duplicate jpg member"):
            list(prepare._iter_wds_samples(tar))


def test_prepare_scan_uses_shared_budget_and_grid(prepare, tmp_path):
    """The scan's grid/length judgment is the runtime's own functions."""
    from torchtitan_npu.models.deepseek_v41.cc12m_loader import assemble_caption_sequence

    tar = tmp_path / "shard0.tar"
    _write_tar(tar, {"k1": {"jpg": _jpeg_bytes(160, 120), "txt": b"hello world"}})
    candidates, filtered = _scan(prepare, [tar], encode=lambda text: [11, 22])
    assert filtered == {}
    entry = candidates[0]
    processor = ImagePatchProcessor()
    # grid comes from the shared target_grid on the actual header size
    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(_jpeg_bytes(160, 120))) as image:
        width, height = image.size
    grid = processor.target_grid(height, width)
    ids, _, _, supervised = assemble_caption_sequence(
        caption_ids=[11, 22],
        grid_hw=grid,
        vocab_size=VOCAB,
        bos_id=BOS,
        eos_id=EOS,
        seq_len=SEQ_LEN,
        downsample_ratio=processor.downsample_ratio,
    )
    assert entry["total"] == len(ids)
    assert entry["supervised"] == sum(supervised)
    assert entry["protocol"] == len(ids) - 2 - 2


# ---------------------------------------------------------------------------
# real V4.1 tokenizer integration (the only deploy-dependent part)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_TOKENIZER.is_dir(), reason=f"V4.1 tokenizer not deployed at {REAL_TOKENIZER}")
def test_real_tokenizer_sequence_contract(tmp_path):
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tok = HuggingFaceTokenizer(tokenizer_path=str(REAL_TOKENIZER))
    assert tok.get_vocab_size() == 129280
    assert tok.bos_id == 0 and tok.eos_id == 1

    root = tmp_path
    images = root / "images"
    images.mkdir()
    shutil.copy(ASSET_IMAGE, images / "img-0.jpg")
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"id": "shard-x/0001", "image": "images/img-0.jpg", "caption": "A cat sitting on a mat."}
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = _Cc12mCaptionDataset(
        manifest_path=str(manifest),
        data_dir=str(root),
        tokenizer_path=str(REAL_TOKENIZER),
        seq_len=512,
        vocab_size=129280,
        dp_rank=0,
        dp_world_size=1,
    )
    input_dict, labels = next(iter(dataset))
    caption_ids = tok.encode("A cat sitting on a mat.", add_bos=False, add_eos=False)
    expected = caption_ids + [1]
    assert labels[labels != -100].tolist() == expected
    assert input_dict["input"][0].item() == 0  # real BOS
    assert int((input_dict["input"] == 1).sum()) == 1  # exactly one EOS
