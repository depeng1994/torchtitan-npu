#!/usr/bin/env python3
"""Build only from an already verified control checkout, never from a PR checkout.

The caller proves U=C and records the computed digest in its reviewed deployment
policy. This builder does not fetch, dispatch, allocate devices or grant release
approval. No Python from the proposed source is imported during bundle creation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
FILES = (
    "catalog.json",
    "runner.py",
    "smoke_rank.py",
    "_lite_gitcode_train.py",
    "torchrun_lease.py",
)
EXPECTED = {
    "worker_os_blue": ("worker_probe", 0, 0, 90),
    "dispatch_smoke_8p": ("npu_smoke", 1, 8, 75),
    "dispatch_smoke_16p": ("npu_smoke", 2, 8, 75),
    "dsv4_flash_a3_8p_example": ("train", 1, 8, 7200),
    "dsv4_flash_a3_8p_adamw": ("train", 1, 8, 7200),
    "dsv4_flash_a3_16p_example": ("train", 2, 8, 7200),
}
ENVIRONMENT = {
    "ASCEND_SET_ENV_PATH",
    "HF_ASSETS_PATH",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "CKPT_INIT_LOAD_PATH",
    "CONFIG",
    "OPTIMIZER_OVERRIDES",
    "TORCHINDUCTOR_NPU_BACKEND",
}


def validate_catalog(document):
    if (
        document.get("schema") != 1
        or document.get("runner_protocol") != "lite-gitcode-torchtitan-v1"
    ):
        raise ValueError("invalid control catalog version")
    if set(document["cases"]) != set(EXPECTED) or set(document["profiles"]) != set(
        EXPECTED
    ):
        raise ValueError("complete TorchTitan Blue catalog required")
    for test_id, expected in EXPECTED.items():
        case, profile = document["cases"][test_id], document["profiles"][test_id]
        resources = case["resources"]
        actual = (
            case["kind"],
            resources["nodes"],
            resources["devices_per_node"],
            case["execution_timeout_seconds"],
        )
        if actual != expected or case["worker"] != "blue" or case["test_id"] != test_id:
            raise ValueError("control case topology/timeout/owner mismatch")
        probe = case["kind"] == "worker_probe"
        if (case["target_pool"], case["docker"]) != (
            (None, None)
            if probe
            else ("a3_shared_blue", "cann_master_torchtitan_master_xyj")
        ):
            raise ValueError("control case environment mismatch")
        if set(profile) != {
            "env_vars",
            "source_paths",
            "payload",
            "train_script",
            "train_args",
            "expected_steps",
        }:
            raise ValueError("unexpected runner profile fields")
        if set(profile["env_vars"]) - ENVIRONMENT:
            raise ValueError(
                "runner profile contains an unapproved environment variable"
            )
        for value in list(profile["env_vars"].values()) + profile["train_args"]:
            if (
                not isinstance(value, str)
                or len(value) > 1024
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("invalid runner profile text")
        for value in profile["source_paths"]:
            path = PurePosixPath(value)
            if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError("unsafe source interface path")
        if case["kind"] == "train" and (
            profile["expected_steps"] != [1, 2, 3, 4, 5]
            or profile["train_script"] not in profile["source_paths"]
        ):
            raise ValueError("training profile must preserve the five-step recipe")
        if (
            case["kind"] == "npu_smoke"
            and profile["payload"] not in profile["source_paths"]
        ):
            raise ValueError("source smoke interface must be declared")
    if document["pipelines"] != {
        "smoke": ["worker_os_blue", "dispatch_smoke_8p", "dispatch_smoke_16p"],
        "nightly": [
            "dsv4_flash_a3_8p_example",
            "dsv4_flash_a3_8p_adamw",
            "dsv4_flash_a3_16p_example",
        ],
    }:
        raise ValueError("pipeline cases differ from reviewed required sets")
    return document


def build(root=ROOT):
    root = Path(root)
    validate_catalog(json.loads((root / "catalog.json").read_text()))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name in FILES:
                path = root / name
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or not 0 < path.stat().st_size <= 128 * 1024
                ):
                    raise ValueError("missing or oversized trusted runner file")
                content = path.read_bytes()
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o644
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ""
                archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build()
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "runner_sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "files": list(FILES),
                "scope": "CONTROL_BUNDLE_BUILD_ONLY_NOT_RELEASE_APPROVAL",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
