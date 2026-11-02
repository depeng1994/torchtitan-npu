#!/usr/bin/env python3
"""Trusted execution-domain entrypoint; invoke with python -I from the Runner Bundle.

Source code is deliberately executed only inside a leased NPU container. This
module and its catalog belong to the pinned control release, not the PR tree.
No platform token is accepted or forwarded. PASS evidence is not resource cleanup
or atomic merge permission; those remain Worker control-domain responsibilities.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SHA = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}\Z")
FORBIDDEN_ENV = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITCODE_API_TOKEN",
    "LITE_ACTIONS_GITHUB_TOKEN",
    "LITE_ACTIONS_GITCODE_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
)
PORT_KEYS = (
    "LITE_CI_MASTER_PORT",
    "LITE_CI_HCCL_IF_BASE_PORT",
    "LITE_CI_HCCL_HOST_PORT_RANGE",
    "LITE_CI_HCCL_NPU_PORT_RANGE",
)


def json_file(path: Path, limit: int = 131072):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("missing, unsafe, or oversized trusted input")
    return json.loads(path.read_text())


def source_path(root: Path, value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("unsafe source interface path")
    result = root.joinpath(*path.parts)
    if (
        result.is_symlink()
        or not result.is_file()
        or not result.resolve().is_relative_to(root)
    ):
        raise ValueError("required source interface absent")
    return result


def catalog_case(test_id: str):
    document = json_file(ROOT / "catalog.json")
    if (
        document.get("schema") != 1
        or document.get("runner_protocol") != "lite-gitcode-torchtitan-v1"
    ):
        raise ValueError("unrecognized trusted runner protocol")
    case, profile = document["cases"][test_id], document["profiles"][test_id]
    if case["worker"] != "blue" or case["kind"] == "worker_probe":
        raise ValueError("OS probes run on the Worker, never in this NPU runner")
    if (
        case["docker"] != "cann_master_torchtitan_master_xyj"
        or case["target_pool"] != "a3_shared_blue"
    ):
        raise ValueError("unapproved TorchTitan execution environment")
    return case, profile


def identities(args, case):
    if any(not SHA.fullmatch(value) for value in (args.source_sha, args.tree_sha)):
        raise ValueError("invalid merge source identity")
    if (
        any(not SAFE_ID.fullmatch(value) for value in (args.lease_id, args.task_marker))
        or args.fencing_token < 1
    ):
        raise ValueError("invalid captured lease identity")
    resources = case["resources"]
    if (args.node_count, args.devices_per_node) != (
        resources["nodes"],
        resources["devices_per_node"],
    ):
        raise ValueError("leased topology differs from trusted case")
    if not 0 <= args.node_rank < args.node_count:
        raise ValueError("invalid node rank")
    ipaddress.IPv4Address(args.master_addr)
    if not 1024 <= args.master_port <= 65535:
        raise ValueError("invalid leased master port")
    if os.environ.get("LITE_CI_JOB_ID") != args.task_marker:
        raise ValueError("process ownership marker mismatch")
    if any(os.environ.get(key) for key in FORBIDDEN_ENV):
        raise ValueError("platform credential is forbidden in the execution domain")
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    if (
        len(devices) != args.devices_per_node
        or len(set(devices)) != len(devices)
        or any(not x.isdecimal() or not 0 <= int(x) < 256 for x in devices)
    ):
        raise ValueError("visible devices differ from the physical lease")
    for key in PORT_KEYS:
        if not os.environ.get(key):
            raise ValueError("complete dispatcher-issued port lease required")
    for key in PORT_KEYS[:2]:
        value = os.environ[key]
        if (
            not value.isascii()
            or not value.isdecimal()
            or not 1024 <= int(value) <= 65535
        ):
            raise ValueError("invalid leased scalar port")
    for key in PORT_KEYS[2:]:
        match = re.fullmatch(r"([0-9]{4,5})-([0-9]{4,5})", os.environ[key])
        if not match or not 1024 <= int(match[1]) <= int(match[2]) <= 65535:
            raise ValueError("invalid leased socket range")
    if int(os.environ["LITE_CI_MASTER_PORT"]) != args.master_port:
        raise ValueError("master port binding mismatch")
    return {
        "schema": 1,
        "test_id": args.test_id,
        "source_sha": args.source_sha,
        "tree_sha": args.tree_sha,
        "lease_id": args.lease_id,
        "fencing_token": args.fencing_token,
        "task_marker": args.task_marker,
        "node_rank": args.node_rank,
        "world_size": args.node_count * args.devices_per_node,
    }


def environment(args, profile, identity, source, output):
    # The outer Worker starts a credential-free environment and sources the
    # approved toolkit. Keep that runtime, then apply only the reviewed profile.
    result = dict(os.environ)
    result.update(profile["env_vars"])
    result.update(
        {
            "LITE_GITCODE_SOURCE_ROOT": str(source),
            "LITE_GITCODE_OUTPUT": str(output),
            "LITE_GITCODE_IDENTITY": json.dumps(identity, sort_keys=True),
            "LITE_GITCODE_EXPECTED_MODEL": "torchtitan_npu.models.deepseek_v4",
            "NNODES": str(args.node_count),
            "NGPU": str(args.devices_per_node),
            "NODE_RANK": str(args.node_rank),
            "MASTER_ADDR": args.master_addr,
            "MASTER_PORT": str(args.master_port),
            "HCCL_IF_BASE_PORT": os.environ["LITE_CI_HCCL_IF_BASE_PORT"],
            "HCCL_HOST_SOCKET_PORT_RANGE": os.environ["LITE_CI_HCCL_HOST_PORT_RANGE"],
            "HCCL_NPU_SOCKET_PORT_RANGE": os.environ["LITE_CI_HCCL_NPU_PORT_RANGE"],
            "TRAIN_FILE": "_lite_gitcode_train",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # source/ cannot shadow the reserved wrapper name (Prepare and Worker both
    # check this). Source business modules stay in the proposed merged tree.
    dependency_path = os.environ.get("PYTHONPATH", "")
    result["PATH"] = (
        str(output / "trusted-bin") + os.pathsep + result.get("PATH", "/usr/bin:/bin")
    )
    result["PYTHONPATH"] = os.pathsep.join(
        x for x in (str(ROOT), str(source), dependency_path) if x
    )
    for key in FORBIDDEN_ENV:
        result.pop(key, None)
    return result


def command(args, case, profile, source, output):
    if case["kind"] == "npu_smoke":
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(args.node_count),
            "--nproc-per-node",
            str(args.devices_per_node),
            "--node-rank",
            str(args.node_rank),
            "--master-addr",
            args.master_addr,
            "--master-port",
            str(args.master_port),
            str(ROOT / "smoke_rank.py"),
            "--payload",
            str(source_path(source, profile["payload"])),
        ]
    return [
        "bash",
        str(source_path(source, profile["train_script"])),
        "--dump_folder",
        str(output / args.test_id / "test_run"),
        *profile["train_args"],
        "--metrics.save_tb_folder=tb_phase_0",
    ]


def finite_losses(run_root: Path, expected_steps: list[int]):
    # Imported solely on the NPU execution/verification node, never in Prepare.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    folder = run_root / "tb_phase_0"
    children = [x for x in folder.iterdir() if x.is_dir() and not x.is_symlink()]
    if len(children) != 1:
        raise ValueError("expected exactly one rank-0 TensorBoard event directory")
    events = EventAccumulator(str(children[0]), size_guidance={"scalars": 0})
    events.Reload()
    values = {}
    for scalar in events.Scalars("loss_metrics/global_avg_loss"):
        if scalar.step in values or not math.isfinite(scalar.value):
            raise ValueError("duplicate/non-finite loss")
        values[scalar.step] = scalar.value
    if sorted(values) != expected_steps:
        raise ValueError("missing or unexpected training steps")
    return values


def verify(args, case, profile, source, output, identity):
    if case["kind"] == "npu_smoke":
        if args.node_rank == 0:
            evidence = json_file(output / "smoke-rank0.json")
            expected = {
                **identity,
                "status": "PASS",
                "all_reduce_sum": identity["world_size"]
                * (identity["world_size"] - 1)
                // 2,
                "payload_path": str(source_path(source, profile["payload"])),
            }
            if any(evidence.get(k) != v for k, v in expected.items()):
                raise ValueError("HCCL source/topology/lease evidence mismatch")
        return {
            "all_reduce_sum": identity["world_size"] * (identity["world_size"] - 1) // 2
        }
    paths = []
    for rank in range(
        args.node_rank * args.devices_per_node,
        (args.node_rank + 1) * args.devices_per_node,
    ):
        evidence = json_file(output / f"module-path-rank-{rank}.json")
        if (
            any(evidence.get(k) != v for k, v in identity.items())
            or evidence.get("rank") != rank
        ):
            raise ValueError("training module-path identity mismatch")
        for name in ("package_path", "model_path", "training_path"):
            value = Path(evidence[name])
            if not value.is_absolute() or not value.resolve().is_relative_to(source):
                raise ValueError(
                    "actual training module did not load from the merged source"
                )
        paths.append(evidence)
    losses = (
        finite_losses(output / args.test_id / "test_run", profile["expected_steps"])
        if args.node_rank == 0
        else {}
    )
    return {
        "rank_module_paths": paths,
        "rank0_losses": losses,
        "numerical_accuracy_alignment": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("launch", "verify"))
    for flag in (
        "test-id",
        "source-root",
        "output-root",
        "source-sha",
        "tree-sha",
        "lease-id",
        "task-marker",
        "master-addr",
    ):
        parser.add_argument("--" + flag, required=True)
    for flag in (
        "node-rank",
        "node-count",
        "devices-per-node",
        "master-port",
        "fencing-token",
    ):
        parser.add_argument("--" + flag, type=int, required=True)
    args = parser.parse_args(argv)
    case, profile = catalog_case(args.test_id)
    identity = identities(args, case)
    source, output = Path(args.source_root), Path(args.output_root)
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise ValueError("verified source directory required")
    source = source.resolve()
    if ROOT.is_relative_to(source) or not output.is_absolute() or output.is_symlink():
        raise ValueError("trusted runner, source, and output must be separate")
    if output.resolve().is_relative_to(source) or output.resolve().is_relative_to(ROOT):
        raise ValueError("output cannot overwrite source or trusted runner")
    for path in profile["source_paths"]:
        source_path(source, path)
    for name in (
        "sitecustomize.py",
        "usercustomize.py",
        "_lite_gitcode_train.py",
        "_lite_gitcode_train",
    ):
        if (source / name).exists():
            raise ValueError("source shadows reserved trusted bootstrap name")
    if args.phase == "launch":
        output.mkdir(parents=True, exist_ok=False)
        if case["kind"] == "train":
            launcher = output / "trusted-bin"
            launcher.mkdir(mode=0o700)
            shim = launcher / "torchrun"
            shim.write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(sys.executable)
                + " -I "
                + shlex.quote(str(ROOT / "torchrun_lease.py"))
                + ' "$@"\n'
            )
            shim.chmod(0o700)
        result = subprocess.run(
            command(args, case, profile, source, output),
            cwd=source,
            env=environment(args, profile, identity, source, output),
            timeout=case["execution_timeout_seconds"],
            check=False,
        )
        if result.returncode:
            return result.returncode
        (output / "launch-exit.json").write_text(
            json.dumps({**identity, "exit_code": 0}, sort_keys=True) + "\n"
        )
        return 0
    exit_evidence = json_file(output / "launch-exit.json")
    if exit_evidence != {**identity, "exit_code": 0}:
        raise ValueError("all-node launch success evidence is missing or inconsistent")
    evidence = {
        **identity,
        "status": "PASS",
        "exit_code": 0,
        **verify(args, case, profile, source, output, identity),
    }
    (output / "case-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "test_id": args.test_id,
                "node_rank": args.node_rank,
                "status": "PASS",
                "source_sha": args.source_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
