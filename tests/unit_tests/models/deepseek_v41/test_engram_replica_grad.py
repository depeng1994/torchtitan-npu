# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed CPU regression for the Host Engram cross-replica reduction."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("ep_size", [1, 2], ids=["replicas_only", "ep2_efsdp2"])
def test_replicas_agree_on_the_summed_sparse_gradient(tmp_path: Path, ep_size: int) -> None:
    """Both replicas must end the step with the same summed gradient.

    Rows are partitioned along EP only, so a replica's SparseAdam would
    otherwise step on its own tokens and the copies would drift apart.
    """
    worker = Path(__file__).with_name("engram_replica_grad_worker.py")
    common_env = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_free_port()),
        "WORLD_SIZE": str(2 * ep_size),
    }
    processes = []
    for rank in range(2 * ep_size):
        env = {**common_env, "RANK": str(rank), "OUT": str(tmp_path / f"rank-{rank}.json")}
        processes.append(
            subprocess.Popen(
                [sys.executable, str(worker)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )

    for process in processes:
        output, _ = process.communicate(timeout=180)
        assert process.returncode == 0, output.decode()

    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2 * ep_size)]
    for rank, report in enumerate(reports):
        assert report["ok"], f"rank {rank}: {report}"
        assert report["rows"] == [0, 2, 4], report
        assert report["row2"] == 11.0 * (rank % ep_size + 1), report
    for ep_rank in range(ep_size):
        assert reports[ep_rank] == reports[ep_rank + ep_size]
