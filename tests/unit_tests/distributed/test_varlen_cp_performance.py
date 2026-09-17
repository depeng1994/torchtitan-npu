# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU regression guard for the common long-sequence CP varlen primitive."""

import importlib.util
from pathlib import Path

import pytest
import torch

from tests.unit_tests.distributed.cp_metadata_perf_utils import (
    BATCH_SIZE_1M,
    CP_SIZE_1M,
    SEQ_LEN_1M,
    run_1m_cp_metadata_perf_guard,
)


_MAX_SECONDS = 5.0

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VARLEN_CP_PATH = (
    _REPO_ROOT
    / "torchtitan_npu"
    / "patches"
    / "torchtitan"
    / "distributed"
    / "varlen_cp.py"
)


def _load_cp_varlen_metadata_class():
    """Load the backported common CP metadata implementation without model imports."""
    spec = importlib.util.spec_from_file_location("varlen_cp_perf_guard", _VARLEN_CP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load CP varlen module from {_VARLEN_CP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CPVarlenMetadata


class _FakeCPMesh:
    """Minimal 1-D ``DeviceMesh`` contract required by ``from_global``."""

    ndim = 1

    def __init__(self, world_size: int, rank: int):
        self._world_size = world_size
        self._rank = rank

    def size(self):
        return self._world_size

    def get_local_rank(self):
        return self._rank


@pytest.mark.cpu
def test_varlen_cp_1m_headtail_metadata_performance():
    """Guard the common primitive against per-rank full-sequence regressions."""
    cp_varlen_metadata = _load_cp_varlen_metadata_class()

    def _build_all_rank_metadata(global_metadata, load_balancer):
        rearrange_indices = load_balancer._generate_indices(restore=False).to(torch.int32)
        rearrange_flat = rearrange_indices.reshape(-1)

        # Invert the global permutation once with O(S) scatter.
        restore_flat = torch.empty_like(rearrange_flat)
        restore_flat[rearrange_flat] = torch.arange(
            SEQ_LEN_1M,
            dtype=rearrange_flat.dtype,
        )
        restore_indices = restore_flat.view_as(rearrange_indices)

        total_local_q = 0
        for rank in range(CP_SIZE_1M):
            metadata = cp_varlen_metadata.from_global(
                global_metadata,
                _FakeCPMesh(CP_SIZE_1M, rank),
                batch_size=BATCH_SIZE_1M,
                seq_length=SEQ_LEN_1M,
                load_balancer=load_balancer,
                precomputed_rearrange_indices=rearrange_indices,
                precomputed_restore_indices=restore_indices,
            )
            total_local_q += int(metadata.cu_seq_q[-1])
        return total_local_q

    total_local_q, _ = run_1m_cp_metadata_perf_guard(
        _build_all_rank_metadata,
        budget_seconds=_MAX_SECONDS,
        label="common CPVarlenMetadata",
    )

    # Functional sanity: the 128 local Q shards cover the full 1M sequence.
    assert total_local_q == SEQ_LEN_1M
