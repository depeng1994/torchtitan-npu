# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coarse CPU complexity guard for the common CP varlen primitive."""

import pytest

from tests.unit_tests.distributed.cp_metadata_perf_utils import (
    BATCH_SIZE_1M,
    CP_SIZE_1M,
    SEQ_LEN_1M,
    SHARD_LEN_1M,
    run_1m_cp_metadata_perf_guard,
)
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
    CPVarlenMetadata,
)


_MAX_SECONDS = 2.5


class _FakeCPMesh:
    """Minimal 1-D DeviceMesh contract required by from_global()."""

    ndim = 1

    def __init__(self, world_size: int, rank: int):
        self._world_size = world_size
        self._rank = rank

    def size(self):
        return self._world_size

    def get_local_rank(self):
        return self._rank


@pytest.mark.cpu
def test_varlen_cp_1m_headtail_metadata_complexity():
    """Guard one common current-rank metadata build at 1M / CP128."""
    # Prevent the common varlen primitive from regressing to multi-second full-sequence work.

    def _build_current_rank_metadata(global_metadata, load_balancer):
        return CPVarlenMetadata.from_global(
            global_metadata,
            _FakeCPMesh(CP_SIZE_1M, 0),
            batch_size=BATCH_SIZE_1M,
            seq_length=SEQ_LEN_1M,
            load_balancer=load_balancer,
        )

    metadata, _ = run_1m_cp_metadata_perf_guard(
        _build_current_rank_metadata,
        budget_seconds=_MAX_SECONDS,
        label="common CPVarlenMetadata.from_global",
    )

    assert int(metadata.cu_seq_q[-1]) == SHARD_LEN_1M
