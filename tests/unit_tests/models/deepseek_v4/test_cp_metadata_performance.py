# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coarse CPU complexity guard for DeepSeek-V4 CP metadata planning."""

import pytest

from tests.unit_tests.distributed.cp_metadata_perf_utils import (
    CP_SIZE_1M,
    SHARD_LEN_1M,
    run_1m_cp_metadata_perf_guard,
)
from torchtitan_npu.models.deepseek_v4.token_dispatcher import build_cp_plan


_MAX_SECONDS = 8.0


@pytest.mark.cpu
def test_dsv4_cp_plan_1m_headtail_metadata_complexity():
    """Guard the full DSV4 planner against 1M / CP128 complexity regressions."""

    def _build_dsv4_plan(global_metadata, load_balancer):
        return build_cp_plan(
            global_metadata,
            load_balancer,
            rank=0,
            cp_size=CP_SIZE_1M,
            shard_len=SHARD_LEN_1M,
            window_size=128,
            ratios=[1, 4, 128],
        )

    (cp_metadata, plans, window), _ = run_1m_cp_metadata_perf_guard(
        _build_dsv4_plan,
        budget_seconds=_MAX_SECONDS,
        label="DeepSeek-V4 build_cp_plan",
    )

    assert int(cp_metadata.cu_seq_q[-1]) == SHARD_LEN_1M
    assert set(plans) == {1, 4, 128}
    assert window is not None
