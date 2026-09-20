# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CPU complexity guard for extreme long-sequence CP metadata.

This is intentionally a coarse regression guard, not a benchmark. The fixed
1M / CP128 shape is large enough to catch accidental O(CP * S) orchestration
while the wall-clock budgets stay loose enough for normal CI host variance.

Model-specific CP implementations should time their top-level metadata builder
through run_1m_cp_metadata_perf_guard so model-owned work is covered in
addition to the common CPVarlenMetadata primitive. When DeepSeek-V4.1 CP
lands, its top-level CP metadata builder should reuse this harness as well.
"""

import time
from collections.abc import Callable
from typing import Any

import torch
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torchtitan.models.common.attention import VarlenMetadata


SEQ_LEN_1M = 1 << 20
CP_SIZE_1M = 128
DOC_LEN_1M = 512
BATCH_SIZE_1M = 1
SHARD_LEN_1M = SEQ_LEN_1M // CP_SIZE_1M


class CountingHeadTailLoadBalancer(_HeadTailLoadBalancer):
    """HeadTail load balancer with a full-layout generation counter."""

    def __init__(
        self,
        seq_length: int = SEQ_LEN_1M,
        world_size: int = CP_SIZE_1M,
    ):
        super().__init__(
            seq_length=seq_length,
            world_size=world_size,
            device="cpu",
        )
        self.forward_generate_calls = 0

    def _generate_indices(self, restore: bool = False):
        if not restore:
            self.forward_generate_calls += 1
        return super()._generate_indices(restore=restore)


def build_1m_global_varlen_metadata() -> VarlenMetadata:
    """Build deterministic 1M metadata: 2048 packed docs x 512 tokens."""
    cu_seqlens = torch.arange(
        0,
        SEQ_LEN_1M + DOC_LEN_1M,
        DOC_LEN_1M,
        dtype=torch.int32,
    )
    return VarlenMetadata(
        cu_seq_q=cu_seqlens,
        cu_seq_k=cu_seqlens,
        max_q=DOC_LEN_1M,
        max_k=DOC_LEN_1M,
    )


def run_1m_cp_metadata_perf_guard(
    builder: Callable[[VarlenMetadata, CountingHeadTailLoadBalancer], Any],
    *,
    budget_seconds: float,
    label: str,
) -> tuple[Any, float]:
    """Run one 1M / CP128 metadata build on one CPU thread.

    The budget is deliberately coarse. It guards algorithmic-complexity
    regressions, for example repeating full-sequence work once per CP rank,
    not small timing changes.
    """
    global_metadata = build_1m_global_varlen_metadata()
    load_balancer = CountingHeadTailLoadBalancer()

    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _ = torch.arange(16, dtype=torch.int32) + 1

        start = time.perf_counter()
        result = builder(global_metadata, load_balancer)
        elapsed = time.perf_counter() - start
    finally:
        torch.set_num_threads(old_num_threads)

    assert load_balancer.forward_generate_calls == 1, (
        f"{label}: the full 1M HeadTail layout was generated "
        f"{load_balancer.forward_generate_calls} times; full-sequence layout "
        "work must not be repeated per CP rank"
    )
    assert elapsed < budget_seconds, (
        f"{label}: 1M / CP128 metadata complexity regression: "
        f"{elapsed:.3f}s >= {budget_seconds:.1f}s budget"
    )
    return result, elapsed
