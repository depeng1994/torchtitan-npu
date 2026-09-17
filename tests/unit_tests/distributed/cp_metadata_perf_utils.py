# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CPU performance guard for extreme long-sequence CP metadata.

Model-specific CP implementations should reuse ``run_1m_cp_metadata_perf_guard``
and put their full metadata builder inside the timed callback. This keeps the
1M / CP128 stress shape identical across models while letting each model guard
its own orchestration on top of the common ``CPVarlenMetadata`` primitive.
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
    """HeadTail load balancer with a structural complexity counter."""

    def __init__(self, seq_length: int = SEQ_LEN_1M, world_size: int = CP_SIZE_1M):
        super().__init__(seq_length=seq_length, world_size=world_size, device="cpu")
        self.generate_indices_calls = 0

    def _generate_indices(self, restore: bool = False):
        self.generate_indices_calls += 1
        return super()._generate_indices(restore=restore)


def build_1m_global_varlen_metadata() -> VarlenMetadata:
    """Build deterministic 1M packed-document metadata without a tokenizer."""
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
    """Time one 1M / CP128 HeadTail metadata builder on a single CPU thread.

    The callback owns the complete metadata construction being guarded. It
    receives deterministic global varlen metadata plus a counting HeadTail load
    balancer. A model-level test should call its top-level CP metadata builder
    here, not only the common primitive.

    Besides wall time, this enforces the key algorithmic invariant exposed by
    the original regression: the full 1M HeadTail permutation may be generated
    once for the whole plan, never once per CP rank.
    """
    global_metadata = build_1m_global_varlen_metadata()
    load_balancer = CountingHeadTailLoadBalancer()

    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Keep one-time CPU dispatcher/thread-pool startup out of the budget.
        _ = torch.arange(16, dtype=torch.int32) + 1

        start = time.perf_counter()
        result = builder(global_metadata, load_balancer)
        elapsed = time.perf_counter() - start
    finally:
        torch.set_num_threads(old_num_threads)

    assert load_balancer.generate_indices_calls == 1, (
        f"{label}: the 1M HeadTail layout was generated "
        f"{load_balancer.generate_indices_calls} times; full-sequence layout "
        "work must be shared across CP128 rather than repeated per rank"
    )
    assert elapsed < budget_seconds, (
        f"{label}: 1M / CP128 metadata performance regression: "
        f"{elapsed:.3f}s >= {budget_seconds:.1f}s budget"
    )
    return result, elapsed
