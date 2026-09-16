# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 Engram Host table and sparse optimizer overrides."""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v41.engram_host import HostEngramTable

if TYPE_CHECKING:
    from .ascendc import HostOffloadEngramTable


@override(
    target=HostEngramTable.Config,
    exact=True,
    description=(
        "Keep the EP-local Engram table and SparseAdam state on Host and use CANN EngramFetch/EngramFetchGrad"
    ),
)
def host_offload(
    cfg: HostEngramTable.Config,
    num_max_tokens_per_rank: int,
    pin_memory: bool = True,
) -> "HostOffloadEngramTable.Config":
    """Replace Torch Host lookup with CANN while retaining sparse training.

    Requires ``EP>1``. The CPU shard is registered once when the operator pack
    supports direct storage registration; older packs refresh it per forward.
    ``num_max_tokens_per_rank`` counts flattened hash-row requests across every
    token and N-gram head, and must cover the largest batch used by the run.
    """
    from .ascendc import HostOffloadEngramTable

    return derive(
        cfg,
        HostOffloadEngramTable.Config,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        pin_memory=pin_memory,
    )


__all__ = ["host_offload"]
