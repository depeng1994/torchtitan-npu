# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU regression guard for extreme long-sequence CP metadata construction.

This test is intentionally model-independent.  It exercises the common
``CPVarlenMetadata.from_global`` path with the 1M-token / CP128 HeadTail case
that exposed a severe performance regression when full-sequence load-balance
work was accidentally repeated once per CP rank.
"""

import importlib.util
import time
from pathlib import Path

import pytest
import torch
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torchtitan.models.common.attention import VarlenMetadata


_SEQ_LEN = 1 << 20  # 1,048,576 tokens
_CP_SIZE = 128
_DOC_LEN = 512
_BATCH_SIZE = 1

# The optimized path is expected to stay comfortably below this on a single
# CPU thread.  The deliberately wide margin keeps the guard robust on shared CI
# hosts while still catching algorithmic regressions such as rebuilding or
# sorting a full 1M-token permutation once per CP rank.
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


class _CountingHeadTailLoadBalancer(_HeadTailLoadBalancer):
    """Count global HeadTail layout construction calls."""

    def __init__(self, seq_length: int, world_size: int):
        super().__init__(seq_length=seq_length, world_size=world_size, device="cpu")
        self.generate_indices_calls = 0

    def _generate_indices(self, restore: bool = False):
        self.generate_indices_calls += 1
        return super()._generate_indices(restore=restore)


def _build_global_varlen_metadata() -> VarlenMetadata:
    # 2048 deterministic packed documents, each 512 tokens.  Using the same
    # tensor object for Q/K boundaries is part of the self-attention contract.
    cu_seqlens = torch.arange(
        0,
        _SEQ_LEN + _DOC_LEN,
        _DOC_LEN,
        dtype=torch.int32,
    )
    return VarlenMetadata(
        cu_seq_q=cu_seqlens,
        cu_seq_k=cu_seqlens,
        max_q=_DOC_LEN,
        max_k=_DOC_LEN,
    )


@pytest.mark.cpu
def test_varlen_cp_1m_headtail_metadata_performance():
    """Guard 1M / CP128 metadata against per-rank full-sequence regressions."""
    cp_varlen_metadata = _load_cp_varlen_metadata_class()
    global_metadata = _build_global_varlen_metadata()
    load_balancer = _CountingHeadTailLoadBalancer(_SEQ_LEN, _CP_SIZE)

    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Keep one-time dispatcher/thread-pool startup out of the performance
        # budget; the test is intended to guard metadata algorithmic complexity.
        _ = torch.arange(16, dtype=torch.int32) + 1

        start = time.perf_counter()

        # Derive the 1M-token HeadTail layout exactly once.  All 128 rank-local
        # metadata builders below must reuse these tensors.
        rearrange_indices = load_balancer._generate_indices(restore=False).to(torch.int32)
        rearrange_flat = rearrange_indices.reshape(-1)
        restore_flat = torch.empty_like(rearrange_flat)
        restore_flat[rearrange_flat] = torch.arange(
            _SEQ_LEN,
            dtype=rearrange_flat.dtype,
        )
        restore_indices = restore_flat.view_as(rearrange_indices)

        layout_done = time.perf_counter()
        total_local_q = 0

        # Reproduce the extreme all-rank plan-construction case.  Do not retain
        # all metadata objects simultaneously; production code only needs the
        # derived geometry, and retaining them would turn this into a memory test.
        for rank in range(_CP_SIZE):
            metadata = cp_varlen_metadata.from_global(
                global_metadata,
                _FakeCPMesh(_CP_SIZE, rank),
                batch_size=_BATCH_SIZE,
                seq_length=_SEQ_LEN,
                load_balancer=load_balancer,
                precomputed_rearrange_indices=rearrange_indices,
                precomputed_restore_indices=restore_indices,
            )
            total_local_q += int(metadata.cu_seq_q[-1])

            # Fail early if an O(CP * S log S)-style regression is restored,
            # rather than making a slow CI worker finish all 128 ranks first.
            if rank % 8 == 7:
                elapsed = time.perf_counter() - start
                assert elapsed < _MAX_SECONDS, (
                    "1M / CP128 varlen metadata exceeded the regression budget "
                    f"after rank {rank}: {elapsed:.3f}s > {_MAX_SECONDS:.1f}s"
                )

        end = time.perf_counter()
    finally:
        torch.set_num_threads(old_num_threads)

    layout_time = layout_done - start
    metadata_time = end - layout_done
    total_time = end - start

    # Functional sanity: the 128 local Q shards cover the full 1M sequence.
    assert total_local_q == _SEQ_LEN

    # Deterministic complexity guard: global HeadTail layout generation must not
    # creep back into the per-rank metadata loop.
    assert load_balancer.generate_indices_calls == 1, (
        "the 1M HeadTail layout was regenerated while deriving rank-local "
        "metadata; reuse precomputed rearrange/restore indices instead"
    )

    assert total_time < _MAX_SECONDS, (
        "1M / CP128 varlen metadata performance regression:\n"
        f"  layout:   {layout_time:.3f}s\n"
        f"  metadata: {metadata_time:.3f}s\n"
        f"  total:    {total_time:.3f}s\n"
        f"  budget:   {_MAX_SECONDS:.1f}s"
    )
