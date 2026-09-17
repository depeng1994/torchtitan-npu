# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel planning and token dispatch for DeepSeek-V4."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import cast

import spmd_types as spmd
import torch
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _LoadBalancer,
)
from torchtitan.config import Configurable
from torchtitan.distributed.utils import get_spmd_backend

from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata

from .metadata import CompressedBlockLayout

__all__ = [
    "CPTokenDispatcher",
    "ExchangePlan",
    "WindowPlan",
    "build_cp_plan",
    "segment_structure",
]


class _RankMesh:
    """Minimal DeviceMesh contract needed by ``CPVarlenMetadata.from_global``."""

    ndim = 1

    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


class _CachedLoadBalancer:
    """DSV4-local view that reuses one already-built load-balancer layout."""

    def __init__(self, rearrange_indices: torch.Tensor):
        self._rearrange_indices = rearrange_indices

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        if restore:
            return torch.argsort(self._rearrange_indices, dim=-1)
        return self._rearrange_indices


# ---------------------------------------------------------------------------
# Plan builder (pure derivation, no plan-time communication)
# ---------------------------------------------------------------------------


def segment_structure(cp_meta) -> list[tuple[int, int, int, int]]:
    """Return ``(doc_start, seg_len, seqlen_k, p0)`` for non-empty segments."""
    cu_q = cp_meta.cu_seq_q.cpu().tolist()
    cu_k = cp_meta.cu_seq_k.cpu().tolist()
    kg = cp_meta.k_global_gather_indices.cpu().tolist()
    segs: list[tuple[int, int, int, int]] = []
    for s in range(len(cu_q) - 1):
        seg_len = cu_q[s + 1] - cu_q[s]
        if seg_len == 0:
            continue
        seqlen_k = cu_k[s + 1] - cu_k[s]
        segs.append((kg[cu_k[s]], seg_len, seqlen_k, seqlen_k - seg_len))
    return segs


def _derive_all_rank_segments(
    global_cu: list[int],
    rearrange: list[int],
    restore: list[int],
    *,
    cp_size: int,
    shard_len: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Derive all-rank segment geometry directly from the global packed stream.

    ``rearrange`` maps permuted stream positions to original packed positions;
    ``restore`` is its inverse. This reproduces ``segment_structure`` without
    constructing CP-1 temporary ``CPVarlenMetadata`` objects or materializing
    their device tensors back to the host.
    """
    doc_starts = set(global_cu[1:-1])
    segs_all: list[list[tuple[int, int, int, int]]] = []

    for rank in range(cp_size):
        q = rearrange[rank * shard_len : (rank + 1) * shard_len]
        if not q:
            segs_all.append([])
            continue

        segs: list[tuple[int, int, int, int]] = []
        first = prev = q[0]
        seg_len = 1

        def append_segment(seg_first: int, seg_last: int, length: int) -> None:
            doc_idx = bisect_right(global_cu, seg_first) - 1
            doc_start = global_cu[doc_idx]
            seqlen_k = seg_last - doc_start + 1
            segs.append(
                (
                    restore[doc_start],
                    length,
                    seqlen_k,
                    seg_first - doc_start,
                )
            )

        for pos in q[1:]:
            if pos != prev + 1 or pos in doc_starts:
                append_segment(first, prev, seg_len)
                first = pos
                seg_len = 1
            else:
                seg_len += 1
            prev = pos

        append_segment(first, prev, seg_len)
        segs_all.append(segs)

    return segs_all


def _window_range(
    seg: tuple[int, int, int, int], window_size: int
) -> tuple[int, int]:
    win_start = max(seg[3] - (window_size - 1), 0)
    return win_start, seg[1] + seg[3] - win_start


def _block_range(
    seg: tuple[int, int, int, int], doc_len: int, *, ratio: int
) -> tuple[int, int, int]:
    p0, seg_len = seg[3], seg[1]
    q0 = p0 + seg_len
    straddle_idx = q0 // ratio
    straddle = (
        q0 % ratio != 0
        and straddle_idx * ratio >= p0
        and (straddle_idx + 1) * ratio <= doc_len
    )
    b_first = (p0 + ratio - 1) // ratio
    b_last = q0 // ratio - 1
    if b_first <= b_last:
        a = (b_first - 1) * ratio if b_first > 0 else 0
        b = (straddle_idx + 1) * ratio if straddle else q0
        strip = 1 if b_first > 0 else 0
    elif straddle and straddle_idx > 0:
        a = (straddle_idx - 1) * ratio
        b = (straddle_idx + 1) * ratio
        strip = 1
    elif straddle:
        a = 0
        b = (straddle_idx + 1) * ratio
        strip = 0
    else:
        a = q0
        b = q0
        strip = 0
    return a, b, strip


def _tensor(vals: list[int], device) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int64, device=device)


def _routing_geometry(
    foreign_all: list[list[int]], *, shard_len: int, cp_size: int
) -> list[tuple[list[int], list[int], list[int], list[int]]]:
    """Build alltoallv routing in linear passes over receiver positions."""
    send_indices: list[list[int]] = [[] for _ in range(cp_size)]
    send_splits: list[list[int]] = [[0] * cp_size for _ in range(cp_size)]
    recv_splits_all: list[list[int]] = []
    recv_offsets_all: list[list[int]] = []

    for dst, positions in enumerate(foreign_all):
        counts = [0] * cp_size
        for p in positions:
            src = p // shard_len
            send_indices[src].append(p % shard_len)
            send_splits[src][dst] += 1
            counts[src] += 1

        starts = [0] * cp_size
        total = 0
        for src, n in enumerate(counts):
            starts[src] = total
            total += n

        seen = [0] * cp_size
        recv_offsets: list[int] = []
        for p in positions:
            src = p // shard_len
            recv_offsets.append(starts[src] + seen[src])
            seen[src] += 1

        recv_splits_all.append(counts)
        recv_offsets_all.append(recv_offsets)

    return [
        (send_indices[r], send_splits[r], recv_splits_all[r], recv_offsets_all[r])
        for r in range(cp_size)
    ]


def _row_order(
    rows: list[int], *, rank: int, shard_len: int
) -> tuple[list[int], int]:
    order: list[int] = []
    recv_of: dict[int, int] = {}
    cursor = 0
    for pos in rows:
        if pos // shard_len == rank:
            order.append(pos - rank * shard_len)
        else:
            if pos not in recv_of:
                recv_of[pos] = cursor
                cursor += 1
            order.append(shard_len + recv_of[pos])
    return order, cursor


def _build_exchange_plan(row, device) -> ExchangePlan:
    send, splits, recv, off = row
    return ExchangePlan(
        send_indices=_tensor(send, device),
        send_splits=splits,
        recv_splits=recv,
        recv_offsets=_tensor(off, device),
    )


def _container_slots(segs_all, seg_blocks_all, *, ratio: int):
    """Build dense first-owner slots for compressed document blocks."""
    doc_nblocks: dict[int, int] = {}
    for segs, blocks in zip(segs_all, seg_blocks_all, strict=True):
        for seg, (_a, block_end, _strip) in zip(segs, blocks, strict=True):
            nblocks = max(block_end // ratio, seg[2] // ratio)
            if nblocks > doc_nblocks.get(seg[0], 0):
                doc_nblocks[seg[0]] = nblocks

    doc_base: dict[int, int] = {}
    total_blocks = 0
    for doc, nblocks in doc_nblocks.items():
        doc_base[doc] = total_blocks
        total_blocks += nblocks

    owner = [-1] * total_blocks
    local_offset = [0] * total_blocks
    max_kept = 0
    for rr, (segs, blocks) in enumerate(
        zip(segs_all, seg_blocks_all, strict=True)
    ):
        off = 0
        for seg, (a, block_end, strip) in zip(segs, blocks, strict=True):
            b0 = a // ratio + strip
            b1 = block_end // ratio
            if b1 <= b0:
                continue
            base = doc_base[seg[0]]
            for b in range(b0, b1):
                idx = base + b
                if owner[idx] < 0:
                    owner[idx] = rr
                    local_offset[idx] = off
                off += 1
        max_kept = max(max_kept, off)

    slots = [-1] * total_blocks
    for i, block_owner in enumerate(owner):
        if block_owner >= 0:
            slots[i] = block_owner * max_kept + local_offset[i]
    return slots, doc_base, max_kept


def _assemble_window_plan(
    segs: list[tuple[int, int, int, int]],
    docs: dict[int, list[int]],
    routing_row,
    *,
    rank: int,
    shard_len: int,
    window_size: int,
    device,
) -> WindowPlan:
    win_rows: list[int] = []
    ori_lens: list[int] = []
    for seg in segs:
        win_start, win_len = _window_range(seg, window_size)
        ori_lens.append(win_len)
        win_rows += docs[seg[0]][win_start : win_start + win_len]
    win_order, _ = _row_order(win_rows, rank=rank, shard_len=shard_len)
    cu_ori = torch.tensor(
        [0, *ori_lens], dtype=torch.int32, device=device
    ).cumsum(0, dtype=torch.int32)
    return WindowPlan(
        exchange=_build_exchange_plan(routing_row, device),
        gather_indices=_tensor(win_order, device),
        cu_seqlens_ori_kv=cu_ori,
    )


def _assemble_block_plan(
    segs: list[tuple[int, int, int, int]],
    segs_all: list[list[tuple[int, int, int, int]]],
    docs: dict[int, list[int]],
    seg_blocks_all: list[list[tuple[int, int, int]]],
    routing_row,
    *,
    ratio: int,
    rank: int,
    shard_len: int,
    device,
) -> CompressedBlockLayout:
    my_blocks = seg_blocks_all[rank]
    rows: list[int] = []
    for seg, (a, block_end, _strip) in zip(segs, my_blocks, strict=True):
        rows += docs[seg[0]][a:block_end]
    order, n_foreign = _row_order(rows, rank=rank, shard_len=shard_len)
    assert n_foreign == len(routing_row[3]), (n_foreign, len(routing_row[3]))

    pos_parts: list[torch.Tensor] = []
    first: list[int] = []
    compressed_rows: list[int] = []
    pool_start = 0
    for seg, (a, block_end, strip) in zip(segs, my_blocks, strict=True):
        if block_end <= a:
            continue
        cnt = (block_end - a) // ratio
        pos_parts.append(
            torch.arange(a, block_end, ratio, dtype=torch.int32, device=device)
        )
        first.append(pool_start)
        compressed_rows += list(range(pool_start + strip, pool_start + cnt))
        pool_start += cnt
    block_positions = (
        torch.cat(pos_parts)
        if pos_parts
        else torch.empty((0,), dtype=torch.int32, device=device)
    )

    cu_cmp = [seg[2] // ratio for seg in segs]
    rem = [seg[2] % ratio for seg in segs]
    cu_cmp_t = torch.tensor(
        [0, *cu_cmp], dtype=torch.int32, device=device
    ).cumsum(0, dtype=torch.int32)

    slots, doc_base, max_kept = _container_slots(
        segs_all, seg_blocks_all, ratio=ratio
    )
    cmp_k_global_gather_indices = [
        slots[doc_base[seg[0]] + b]
        for seg in segs
        for b in range(seg[2] // ratio)
    ]
    return CompressedBlockLayout(
        cu_seqlens_cmp_k=cu_cmp_t,
        block_remainder=torch.tensor(rem, dtype=torch.int32, device=device),
        gather_indices=_tensor(order, device),
        block_positions=block_positions,
        first_indices=_tensor(first, device),
        exchange=_build_exchange_plan(routing_row, device),
        compressed_rows=_tensor(compressed_rows, device),
        out_width=max_kept,
        cmp_k_global_gather_indices=_tensor(
            cmp_k_global_gather_indices, device
        ),
    )


def build_cp_plan(
    global_varlen,
    load_balancer,
    *,
    rank: int,
    cp_size: int,
    shard_len: int,
    window_size: int,
    ratios: list[int],
) -> tuple[CPVarlenMetadata, dict[int, CompressedBlockLayout], WindowPlan]:
    """Build the rank-local DSV4 plan from global packed boundaries.

    Only the current rank's common ``CPVarlenMetadata`` is constructed. The
    planner-specific all-rank segment geometry is derived directly from the
    global document boundaries and one load-balancer permutation on the host.
    """
    global_cu = global_varlen.cu_seq_q
    device = global_cu.device
    seq_len = int(global_cu[-1].item())

    if load_balancer is not None:
        rearrange_indices = load_balancer._generate_indices(restore=False)
        if rearrange_indices is None:
            raise ValueError("load_balancer._generate_indices() returned None")
        rearrange_indices = rearrange_indices.to(global_cu.dtype)
        if rearrange_indices.numel() != seq_len:
            raise ValueError(
                "DeepSeek-V4 CP planning currently requires local_batch_size=1; "
                f"got {rearrange_indices.numel()} load-balance indices for "
                f"seq_len={seq_len}."
            )
        rearrange = rearrange_indices.reshape(-1)
        restore = torch.empty_like(rearrange)
        restore[rearrange] = torch.arange(
            seq_len, dtype=rearrange.dtype, device=device
        )
        cp_load_balancer = cast(
            _LoadBalancer, _CachedLoadBalancer(rearrange_indices)
        )
    else:
        rearrange = torch.arange(
            seq_len, dtype=global_cu.dtype, device=device
        )
        restore = rearrange
        cp_load_balancer = None

    cp_metadata = CPVarlenMetadata.from_global(
        global_varlen,
        _RankMesh(cp_size, rank),  # pyrefly: ignore [bad-argument-type]
        1,
        seq_len,
        cp_load_balancer,
    )

    # One host materialization of the global planner inputs replaces CP copies
    # of CPVarlenMetadata plus per-rank ``segment_structure().cpu().tolist()``.
    rearrange_host = rearrange.tolist()
    restore_host = restore.tolist()
    global_cu_host = global_cu.tolist()
    segs_all = _derive_all_rank_segments(
        global_cu_host,
        rearrange_host,
        restore_host,
        cp_size=cp_size,
        shard_len=shard_len,
    )
    my_segs = segs_all[rank]

    docs: dict[int, list[int]] = {}
    for d0, d1 in zip(global_cu_host[:-1], global_cu_host[1:], strict=True):
        if d1 > d0:
            docs[restore_host[d0]] = restore_host[d0:d1]

    win_foreign: list[list[int]] = [[] for _ in range(cp_size)]
    for r in range(cp_size):
        for seg in segs_all[r]:
            win_start, win_len = _window_range(seg, window_size)
            doc = docs[seg[0]]
            win_foreign[r] += [
                p
                for p in doc[win_start : win_start + win_len]
                if p // shard_len != r
            ]
    routing = _routing_geometry(
        win_foreign, shard_len=shard_len, cp_size=cp_size
    )
    window = _assemble_window_plan(
        my_segs,
        docs,
        routing[rank],
        rank=rank,
        shard_len=shard_len,
        window_size=window_size,
        device=device,
    )

    plans: dict[int, CompressedBlockLayout] = {}
    for ratio in ratios:
        if ratio == 1:
            plans[1] = CompressedBlockLayout(
                cu_seqlens_cmp_k=None,
                block_remainder=None,
                gather_indices=None,
            )
            continue

        block_foreign: list[list[int]] = [[] for _ in range(cp_size)]
        seg_blocks_all: list[list[tuple[int, int, int]]] = [
            [] for _ in range(cp_size)
        ]
        for r in range(cp_size):
            for seg in segs_all[r]:
                a, b, strip = _block_range(
                    seg, len(docs[seg[0]]), ratio=ratio
                )
                block_end = (b // ratio) * ratio
                seg_blocks_all[r].append((a, block_end, strip))
                block_foreign[r] += [
                    p
                    for p in docs[seg[0]][a:block_end]
                    if p // shard_len != r
                ]
        routing = _routing_geometry(
            block_foreign, shard_len=shard_len, cp_size=cp_size
        )
        plans[ratio] = _assemble_block_plan(
            my_segs,
            segs_all,
            docs,
            seg_blocks_all,
            routing[rank],
            ratio=ratio,
            rank=rank,
            shard_len=shard_len,
            device=device,
        )

    return cp_metadata, plans, window


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class ExchangePlan:
    send_indices: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    recv_offsets: torch.Tensor
    send_splits_tensor: torch.Tensor = field(init=False)
    recv_splits_tensor: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.send_splits_tensor = torch.tensor(
            self.send_splits, dtype=torch.int64, device="cpu"
        )
        self.recv_splits_tensor = torch.tensor(
            self.recv_splits, dtype=torch.int64, device="cpu"
        )

    def splits_for_collective(self) -> tuple[list[int], list[int]]:
        if (
            torch.compiler.is_compiling()
            or torch.compiler._is_non_strict_tracing()
        ):
            return (
                self.send_splits_tensor.tolist(),
                self.recv_splits_tensor.tolist(),
            )
        return self.send_splits, self.recv_splits


@dataclass(kw_only=True, slots=True)
class WindowPlan:
    exchange: ExchangePlan
    gather_indices: torch.Tensor
    cu_seqlens_ori_kv: torch.Tensor


class CPTokenDispatcher(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        self.cp_mesh = None
        self.rank: int | None = None
        self.cp_size: int | None = None

    def wire_meshes(self, *, cp_mesh=None):
        self.cp_mesh = cp_mesh
        self.rank = (
            cp_mesh.get_local_rank() if cp_mesh is not None else None
        )
        self.cp_size = cp_mesh.size() if cp_mesh is not None else None

    def _all_to_all(self, x, in_splits, out_splits):
        mesh = self.cp_mesh
        assert mesh is not None, (
            "CPTokenDispatcher must be wired to a CP mesh before an exchange"
        )
        if (
            torch.compiler.is_compiling()
            or torch.compiler._is_non_strict_tracing()
        ) or get_spmd_backend() != "spmd_types":
            return all_to_all_single(
                x, out_splits, in_splits, group=mesh.get_group()
            )
        return spmd.all_to_all(
            x,
            mesh.get_group(),
            src=spmd.V,
            dst=spmd.V,
            input_split_sizes=in_splits,
            output_split_sizes=out_splits,
        )

    def gather(
        self, x: torch.Tensor, plan: CompressedBlockLayout | WindowPlan
    ) -> torch.Tensor:
        if plan is None or plan.exchange is None:
            if plan is not None and plan.gather_indices is not None:
                return x.flatten(0, 1)[plan.gather_indices].view(
                    1, -1, *x.shape[2:]
                )
            return x
        ex = plan.exchange
        send_splits, recv_splits = ex.splits_for_collective()
        rows = self._all_to_all(
            x.flatten(0, 1)[ex.send_indices],
            send_splits,
            recv_splits,
        )
        aug = torch.cat(
            [x.flatten(0, 1), rows[ex.recv_offsets]], dim=0
        )[plan.gather_indices]
        return aug.view(1, -1, *x.shape[2:])

    def select(
        self, x: torch.Tensor, plan: CompressedBlockLayout
    ) -> torch.Tensor:
        x2 = x.flatten(0, 1) if x.ndim > 2 else x
        out = (
            x2
            if plan.compressed_rows is None
            else x2[plan.compressed_rows]
        )
        out_width = plan.out_width
        assert out_width is not None, "select requires the container width"
        pad = x.new_zeros((out_width - out.shape[0], x.shape[-1]))
        return torch.cat([out, pad], dim=0).unsqueeze(0)
