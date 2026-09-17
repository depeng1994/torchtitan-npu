# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallel for DeepSeek-V4: the plan and the token dispatcher.

Two halves live here (the single home of all DSV4 CP logic):

1. **The plan builder** — a pure derivation from the global context: the
   pre-shard document structure (the model's ``get_attention_masks``
   result) plus the load-balancer permutation.  The planner derives all-rank
   segment geometry directly from that global context and materializes only the
   current rank's ``CPVarlenMetadata`` through the shard path's own builder, so
   there is **no plan-time communication** at all.  The whole derivation
   is expressed in the documents' **permuted slices** (``docs``, keyed by
   the segment's doc identity): every row a plan consumes — window
   ``[win_start, q0)`` or plan blocks ``[A, B)`` — is a plain slice of
   one document's slice.

   The plan has two independent parts:

   - the **window plan** (``WindowPlan``, ratio-independent): the per-
     segment sliding-window rows — the exchange routing, the packed ori
     stream's ``gather_indices``, and the packed cumsum
     ``cu_seqlens_ori_kv``.  The Attention gathers the **post-RoPE**
     ``swa_k`` rows through it;
   - the **block plans** (``CompressedBlockLayout`` at ``plans[ratio]``,
     per ratio > 1): the per-segment ratio-aligned plan-block region
     ``[A, B)`` (``_block_range`` — the borrow-source blocks included) —
     the exchange routing + the pooled order's ``gather_indices`` (the
     assembly into ``cat([x_local, recv])``), the directly derived
     compressor contract (``block_positions`` = the doc-relative block
     starts, ``first_indices`` = the per-segment pooled starts), and the
     container-packing / compressed-gather fields.  Each Compressor
     gathers its **projected kv/score** rows through it.

2. **The dispatcher** — the ``BaseEPTokenDispatcher``-shaped mechanism,
   a plain ``Configurable``: one instance on the ``Attention`` (the
   window gather of ``swa_k``) and one per ``Compressor`` (the block
   gather of kv/score), all wired to the CP mesh by their owners'
   ``parallelize``.  The two plan-driven ops are ``gather`` (a local
   gather without an exchange — the plan's ``gather_indices`` over the
   local stream; a remote gather + permute with one — the exchange plus
   the same ``gather_indices`` over ``cat([x_local, recv])``) and
   ``select`` (the container packing).  The exchanges are uneven
   all-to-alls over the CP process group (``spmd_types``); ``_all_to_all``
   is the override seam — the CPU tests subclass and replace it with
   their mock / gloo-capable exchange, so no portable fallback lives in
   production code.

The model's ``build_attention_masks`` calls ``build_cp_plan`` and returns
``CompressedVarlenMetadata`` (the varlen + the per-ratio block plans + the
``window`` plan); the per-layer forward then runs the window gather, the
per-compressor block gathers, and the declarative all-gather of the padded
containers (the core's ``ShardingConfig`` ``cp: S(1) -> R``), assembled per
segment with each ratio plan's ``cmp_k_global_gather_indices``.
"""

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
    """``CPVarlenMetadata.from_global``'s ``DeviceMesh`` contract for one rank."""

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
# The plan builder (pure host math, per batch, no communication)
# ---------------------------------------------------------------------------


def segment_structure(cp_meta) -> list[tuple[int, int, int, int]]:
    """Per non-empty segment: ``(doc_start, seg_len, seqlen_k, p0)``.

    ``doc_start = kgather[cu_seq_k[s]]`` is the document identity
    (identical across ranks, permuted coordinates — the key of the
    document's permuted slice in ``docs``); ``p0 = seqlen_k - seg_len`` is
    the fragment's doc-relative start.
    """
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
    """Derive planner-only all-rank segments from global packed boundaries.

    This reproduces ``segment_structure`` without constructing CP-1 temporary
    ``CPVarlenMetadata`` objects and copying their rank-local tensors to host.
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


def _window_range(seg: tuple[int, int, int, int], window_size: int) -> tuple[int, int]:
    """``(win_start, win_len)`` of the segment's window rows — ratio-
    independent: the doc-relative range ``[max(p0 - (window - 1), 0),
    p0 + seg_len)``."""
    win_start = max(seg[3] - (window_size - 1), 0)
    return win_start, seg[1] + seg[3] - win_start


def _block_range(seg: tuple[int, int, int, int], doc_len: int, *, ratio: int) -> tuple[int, int, int]:
    """``(A, B, strip)`` of the segment's ratio-aligned plan-block region.

    ``[A, B)`` covers the segment's complete blocks plus their borrow
    source: ``A`` is the borrow-source block start (ratio-aligned), ``B``
    the straddle end (``q0`` without a straddle — the fragment-end block
    that ends mid-block inside the document and starts in the fragment),
    and ``strip`` counts the leading borrow-source blocks the container
    drops.  Requires ``ratio > 1``.
    """
    p0, seg_len = seg[3], seg[1]
    q0 = p0 + seg_len
    straddle_idx = q0 // ratio
    straddle = q0 % ratio != 0 and straddle_idx * ratio >= p0 and (straddle_idx + 1) * ratio <= doc_len
    b_first = (p0 + ratio - 1) // ratio
    b_last = q0 // ratio - 1
    if b_first <= b_last:
        # Case A/B: the prepend block (b_first - 1) completes the overlap
        # chain of the local blocks.
        A = (b_first - 1) * ratio if b_first > 0 else 0
        B = (straddle_idx + 1) * ratio if straddle else q0
        strip = 1 if b_first > 0 else 0
    elif straddle and straddle_idx > 0:
        # Case C: the pred block is the straddle block's overlap
        # predecessor (a stripped borrow source).
        A = (straddle_idx - 1) * ratio
        B = (straddle_idx + 1) * ratio
        strip = 1
    elif straddle:
        # The document's first block straddles the fragment (p0 == 0): it
        # is the fragment's only plan block, kept with no borrow.
        A = 0
        B = (straddle_idx + 1) * ratio
        strip = 0
    else:
        # No plan blocks: the sub-range is empty.
        A = q0
        B = q0
        strip = 0
    return A, B, strip


def _tensor(vals: list[int], device) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int64, device=device)


def _routing_geometry(
    foreign_all: list[list[int]], *, shard_len: int, cp_size: int
) -> list[tuple[list[int], list[int], list[int], list[int]]]:
    """The alltoallv routing of one exchange.

    ``foreign_all[r]`` = rank r's foreign positions (permuted-stream
    coordinates, in its per-segment receive order).  Build the same routing
    in two linear passes over each receiver instead of rescanning every
    receiver once per source rank.
    """
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


def _row_order(rows: list[int], *, rank: int, shard_len: int) -> tuple[list[int], int]:
    """The row order of one packed stream and its unique-receive count.

    ``rows`` are the stream's rows in packed order (permuted
    coordinates); local rows map to their local offset, foreign rows to
    ``shard_len + receive slot`` — the first-appearance order matching the
    routing's ``recv_offsets``.
    """
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
    """One rank's routing row as the tensor/list mix the collective APIs
    need: the splits stay host lists (built per batch) so the per-layer
    exchange never syncs."""
    send, splits, recv, off = row
    return ExchangePlan(
        send_indices=_tensor(send, device),
        send_splits=splits,
        recv_splits=recv,
        recv_offsets=_tensor(off, device),
    )


def _container_slots(segs_all, seg_blocks_all, *, ratio: int):
    """Build dense first-owner slots and the uniform container width.

    Avoid a tuple-key Python dict lookup for every compressed block.  Document
    identities stay as dict keys only once per document; block ownership and
    local offsets use flat lists.
    """
    doc_nblocks: dict[int, int] = {}
    for segs, blocks in zip(segs_all, seg_blocks_all, strict=True):
        for seg, (_A, block_end, _strip) in zip(segs, blocks, strict=True):
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
    for rr, (segs, blocks) in enumerate(zip(segs_all, seg_blocks_all, strict=True)):
        off = 0
        for seg, (A, block_end, strip) in zip(segs, blocks, strict=True):
            b0 = A // ratio + strip
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
    """The ratio-independent window plan: the exchange + the packed ori
    stream's ``gather_indices`` (``cu_seqlens_ori_kv`` bounds the
    per-segment window rows)."""
    win_rows: list[int] = []
    ori_lens: list[int] = []
    for seg in segs:
        win_start, win_len = _window_range(seg, window_size)
        ori_lens.append(win_len)
        win_rows += docs[seg[0]][win_start : win_start + win_len]
    # The receive slots number the stream-wide foreign order (matching the
    # routing's recv_offsets), so the gather indices run once over all rows.
    win_order, _ = _row_order(win_rows, rank=rank, shard_len=shard_len)
    cu_ori = torch.tensor([0, *ori_lens], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
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
    """One ratio's block plan from the pure global-context derivation.

    ``seg_blocks_all[r]`` holds rank r's per-segment ``(A, block_end,
    strip)`` scalars.  Part 1 (the unified compressor/kernel contract) is
    derived directly over the plan blocks; the ``gather_indices`` order
    the pooled stream the exchange produces (``cat([x_local, recv])``)."""
    my_blocks = seg_blocks_all[rank]
    rows: list[int] = []
    for seg, (A, block_end, _strip) in zip(segs, my_blocks, strict=True):
        rows += docs[seg[0]][A:block_end]
    order, n_foreign = _row_order(rows, rank=rank, shard_len=shard_len)
    # The plan blocks of one rank never overlap, so every foreign row is
    # received exactly once (the recv_offsets length is the receive count).
    assert n_foreign == len(routing_row[3]), (n_foreign, len(routing_row[3]))

    # The compressor contract, derived directly over the plan blocks.
    pos_parts: list[torch.Tensor] = []
    first: list[int] = []
    compressed_rows: list[int] = []
    pool_start = 0
    for seg, (A, block_end, strip) in zip(segs, my_blocks, strict=True):
        # Segments without complete blocks have ``block_end <= A`` (the
        # no-plan-block case: ``A == B == q0`` with ``block_end < q0``).
        if block_end <= A:
            continue
        cnt = (block_end - A) // ratio
        pos_parts.append(torch.arange(A, block_end, ratio, dtype=torch.int32, device=device))
        first.append(pool_start)
        compressed_rows += list(range(pool_start + strip, pool_start + cnt))
        pool_start += cnt
    block_positions = torch.cat(pos_parts) if pos_parts else torch.empty((0,), dtype=torch.int32, device=device)
    # The packed kernel tensors (the real causal-prefix counts).
    cu_cmp = [seg[2] // ratio for seg in segs]
    rem = [seg[2] % ratio for seg in segs]
    cu_cmp_t = torch.tensor([0, *cu_cmp], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    # ---- compressed-level gather: ownership + assembly ----
    slots, doc_base, max_kept = _container_slots(segs_all, seg_blocks_all, ratio=ratio)
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
        cmp_k_global_gather_indices=_tensor(cmp_k_global_gather_indices, device),
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
    """Derive one rank's DSV4 plan from global packed boundaries.

    The common ``CPVarlenMetadata`` is built only for the current rank.  DSV4's
    planner-only all-rank segment geometry comes directly from global document
    boundaries plus one load-balancer layout, so no CP-sized set of temporary
    common metadata is materialized.
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
        # DSV4 needs the inverse once for planner host geometry.  The common
        # upstream-like builder keeps its own inverse contract for current-rank
        # metadata; do not extend that public API with a DSV4-only fast path.
        restore = torch.empty_like(rearrange)
        restore[rearrange] = torch.arange(seq_len, dtype=rearrange.dtype, device=device)
        cp_load_balancer = cast(_LoadBalancer, _CachedLoadBalancer(rearrange_indices))
    else:
        rearrange = torch.arange(seq_len, dtype=global_cu.dtype, device=device)
        restore = rearrange
        cp_load_balancer = None

    cp_metadata = CPVarlenMetadata.from_global(
        global_varlen,
        _RankMesh(cp_size, rank),  # pyrefly: ignore [bad-argument-type]
        1,
        seq_len,
        cp_load_balancer,
    )

    # One host materialization of global planner inputs replaces CP copies of
    # CPVarlenMetadata plus per-rank ``segment_structure().cpu().tolist()``.
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

    # The full permuted document slices, keyed by the segment's doc identity.
    docs: dict[int, list[int]] = {}
    for d0, d1 in zip(global_cu_host[:-1], global_cu_host[1:], strict=True):
        if d1 > d0:
            docs[restore_host[d0]] = restore_host[d0:d1]

    # ---- the window plan (ratio-independent) ----
    win_foreign: list[list[int]] = [[] for _ in range(cp_size)]
    for r in range(cp_size):
        for seg in segs_all[r]:
            win_start, win_len = _window_range(seg, window_size)
            doc = docs[seg[0]]
            win_foreign[r] += [p for p in doc[win_start : win_start + win_len] if p // shard_len != r]
    routing = _routing_geometry(win_foreign, shard_len=shard_len, cp_size=cp_size)
    window = _assemble_window_plan(
        my_segs,
        docs,
        routing[rank],
        rank=rank,
        shard_len=shard_len,
        window_size=window_size,
        device=device,
    )

    # ---- the per-ratio block plans ----
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
        seg_blocks_all: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
        for r in range(cp_size):
            for seg in segs_all[r]:
                A, B, strip = _block_range(seg, len(docs[seg[0]]), ratio=ratio)
                block_end = (B // ratio) * ratio
                seg_blocks_all[r].append((A, block_end, strip))
                block_foreign[r] += [p for p in docs[seg[0]][A:block_end] if p // shard_len != r]
        routing = _routing_geometry(block_foreign, shard_len=shard_len, cp_size=cp_size)
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
# The dispatcher (the BaseEPTokenDispatcher mirror)
# ---------------------------------------------------------------------------
#
# A plain ``Configurable`` (no learnable state): one instance on the
# ``Attention`` (the window gather of the post-RoPE ``swa_k`` rows) and one
# per ``Compressor`` (the block gather of the projected kv/score rows),
# wired to the CP mesh by their owners' ``parallelize``.  The two
# plan-driven ops: ``gather`` (a local gather without an exchange — the
# plan's ``gather_indices`` over the local stream; a remote gather +
# permute with one — the exchange plus the same ``gather_indices`` over
# ``cat([x_local, recv])``) and ``select`` (the pooled-key container
# packing).  The compressed-level gather of the padded containers is
# declarative (the core's ``ShardingConfig`` ``cp: S(1) -> R``), not a
# dispatcher op.


@dataclass(kw_only=True, slots=True)
class ExchangePlan:
    """The alltoallv routing of one exchange (the EP TokenDispatcher form).

    The send payload is ``x_local[send_indices]`` — the rank's rows grouped
    by receiver (receiver order), so the exchange is a native all_to_all
    with ``send_splits`` / ``recv_splits``.  ``recv_offsets[k]`` is the flat
    offset of the k-th foreign receive position in the exchange output
    (cat over senders of [my rows from that sender]).

    The splits are plain host lists, built once per batch: the eager
    collective APIs need Python ints and the per-layer exchange must never
    call ``.tolist()`` (a D2H sync per layer per step). During tracing, CPU
    tensors expose the same sizes as data-dependent ``SymInt`` values so a new
    packed batch does not recompile every block. Keeping those scalar inputs on
    CPU avoids a per-layer NPU-to-CPU sync.
    """

    send_indices: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    recv_offsets: torch.Tensor
    send_splits_tensor: torch.Tensor = field(init=False)
    recv_splits_tensor: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.send_splits_tensor = torch.tensor(self.send_splits, dtype=torch.int64, device="cpu")
        self.recv_splits_tensor = torch.tensor(self.recv_splits, dtype=torch.int64, device="cpu")

    def splits_for_collective(self) -> tuple[list[int], list[int]]:
        """Return dynamic sizes while tracing and host sizes in eager mode."""
        if torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing():
            return (
                self.send_splits_tensor.tolist(),
                self.recv_splits_tensor.tolist(),
            )
        return self.send_splits, self.recv_splits


@dataclass(kw_only=True, slots=True)
class WindowPlan:
    """The sliding-window plan (ratio-independent, CP only).

    Describes the Attention's ``swa_k`` gather: the exchange routing of
    the per-segment window rows ``[win_start, q0)``, the packed ori
    stream's ``gather_indices`` (indices into
    ``cat([x_local, recv])``), and the packed-ori cumsum
    (``cu_seqlens_ori_kv``) the kernels consume.  Both the window rows and
    this plan are ratio-independent — one object per rank.
    """

    exchange: ExchangePlan
    gather_indices: torch.Tensor
    cu_seqlens_ori_kv: torch.Tensor


class CPTokenDispatcher(Configurable):
    """The DSV4 CP token dispatcher (the ``BaseEPTokenDispatcher`` mirror):
    a plain ``Configurable`` — not an ``nn.Module`` — with no learnable
    parameters or buffers.  One instance per consumer: the ``Attention``
    (the window gather of ``swa_k``) and each ``Compressor`` (the block
    gather of kv/score).  The CP mesh is installed once by ``wire_meshes``
    (from the owner's ``parallelize``).  ``gather`` is self-guarding:
    without a plan it is the identity, without an exchange it is a plain
    local gather — the forward path never special-cases context parallel.

    The CPU tests subclass and override ``_all_to_all`` with their mock /
    gloo-capable exchange — no portable fallback lives in production code.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        self.cp_mesh = None
        self.rank: int | None = None
        self.cp_size: int | None = None

    def wire_meshes(self, *, cp_mesh=None):
        """Install the CP mesh (mirrors ``BaseEPTokenDispatcher.wire_meshes``;
        called once from the owner's ``parallelize``)."""
        self.cp_mesh = cp_mesh
        self.rank = cp_mesh.get_local_rank() if cp_mesh is not None else None
        self.cp_size = cp_mesh.size() if cp_mesh is not None else None

    def _all_to_all(self, x, in_splits, out_splits):
        """The uneven all-to-all (mirrors ``AllToAllTokenDispatcher``'s
        transport selection: ``spmd_types`` eager, the native
        ``all_to_all_single`` under compile/tracing or non-spmd backends)."""
        mesh = self.cp_mesh
        assert mesh is not None, "CPTokenDispatcher must be wired to a CP mesh before an exchange"
        if (
            torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()
        ) or get_spmd_backend() != "spmd_types":
            return all_to_all_single(x, out_splits, in_splits, group=mesh.get_group())
        return spmd.all_to_all(
            x,
            mesh.get_group(),
            src=spmd.V,
            dst=spmd.V,
            input_split_sizes=in_splits,
            output_split_sizes=out_splits,
        )

    def gather(self, x: torch.Tensor, plan: CompressedBlockLayout | WindowPlan) -> torch.Tensor:
        """The plan-driven row gather.

        ``plan`` is the ``WindowPlan`` (the swa path — the post-RoPE
        ``swa_k`` rows) or the ``CompressedBlockLayout`` (the compressor
        path — the projected kv/score rows).  Without an exchange the
        plan's ``gather_indices`` are a plain local gather over the
        stream (the identity without a plan — the non-CP window path);
        with one, the exchange gathers the foreign rows and the same
        ``gather_indices`` order ``cat([x_local, recv])`` into the pooled
        stream — a remote gather + permute.  Always returns
        ``[1, N, D]``.
        """
        if plan is None or plan.exchange is None:
            if plan is not None and plan.gather_indices is not None:
                return x.flatten(0, 1)[plan.gather_indices].view(1, -1, *x.shape[2:])
            return x
        ex = plan.exchange
        send_splits, recv_splits = ex.splits_for_collective()
        rows = self._all_to_all(
            x.flatten(0, 1)[ex.send_indices],
            send_splits,
            recv_splits,
        )
        aug = torch.cat([x.flatten(0, 1), rows[ex.recv_offsets]], dim=0)[plan.gather_indices]
        return aug.view(1, -1, *x.shape[2:])

    def select(self, x: torch.Tensor, plan: CompressedBlockLayout) -> torch.Tensor:
        """The plan-indexed container packing: selects the pooled stream's
        ``compressed_rows`` (the kept blocks) and zero-pads them into the
        uniform container grid ``[1, out_width, D]``.

        Functional padding (``torch.cat``): a pre-allocated zero container
        filled behind a data-dependent ``if out.shape[0]:`` is rejected by
        the tracer and dropped by aot_autograd recompute.
        """
        # Flatten only the batch+sequence of the 3-D tensors; the 2-D
        # pooled streams select directly.
        x2 = x.flatten(0, 1) if x.ndim > 2 else x
        out = x2 if plan.compressed_rows is None else x2[plan.compressed_rows]
        out_width = plan.out_width
        assert out_width is not None, "select requires the container width"
        pad = x.new_zeros((out_width - out.shape[0], x.shape[-1]))
        return torch.cat([out, pad], dim=0).unsqueeze(0)
