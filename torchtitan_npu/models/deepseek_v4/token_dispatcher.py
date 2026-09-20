# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 context-parallel planning and token dispatch.

The planner derives current-rank CP metadata plus all-rank routing/container
geometry from global varlen boundaries and one load-balancer permutation.
The O(S) planner math stays on the planner device; only the uneven-all-to-all
split sizes cross to host because the current collective API requires Python
split lists.

The dispatcher consumes the resulting window/block plans for local/remote
gathers and compressed-container packing.
"""

from __future__ import annotations

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

from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
    CPVarlenMetadata,
    _argsort_indices,
)

from .metadata import CompressedBlockLayout

__all__ = [
    "CPTokenDispatcher",
    "ExchangePlan",
    "WindowPlan",
    "build_cp_plan",
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
            return _argsort_indices(self._rearrange_indices)
        return self._rearrange_indices


# ---------------------------------------------------------------------------
# The plan builder (device tensor math; no plan-time communication)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SegmentGeometry:
    ranks: torch.Tensor
    doc_starts: torch.Tensor
    doc_lens: torch.Tensor
    seg_lens: torch.Tensor
    seqlens_k: torch.Tensor
    p0: torch.Tensor


def _segment_geometry(
    global_cu: torch.Tensor,
    rearrange: torch.Tensor,
    *,
    cp_size: int,
    shard_len: int,
) -> _SegmentGeometry:
    """All-rank segment geometry as device tensors."""
    q = rearrange.reshape(cp_size, shard_len)
    doc_idx = torch.searchsorted(global_cu[1:], q, right=True)

    breaks = torch.ones_like(q, dtype=torch.bool)
    if shard_len > 1:
        breaks[:, 1:] = (q[:, 1:] != q[:, :-1] + 1) | (
            doc_idx[:, 1:] != doc_idx[:, :-1]
        )

    seg_local = breaks.cumsum(1) - 1
    counts = breaks.sum(1)
    seg_ids = seg_local + (counts.cumsum(0) - counts)[:, None]

    seg_lens = torch.bincount(seg_ids.reshape(-1)).to(global_cu.dtype)
    first = q[breaks]
    seg_doc = doc_idx[breaks]
    doc_starts = global_cu[seg_doc]
    p0 = first - doc_starts
    return _SegmentGeometry(
        ranks=torch.repeat_interleave(
            torch.arange(cp_size, device=q.device), counts.to(torch.long)
        ),
        doc_starts=doc_starts,
        doc_lens=global_cu[seg_doc + 1] - doc_starts,
        seg_lens=seg_lens,
        seqlens_k=p0 + seg_lens,
        p0=p0,
    )


def _expand_ranges(
    starts: torch.Tensor,
    lengths: torch.Tensor,
    restore: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand non-negative original-position ranges to permuted rows."""
    repeats = lengths.clamp_min(0).to(torch.long)
    seg_ids = torch.repeat_interleave(
        torch.arange(repeats.numel(), device=starts.device), repeats
    )
    base = repeats.cumsum(0) - repeats
    rel = torch.arange(repeats.sum(), device=starts.device) - torch.repeat_interleave(
        base, repeats
    )
    original = starts[seg_ids].to(torch.long) + rel
    return restore[original], seg_ids


def _routing_tensors(
    rows: torch.Tensor,
    dest_ranks: torch.Tensor,
    *,
    rank: int,
    shard_len: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Current-rank all-to-all routing plus packed gather order on device."""
    src_ranks = torch.div(rows, shard_len, rounding_mode="floor")
    foreign = src_ranks != dest_ranks

    send_mask = foreign & (src_ranks == rank)
    send_indices = (rows[send_mask] % shard_len).to(torch.long)
    send_splits = torch.bincount(
        dest_ranks[send_mask].to(torch.long), minlength=cp_size
    )

    recv_mask = foreign & (dest_ranks == rank)
    recv_src = src_ranks[recv_mask].to(torch.long)
    recv_splits = torch.bincount(recv_src, minlength=cp_size)
    # all_to_all output is source-major while planner rows are destination-
    # major.  The inverse stable source sort maps planner order back to the
    # receive buffer without an O(n_recv * cp_size) one-hot/cumsum matrix.
    recv_order = torch.argsort(recv_src.to(torch.float32), stable=True)
    recv_offsets = torch.empty_like(recv_order)
    recv_offsets[recv_order] = torch.arange(
        recv_order.numel(), device=rows.device
    )

    packed = rows[dest_ranks == rank].to(torch.long)
    packed_src = torch.div(packed, shard_len, rounding_mode="floor")
    local = packed_src == rank
    nrows = packed.numel()
    row_ids = torch.arange(nrows, device=rows.device)
    sentinel = torch.full((), nrows, dtype=torch.long, device=rows.device)

    first = torch.full(
        (cp_size * shard_len,), nrows, dtype=torch.long, device=rows.device
    )
    first.scatter_reduce_(
        0,
        packed,
        torch.where(local, sentinel, row_ids),
        reduce="amin",
        include_self=True,
    )
    is_first = (~local) & (first[packed] == row_ids)
    slots = is_first.to(torch.long).cumsum(0) - 1

    slot_by_row = torch.full_like(first, nrows)
    slot_by_row.scatter_reduce_(
        0,
        packed,
        torch.where(is_first, slots, sentinel),
        reduce="amin",
        include_self=True,
    )
    gather_indices = torch.where(
        local,
        packed - rank * shard_len,
        shard_len + slot_by_row[packed],
    )
    return send_indices, send_splits, recv_splits, recv_offsets, gather_indices


def _container_layout(
    ranks: torch.Tensor,
    doc_starts: torch.Tensor,
    seqlens_k: torch.Tensor,
    A: torch.Tensor,
    block_end: torch.Tensor,
    strip: torch.Tensor,
    *,
    ratio: int,
    rank: int,
    cp_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-owner compressed slots on device."""
    b0 = torch.div(A, ratio, rounding_mode="floor") + strip
    b1 = torch.div(block_end, ratio, rounding_mode="floor")
    kept = (b1 - b0).clamp_min(0).to(torch.long)

    seg_ids = torch.repeat_interleave(
        torch.arange(kept.numel(), device=A.device), kept
    )
    base = kept.cumsum(0) - kept
    rel = torch.arange(kept.sum(), device=A.device) - torch.repeat_interleave(
        base, kept
    )
    block_keys = (
        doc_starts[seg_ids].to(torch.long)
        + (b0[seg_ids].to(torch.long) + rel) * ratio
    )
    kept_ranks = ranks[seg_ids].to(torch.long)

    per_rank = torch.bincount(kept_ranks, minlength=cp_size)
    rank_base = per_rank.cumsum(0) - per_rank
    local_offset = torch.arange(block_keys.numel(), device=A.device) - rank_base[
        kept_ranks
    ]
    max_kept = per_rank.max()

    sentinel = cp_size * max_kept + seq_len
    slots = torch.zeros(seq_len, dtype=torch.long, device=A.device) + sentinel
    slots.scatter_reduce_(
        0,
        block_keys,
        kept_ranks * max_kept + local_offset,
        reduce="amin",
        include_self=True,
    )

    mine = ranks == rank
    prefix = torch.div(
        seqlens_k[mine], ratio, rounding_mode="floor"
    ).to(torch.long)
    seg_ids = torch.repeat_interleave(
        torch.arange(prefix.numel(), device=A.device), prefix
    )
    base = prefix.cumsum(0) - prefix
    rel = torch.arange(prefix.sum(), device=A.device) - torch.repeat_interleave(
        base, prefix
    )
    target = doc_starts[mine][seg_ids].to(torch.long) + rel * ratio
    return slots[target], per_rank


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
    """Build the current-rank DSV4 CP plan with tensorized planner math."""
    global_cu = global_varlen.cu_seq_q
    device = global_cu.device
    seq_len = shard_len * cp_size

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
    geometry = _segment_geometry(
        global_cu,
        rearrange,
        cp_size=cp_size,
        shard_len=shard_len,
    )
    mine = geometry.ranks == rank

    def route(rows: torch.Tensor, dest_ranks: torch.Tensor):
        return _routing_tensors(
            rows,
            dest_ranks,
            rank=rank,
            shard_len=shard_len,
            cp_size=cp_size,
        )

    def exchange_plan(send, send_splits, recv_splits, recv_offsets):
        # Current alltoallv APIs require host split lists.  Keep this as one
        # O(CP) control-plane D2H instead of separate transfers per list.
        splits = torch.stack([send_splits, recv_splits]).cpu().tolist()
        return ExchangePlan(
            send_indices=send,
            send_splits=splits[0],
            recv_splits=splits[1],
            recv_offsets=recv_offsets,
        )

    # Sliding-window rows.
    win_start = torch.clamp(geometry.p0 - (window_size - 1), min=0)
    win_lens = geometry.seg_lens + geometry.p0 - win_start
    win_rows, win_seg = _expand_ranges(
        geometry.doc_starts + win_start, win_lens, restore
    )
    win_send, win_ss, win_rs, win_off, win_order = route(
        win_rows, geometry.ranks[win_seg]
    )
    window = WindowPlan(
        exchange=exchange_plan(win_send, win_ss, win_rs, win_off),
        gather_indices=win_order,
        cu_seqlens_ori_kv=torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                win_lens[mine].to(torch.int32),
            ]
        ).cumsum(0, dtype=torch.int32),
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

        p0 = geometry.p0
        q0 = p0 + geometry.seg_lens
        straddle_idx = torch.div(q0, ratio, rounding_mode="floor")
        straddle = (
            (q0.remainder(ratio) != 0)
            & (straddle_idx * ratio >= p0)
            & ((straddle_idx + 1) * ratio <= geometry.doc_lens)
        )
        b_first = torch.div(p0 + ratio - 1, ratio, rounding_mode="floor")
        b_last = torch.div(q0, ratio, rounding_mode="floor") - 1
        complete = b_first <= b_last
        borrow = (~complete) & straddle & (straddle_idx > 0)
        first_block = (~complete) & straddle & (straddle_idx == 0)

        A = torch.where(
            complete,
            torch.where(
                b_first > 0, (b_first - 1) * ratio, torch.zeros_like(q0)
            ),
            torch.where(
                borrow,
                (straddle_idx - 1) * ratio,
                torch.where(first_block, torch.zeros_like(q0), q0),
            ),
        )
        B = torch.where(
            complete,
            torch.where(straddle, (straddle_idx + 1) * ratio, q0),
            torch.where(
                borrow | first_block, (straddle_idx + 1) * ratio, q0
            ),
        )
        strip = torch.where(
            complete,
            (b_first > 0).to(q0.dtype),
            borrow.to(q0.dtype),
        )
        block_end = torch.div(B, ratio, rounding_mode="floor") * ratio

        block_rows, block_seg = _expand_ranges(
            geometry.doc_starts + A,
            (block_end - A).clamp_min(0),
            restore,
        )
        block_send, block_ss, block_rs, block_off, block_order = route(
            block_rows, geometry.ranks[block_seg]
        )

        A_local = A[mine]
        end_local = block_end[mine]
        strip_local = strip[mine].to(torch.long)
        block_count = torch.div(
            (end_local - A_local).clamp_min(0),
            ratio,
            rounding_mode="floor",
        ).to(torch.long)
        pool_start = block_count.cumsum(0) - block_count

        seg_ids = torch.repeat_interleave(
            torch.arange(block_count.numel(), device=device), block_count
        )
        rel = torch.arange(block_count.sum(), device=device) - torch.repeat_interleave(
            pool_start, block_count
        )
        block_positions = (
            A_local[seg_ids].to(torch.long) + rel * ratio
        ).to(torch.int32)

        kept = (block_count - strip_local).clamp_min(0)
        kept_seg = torch.repeat_interleave(
            torch.arange(kept.numel(), device=device), kept
        )
        kept_base = kept.cumsum(0) - kept
        kept_rel = torch.arange(kept.sum(), device=device) - torch.repeat_interleave(
            kept_base, kept
        )
        compressed_rows = (
            pool_start[kept_seg] + strip_local[kept_seg] + kept_rel
        ).to(torch.long)

        cmp_gather, kept_per_rank = _container_layout(
            geometry.ranks,
            geometry.doc_starts,
            geometry.seqlens_k,
            A,
            block_end,
            strip,
            ratio=ratio,
            rank=rank,
            cp_size=cp_size,
            seq_len=seq_len,
        )
        control = torch.stack([block_ss, block_rs, kept_per_rank]).cpu().tolist()
        block_exchange = ExchangePlan(
            send_indices=block_send,
            send_splits=control[0],
            recv_splits=control[1],
            recv_offsets=block_off,
        )
        out_width = max(control[2])

        cmp_lens = torch.div(
            geometry.seqlens_k[mine], ratio, rounding_mode="floor"
        ).to(torch.int32)
        plans[ratio] = CompressedBlockLayout(
            cu_seqlens_cmp_k=torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=device),
                    cmp_lens,
                ]
            ).cumsum(0, dtype=torch.int32),
            block_remainder=geometry.seqlens_k[mine].remainder(ratio).to(
                torch.int32
            ),
            gather_indices=block_order,
            block_positions=block_positions,
            first_indices=pool_start[block_count > 0].to(torch.long),
            exchange=block_exchange,
            compressed_rows=compressed_rows,
            out_width=out_width,
            cmp_k_global_gather_indices=cmp_gather,
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

    The splits are plain host lists because the current uneven-all-to-all
    APIs require host integers. They are materialized once per batch at the
    planner boundary; per-layer exchange never performs a D2H conversion.
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
