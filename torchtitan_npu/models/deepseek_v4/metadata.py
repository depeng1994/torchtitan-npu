# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed compression metadata for DeepSeek-V4.

Built once per batch by the model's ``build_attention_masks`` from a
``VarlenMetadata`` stream.  The container grid is ``[1, S]`` (the DSV4 packed
scenario runs with ``local_batch_size == 1``; raise ``seq_len`` instead of
``local_batch_size``), so ``batch_size == 1`` and ``seq_len`` is the total
token count ``cu_seq_q[-1]``.

This module carries only the **common** contract — the kernel-contract
``plans`` (``cu_seqlens_cmp_k`` / ``block_remainder`` / ``gather_indices`` /
``block_positions`` / ``first_indices`` per ratio) consumed by the
Compressor, the kernels, and every attention path.  Further layers live
outside this module; the LI metadata provider fills ``plans[4].li_metadata``:

- the **reference tier** (per-token document ids/positions, the dense
  attendability mask, the container-slot scatter, the static block
  listing) is delivered by the ``metadata_extension`` seam
  (``reference.py``'s ``ReferenceMetadataExtension``, the default);
- the **context-parallel layer** (the plan builder + the dispatcher) lives
  in ``token_dispatcher.py``: a ratio-independent ``WindowPlan`` on the
  metadata (``window``) and the per-ratio block plans at ``plans[ratio]``,
  whose part 1 is derived directly over the plan blocks.

The kernel-layout derivation is the **plain-stream** contract:

- ``build_kernel_layout`` derives the per-document plans from the document
  boundaries alone (each document contributes its complete leading blocks,
  gathered contiguously; the ``len % ratio`` tail produces no entry).  It
  refuses context-parallel-shaped streams — those plans come from
  ``build_cp_plan`` (``token_dispatcher.py``).

Documents never span rows, and complete blocks never cross documents: a row's
compressed region is the concatenation of its documents' complete blocks,
padded to ``S // ratio`` slots.
"""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import torch
from torchtitan.models.common.attention import VarlenMetadata

if TYPE_CHECKING:
    from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
        CPVarlenMetadata,
    )

    from .token_dispatcher import ExchangePlan, WindowPlan

__all__ = [
    "CompressedBlockLayout",
    "CompressedKernelContract",
    "CompressedVarlenMetadata",
    "build_compressed_varlen_metadata",
    "build_index_dense_mask",
    "build_kernel_layout",
    "ensure_index_dense_masks",
]


@dataclass(kw_only=True, slots=True)
class CompressedBlockLayout:
    """Kernel contract for one compression ratio (the key of ``plans``).

    All tensors are built once per batch by the plan build.  Their leading
    dimensions are marked dynamic by the fused path so ``torch.compile``
    does not specialize on batch contents.

    The plan has two parts:

    - **part 1 — the unified contract** (CP and non-CP): the kernel tensors
      and the compressor contract (``gather_indices`` / ``block_positions``
      / ``first_indices``), consumed identically by the Compressor's
      dispatcher and the kernels regardless of context parallel;
    - **part 2 — the dispatcher fields** (CP only, ``None`` without): the
      block exchange routing (``exchange``), the container packing
      (``compressed_rows`` / ``out_width``) and the compressed-level
      gather (``cmp_k_global_gather_indices``).
    """

    cu_seqlens_cmp_k: torch.Tensor | None = None
    """Cumulative compressed-block lengths over the packed stream (int32)."""

    n_cmp_blocks_host: int | None = None
    """Host-cached total compressed block count."""

    block_remainder: torch.Tensor | None
    """Per-sequence incomplete-block remainder; ``None`` for ratio-1 plans."""

    gather_indices: torch.Tensor | None
    """Pooled block-row indices consumed by the token dispatcher."""

    block_positions: torch.Tensor | None = None
    """Document-relative block-start positions for compressed-key RoPE."""

    first_indices: torch.Tensor | None = None
    """Document-first block ids used by overlap masking."""

    li_metadata: torch.Tensor | None = None
    """Opaque ratio-4 LI metadata produced by the selected provider."""

    exchange: "ExchangePlan | None" = None
    """CP block-exchange routing; ``None`` without context parallel."""

    compressed_rows: torch.Tensor | None = None
    """Container-packing selection under context parallel."""

    out_width: int | None = None
    """Container grid width when the ratio owns a packed container."""

    cmp_k_global_gather_indices: torch.Tensor | None = None
    """Compressed-key global gather indices under context parallel."""


@dataclass(kw_only=True, slots=True)
class CompressedVarlenMetadata:
    """The DeepSeek-V4 varlen attention contract (the common part)."""

    varlen: "VarlenMetadata | CPVarlenMetadata"
    """Token-stream boundaries for the current rank."""

    plans: dict[int, CompressedBlockLayout]
    """Kernel contract for each ratio present in the model."""

    window: "WindowPlan | None" = None
    """Sliding-window exchange/assembly plan under context parallel."""

    seq_len_host: int | None = None
    """Host-cached total token count."""

    index_dense_masks: dict[int, torch.Tensor] = field(default_factory=dict)
    """Backend-independent causal masks for sparse index/candidate stages."""

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def seq_len(self) -> int:
        if self.seq_len_host is not None:
            return self.seq_len_host
        return int(self.varlen.cu_seq_q[-1].item())


@runtime_checkable
class CompressedKernelContract(Protocol):
    """The per-ratio compressor/kernel plan shape shared by all backends."""

    plans: dict[int, CompressedBlockLayout]


def build_kernel_layout(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
) -> dict[int, CompressedBlockLayout]:
    """Build the kernel-contract tier for a plain packed stream."""
    if not hasattr(varlen, "cu_seq_q"):
        raise TypeError(f"build_kernel_layout expects a varlen stream, got {type(varlen)}.")

    cu_seq_q = varlen.cu_seq_q
    if int(cu_seq_q[0].item()) != 0:
        raise ValueError(f"varlen stream must start at token 0, got cu_seq_q[0]={cu_seq_q[0]}.")
    if not torch.equal(cu_seq_q, varlen.cu_seq_k):
        raise ValueError(
            "build_kernel_layout requires a plain stream (cu_seq_q == "
            "cu_seq_k); context-parallel plans come from build_cp_plan."
        )
    seq_len = int(cu_seq_q[-1].item())

    cu = cu_seq_q.cpu().tolist()
    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    distinct_ratios = sorted({int(r) for r in compress_ratios})
    device = cu_seq_q.device
    plans: dict[int, CompressedBlockLayout] = {}
    for ratio in distinct_ratios:
        if ratio <= 1:
            plans[ratio] = CompressedBlockLayout(
                cu_seqlens_cmp_k=None,
                block_remainder=None,
                gather_indices=None,
            )
            continue
        c_lens = [length // ratio for length in lengths]
        cu_seqs = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=device),
                torch.tensor(c_lens, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32),
            ]
        )
        pieces = [
            torch.arange(k_start, k_start + ratio * cnt, dtype=torch.int64, device=device)
            for k_start, cnt in zip(cu[:-1], c_lens, strict=True)
            if cnt
        ]
        positions = [torch.arange(0, ratio * cnt, ratio, dtype=torch.int32, device=device) for cnt in c_lens if cnt]
        gather = torch.cat(pieces, dim=0) if pieces else torch.empty((0,), dtype=torch.int64, device=device)
        block_positions = (
            torch.cat(positions, dim=0) if positions else torch.empty((0,), dtype=torch.int32, device=device)
        )
        first_indices = cu_seqs[:-1][torch.diff(cu_seqs) > 0].to(torch.int64)
        plans[ratio] = CompressedBlockLayout(
            cu_seqlens_cmp_k=cu_seqs,
            n_cmp_blocks_host=sum(length // ratio for length in lengths),
            block_remainder=torch.tensor(
                [length % ratio for length in lengths],
                dtype=torch.int32,
                device=device,
            ),
            gather_indices=gather,
            block_positions=block_positions,
            first_indices=first_indices,
            compressed_rows=None,
            out_width=seq_len // ratio,
        )
    return plans


def build_compressed_varlen_metadata(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
) -> CompressedVarlenMetadata:
    """Build the common DSV4 varlen contract for one rank-local token stream."""
    plans = build_kernel_layout(varlen, compress_ratios)
    metadata = CompressedVarlenMetadata(
        varlen=varlen,
        plans=plans,
        seq_len_host=int(varlen.cu_seq_q[-1].item()),
    )
    return ensure_index_dense_masks(metadata)


def build_index_dense_mask(
    metadata: CompressedVarlenMetadata,
    ratio: int,
) -> torch.Tensor:
    """Build the causal index-selection mask for any metadata backend.

    The returned shape is ``[1, 1, query_len, container_len]`` and includes
    document and causal reachability only; candidate filtering remains the
    indexer's responsibility.
    """
    cached = metadata.index_dense_masks.get(ratio)
    if cached is not None:
        return cached
    if ratio < 0:
        raise ValueError(f"compression ratio must be non-negative, got {ratio}")
    ratio = max(ratio, 1)
    cu_q = metadata.varlen.cu_seq_q
    seq_len = metadata.seq_len
    device = cu_q.device
    lengths = torch.diff(cu_q).to(torch.long)
    doc_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=device, dtype=torch.long), lengths)
    token_starts = cu_q[:-1].to(torch.long)
    positions = torch.arange(seq_len, device=device, dtype=torch.long) - token_starts[doc_ids]

    if ratio == 1:
        return (
            ((doc_ids[:, None] == doc_ids[None, :]) & (positions[:, None] >= positions[None, :]))
            .unsqueeze(0)
            .unsqueeze(1)
        )

    plan = metadata.plans.get(ratio)
    if plan is None or plan.cu_seqlens_cmp_k is None:
        raise ValueError(f"metadata has no compressed plan for ratio={ratio}")
    cu_cmp = plan.cu_seqlens_cmp_k.to(torch.long)
    container_len = plan.out_width
    if container_len is None:
        container_len = seq_len // ratio
    block_ids = torch.arange(container_len, device=device, dtype=torch.long)
    block_doc = torch.searchsorted(cu_cmp[1:], block_ids, right=True)
    block_local = block_ids - cu_cmp[block_doc]
    valid = block_ids < cu_cmp[-1]
    block_doc = block_doc.masked_fill(~valid, -1)
    block_local = block_local.masked_fill(~valid, -1)
    causal_limit = (positions + 1) // ratio
    return (
        ((doc_ids[:, None] == block_doc[None, :]) & (block_local[None, :] < causal_limit[:, None]))
        .unsqueeze(0)
        .unsqueeze(1)
    )


def ensure_index_dense_masks(metadata: CompressedVarlenMetadata) -> CompressedVarlenMetadata:
    """Materialize index masks once at the eager metadata boundary."""
    if not metadata.index_dense_masks:
        metadata.index_dense_masks = {ratio: build_index_dense_mask(metadata, ratio) for ratio in metadata.plans}
    return metadata


def register_pytree_node_for_dataclass(cls: type) -> None:
    """Register a kw-only dataclass as a pytree node (idempotent)."""
    from torch.utils._pytree import SUPPORTED_NODES, GetAttrKey, KeyEntry, register_pytree_node

    if cls in SUPPORTED_NODES:
        return
    field_names = [f.name for f in fields(cls)]

    def flatten(obj):
        return [getattr(obj, name) for name in field_names], None

    def flatten_with_keys(obj) -> tuple[list[tuple[KeyEntry, Any]], None]:
        keys = cast(
            "list[tuple[KeyEntry, Any]]",
            [(GetAttrKey(name), getattr(obj, name)) for name in field_names],
        )
        return keys, None

    def unflatten(values, context):
        return cls(**dict(zip(field_names, values, strict=True)))

    register_pytree_node(
        cls,
        flatten,
        unflatten,
        flatten_with_keys_fn=flatten_with_keys,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
    )


register_pytree_node_for_dataclass(CompressedBlockLayout)
register_pytree_node_for_dataclass(CompressedVarlenMetadata)
