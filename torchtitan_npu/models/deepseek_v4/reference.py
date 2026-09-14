# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DSV4 reference-attention metadata tier and its extension.

The model-dir default attention (``CompressedSparseInnerAttention``) and the
eager golden reference consume a reference tier on the common metadata: the
per-token document ids and positions, the per-ratio dense attendability mask,
the container-slot scatter, and the static attention block listing.  The tier
is **no-CP-shaped** (contiguous documents): the reference build enforces
``cu_seq_q == cu_seq_k``.

Whether a ratio is materialized as a real compressed/global-KV container is an
explicit metadata-extension policy.  The ratio value itself does not imply
ownership or materialization; the default V4 policy keeps ratio-1 unmaterialized.
"""

from dataclasses import dataclass
from typing import cast

import torch

from torchtitan_npu.models.common.metadata_extension import MetadataExtension

from .metadata import CompressedBlockLayout, CompressedVarlenMetadata


@dataclass(kw_only=True, slots=True)
class ReferenceRatioLayout:
    """Reference-attention tensors for one compression ratio."""

    dense_mask: torch.Tensor | None = None
    """Boolean attendability over a materialized container grid."""

    doc_of_block: torch.Tensor | None = None
    """Document id of each materialized container slot."""

    block_local: torch.Tensor | None = None
    """Document-local position/block index of each materialized slot."""

    static_blocks: torch.Tensor | None = None
    """Static window/compressed/sink block listing for the reference core."""


@dataclass(kw_only=True, slots=True)
class ReferenceLayout:
    """The model-dir reference-attention tier."""

    doc_of_token: torch.Tensor
    pos_in_doc: torch.Tensor
    ratios: dict[int, ReferenceRatioLayout]


def _build_dense_mask(
    doc_of_block: torch.Tensor,
    block_local: torch.Tensor,
    doc_of_token: torch.Tensor,
    pos_in_doc: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """Attendability over the container grid: same document and causal."""
    same_doc = doc_of_block.unsqueeze(1) == doc_of_token.unsqueeze(2)
    causal_limit = torch.div(pos_in_doc + 1, ratio, rounding_mode="floor").unsqueeze(2)
    causal = block_local.unsqueeze(1) < causal_limit
    return (same_doc & causal).unsqueeze(1)


def _build_static_blocks(
    seq_len: int,
    n_cmp: int,
    ratio: int,
    window_size: int,
    block_size: int | tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Static sliding-window, materialized-container and sink block listing."""
    bq, bk = block_size if isinstance(block_size, tuple) else (block_size, block_size)
    assert seq_len % bq == 0, f"seq_len ({seq_len}) must be divisible by {bq}"
    kv_len = seq_len + n_cmp + 1
    n_kv_blocks = (kv_len + bk - 1) // bk
    n_q_blocks = seq_len // bq
    sink_idx = seq_len + n_cmp

    bm = torch.zeros(1, 1, n_q_blocks, n_kv_blocks, dtype=torch.int32, device=device)
    q0 = (torch.arange(n_q_blocks, device=device) * bq).unsqueeze(1)
    kv_ids = torch.arange(n_kv_blocks, device=device).unsqueeze(0)

    first_window_block = (q0 - window_size + 1).clamp_min(0) // bk
    last_window_block = (q0 + bq - 1) // bk
    window_blocks = (kv_ids >= first_window_block) & (kv_ids <= last_window_block)
    bm[:, 0] = (bm[:, 0] > 0).to(torch.int32) | window_blocks.to(torch.int32)

    if n_cmp > 0:
        first_cmp_block = seq_len // bk
        last_cmp_block = (seq_len + n_cmp - 1) // bk
        cmp_blocks = (kv_ids >= first_cmp_block) & (kv_ids <= last_cmp_block)
        bm[:, 0] = (bm[:, 0] > 0).to(torch.int32) | cmp_blocks.to(torch.int32)

    bm[:, 0, :, sink_idx // bk] = 1
    return bm


def _materialize_identity_ratio(
    *,
    batch_size: int,
    seq_len: int,
    doc_of_token: torch.Tensor,
    pos_in_doc: torch.Tensor,
    window_size: int,
    block_size: int | tuple[int, int],
    device: torch.device,
) -> ReferenceRatioLayout:
    """Materialize ratio-1 as a true token-for-token global-KV container."""
    doc_of_block = doc_of_token.clone()
    block_local = pos_in_doc.clone()
    return ReferenceRatioLayout(
        dense_mask=_build_dense_mask(
            doc_of_block,
            block_local,
            doc_of_token,
            pos_in_doc,
            1,
        ),
        doc_of_block=doc_of_block.view(batch_size, seq_len),
        block_local=block_local.view(batch_size, seq_len),
        static_blocks=_build_static_blocks(
            seq_len,
            seq_len,
            1,
            window_size,
            block_size,
            device,
        ),
    )


def derive_reference_layout(
    cu_seq_q: torch.Tensor,
    plans: dict[int, "CompressedBlockLayout"],
    batch_size: int,
    seq_len: int,
    window_size: int,
    block_size: int | tuple[int, int],
    device: torch.device,
    *,
    materialized_ratios: tuple[int, ...] = (),
) -> ReferenceLayout:
    """Build the no-CP reference tier for the configured materialization policy."""
    total_tokens = int(cu_seq_q[-1].item())
    lengths = torch.diff(cu_seq_q).to(torch.int32)
    doc_of_token_flat = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32),
        lengths,
    )
    pos_in_doc_flat = (torch.arange(total_tokens, device=device) - cu_seq_q[doc_of_token_flat.long()]).to(torch.int32)
    doc_of_token = doc_of_token_flat.view(batch_size, seq_len)
    pos_in_doc = pos_in_doc_flat.view(batch_size, seq_len)

    materialized = set(materialized_ratios)
    ratios: dict[int, ReferenceRatioLayout] = {}
    for ratio, plan in plans.items():
        if ratio == 1 and ratio in materialized:
            ratios[ratio] = _materialize_identity_ratio(
                batch_size=batch_size,
                seq_len=seq_len,
                doc_of_token=doc_of_token,
                pos_in_doc=pos_in_doc,
                window_size=window_size,
                block_size=block_size,
                device=device,
            )
            continue
        if ratio <= 1:
            ratios[ratio] = ReferenceRatioLayout(
                static_blocks=_build_static_blocks(seq_len, 0, max(ratio, 1), window_size, block_size, device),
            )
            continue

        container_width = seq_len // ratio
        cu_cmp = plan.cu_seqlens_cmp_k
        assert cu_cmp is not None, "materialized ratio > 1 plans must carry cu_seqlens_cmp_k"
        n_blocks = int(cu_cmp[-1].item())
        if n_blocks == 0:
            empty_slots = torch.full(
                (batch_size * container_width,),
                -1,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, container_width)
            doc_of_block = empty_slots
            block_local = empty_slots
            dense_mask = _build_dense_mask(
                empty_slots,
                empty_slots,
                doc_of_token,
                pos_in_doc,
                ratio,
            )
        else:
            bids = torch.arange(n_blocks, device=device, dtype=torch.int64)
            seq_ids = torch.searchsorted(cu_cmp[1:], bids, right=True)
            local_idx = bids - cu_cmp[seq_ids]
            doc_of_block = torch.full(
                (batch_size * container_width,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            block_local = torch.full(
                (batch_size * container_width,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            doc_of_block[:n_blocks] = seq_ids.to(torch.int32)
            block_local[:n_blocks] = local_idx.to(torch.int32)
            dense_mask = _build_dense_mask(
                doc_of_block.view(batch_size, container_width),
                block_local.view(batch_size, container_width),
                doc_of_token,
                pos_in_doc,
                ratio,
            )
        ratios[ratio] = ReferenceRatioLayout(
            dense_mask=dense_mask,
            doc_of_block=doc_of_block.view(batch_size, container_width),
            block_local=block_local.view(batch_size, container_width),
            static_blocks=_build_static_blocks(
                seq_len,
                seq_len // ratio,
                ratio,
                window_size,
                block_size,
                device,
            ),
        )

    return ReferenceLayout(
        doc_of_token=doc_of_token,
        pos_in_doc=pos_in_doc,
        ratios=ratios,
    )


@dataclass(kw_only=True, slots=True)
class ReferenceCompressedVarlenMetadata(CompressedVarlenMetadata):
    """The common contract plus the reference tier."""

    reference: ReferenceLayout


class ReferenceMetadataExtension(MetadataExtension):
    """Reference-tier post-process of the common attention metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(MetadataExtension.Config):
        block_size: int | tuple[int, int] = (128, 128)
        materialized_ratios: tuple[int, ...] = ()
        """Ratios that represent real long-range containers in this model."""

    def __call__(self, metadata) -> ReferenceCompressedVarlenMetadata:
        if not isinstance(metadata, CompressedVarlenMetadata):
            raise TypeError(
                "the reference tier requires the model-dir common metadata "
                f"(CompressedVarlenMetadata), got {type(metadata)}."
            )
        if not torch.equal(metadata.varlen.cu_seq_q, metadata.varlen.cu_seq_k):
            raise ValueError(
                "the reference tier requires cu_seq_q == cu_seq_k (contiguous "
                "documents); under context parallel the fused path is required"
            )
        cfg = cast("ReferenceMetadataExtension.Config", self.config)
        reference = derive_reference_layout(
            metadata.varlen.cu_seq_q,
            metadata.plans,
            metadata.batch_size,
            metadata.seq_len,
            cfg.window_size,
            cfg.block_size,
            metadata.varlen.cu_seq_q.device,
            materialized_ratios=cfg.materialized_ratios,
        )
        return ReferenceCompressedVarlenMetadata(
            varlen=metadata.varlen,
            plans=metadata.plans,
            window=metadata.window,
            seq_len_host=metadata.seq_len_host,
            index_dense_masks=metadata.index_dense_masks,
            reference=reference,
        )
