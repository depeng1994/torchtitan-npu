# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license in the repository.
"""V4.1 CP1 sparse attention. Hardware and numerical validation are pending.

Second-stream presence is independent of compression: ratio 1 carries shared
full-resolution KV, ratio 2 carries compressed KV. Indices are provided by the
model; this module never computes LI or its auxiliary loss.
"""

from dataclasses import dataclass, fields

import torch
from cann_ops_transformer import sparse_flash_mla_grad_metadata, sparse_flash_mla_metadata

from torchtitan_npu.models.deepseek_v41.metadata import register_pytree_node_for_dataclass
from torchtitan_npu.models.deepseek_v41.reference import (
    ReferenceCompressedVarlenMetadata,
    ReferenceMetadataExtension,
)
from torchtitan_npu.models.deepseek_v41.sparse_attention import (
    V41SparseAttention,
    _localize_precomputed_indices,
)
from torchtitan_npu.override import _IS_A5


@dataclass(kw_only=True, slots=True)
class AscV41Metadata(ReferenceCompressedVarlenMetadata):
    # Explicit ratio-1 second-stream boundaries; no fake compressed block plan.
    shared_cu: torch.Tensor
    shared_remainder: torch.Tensor


register_pytree_node_for_dataclass(AscV41Metadata)


class AscV41MetadataExtension(ReferenceMetadataExtension):
    @dataclass(kw_only=True, slots=True)
    class Config(ReferenceMetadataExtension.Config):
        pass

    def __call__(self, metadata):
        reference = super().__call__(metadata)
        cu = reference.varlen.cu_seq_q
        return AscV41Metadata(
            **{f.name: getattr(reference, f.name) for f in fields(reference)},
            shared_cu=cu,
            shared_remainder=torch.zeros_like(cu[1:]),
        )


def _kernel_options(cu_q, cu_cmp, remainder, ratio, window):
    return dict(
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_q,
        cu_seqlens_cmp_kv=cu_cmp,
        cmp_residual_kv=remainder,
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=4,
        cmp_mask_mode=0 if _IS_A5 and cu_cmp is None else 3,
        ori_win_left=window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
    )


class _SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        original,
        shared,
        indices,
        sink,
        cu_q,
        cu_cmp,
        remainder,
        scale,
        ratio,
        window,
    ):
        options = _kernel_options(cu_q, cu_cmp, remainder, ratio, window)
        # K can differ between the candidate and indexer paths. Use the actual
        # supplied shape rather than assuming the LI configuration's top-k.
        geometry = dict(
            ori_topk=0,
            cmp_topk=0 if indices is None else indices.shape[-1],
            has_ori_kv=True,
            has_cmp_kv=shared is not None,
        )
        forward_meta = sparse_flash_mla_metadata(
            q.shape[1],
            1,
            q.shape[2],
            ori_topk_length=None,
            cmp_topk_length=None,
            **options,
            **geometry,
        )
        backward_meta = sparse_flash_mla_grad_metadata(q.shape[1], 1, q.shape[2], **options, **geometry)
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=original,
            cmp_kv=shared,
            cmp_sparse_indices=indices,
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sink,
            metadata=forward_meta,
            softmax_scale=scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(q, original, shared, indices, sink, cu_q, cu_cmp, remainder, output, lse, backward_meta)
        ctx.scale, ctx.ratio, ctx.window = scale, ratio, window
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        q, original, shared, indices, sink, cu_q, cu_cmp, remainder, output, lse, metadata = ctx.saved_tensors
        dq, doriginal, dshared, dsink, _, _ = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=original,
            cmp_kv=shared,
            ori_sparse_indices=None,
            cmp_sparse_indices=indices,
            sinks=sink,
            metadata=metadata,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            ori_topk_length=None,
            cmp_topk_length=None,
            softmax_scale=ctx.scale,
            **_kernel_options(cu_q, cu_cmp, remainder, ctx.ratio, ctx.window),
        )
        return dq, doriginal, dshared if shared is not None else None, None, dsink, None, None, None, None, None, None


class AscV41SparseAttention(V41SparseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(V41SparseAttention.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        sparse_indices=None,
        attn_sink=None,
        *,
        attention_masks=None,
        compress_ratio=None,
    ):
        metadata = attention_masks
        if not isinstance(metadata, AscV41Metadata):
            raise TypeError("V4.1 asc requires the deepseek_v41.sparse_attn.asc_metadata override")
        ratio = self.compress_ratio if compress_ratio is None else compress_ratio
        if ratio not in (0, 1, 2):
            raise ValueError(f"V4.1 sparse adapter supports ratios 0/1/2, got {ratio}")
        if q.ndim != 4 or q.shape[0] != 1 or swa_k.shape != (1, q.shape[1], q.shape[-1]):
            raise ValueError("V4.1 SMLA requires CP1 packed Q [1,S,H,D] and original KV [1,S,D]")
        if attn_sink is None:
            raise ValueError("V4.1 SMLA requires per-head attention sinks")
        if q.dtype != torch.bfloat16 or swa_k.dtype != q.dtype:
            raise ValueError("V4.1 SMLA requires BF16 Q/KV; no implicit precision conversion")
        shared = cu_cmp = remainder = indices = None
        if ratio == 0 and cmp_k is not None:
            raise ValueError("window-only ratio 0 must not receive a second KV stream")
        if ratio > 0:
            if cmp_k is None or cmp_k.ndim != 3 or cmp_k.shape[0] != 1 or cmp_k.dtype != q.dtype:
                raise ValueError("ratio 1/2 requires a BF16 second KV stream [1,N,D]")
            if sparse_indices is None:
                raise ValueError("ratio 1/2 fused attention requires externally supplied sparse indices")
            if ratio == 1:
                if cmp_k.shape != swa_k.shape:
                    raise ValueError("ratio 1 requires full-resolution shared KV")
                cu_cmp, remainder = metadata.shared_cu, metadata.shared_remainder
                shared = cmp_k.flatten(0, 1)
            else:
                plan = metadata.plans.get(ratio)
                if plan is None or plan.cu_seqlens_cmp_k is None or plan.n_cmp_blocks_host is None:
                    raise ValueError("ratio 2 requires complete packed compression metadata")
                cu_cmp, remainder = plan.cu_seqlens_cmp_k, plan.block_remainder
                shared = cmp_k.flatten(0, 1)[: plan.n_cmp_blocks_host]
            # TND kernels consume document-local indices; cu_cmp supplies the
            # document offsets. Preserve the reference cross-document mask.
            local = _localize_precomputed_indices(sparse_indices, metadata, ratio).flatten(0, 1)
            lengths = torch.diff(metadata.varlen.cu_seq_q).long()
            docs = torch.repeat_interleave(torch.arange(lengths.numel(), device=q.device), lengths)
            starts, ends = cu_cmp[docs], cu_cmp[docs + 1]
            valid = local >= 0
            if ratio > 1:
                # The reference takes at most the document's compressed count
                # from each index row, retaining padding as invalid slots.
                valid = valid & (torch.arange(local.shape[-1], device=q.device) < (ends - starts).unsqueeze(-1))
            indices = torch.where(valid, local, -1).to(torch.int32).unsqueeze(1).contiguous()
            shared = shared.unsqueeze(1).contiguous()
        output = _SparseMLA.apply(
            q.flatten(0, 1).contiguous(),
            swa_k.flatten(0, 1).unsqueeze(1).contiguous(),
            shared,
            indices,
            attn_sink.float(),
            metadata.varlen.cu_seq_q,
            cu_cmp,
            remainder,
            self.softmax_scale,
            ratio,
            self.window_size,
        )
        return output.reshape_as(q)
