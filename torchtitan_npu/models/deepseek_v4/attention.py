# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask
from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttention,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.models.deepseek_v4.golden import golden_enabled
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor, Indexer, LightningIndexer
from .metadata import CompressedVarlenMetadata
from .reference import ReferenceCompressedVarlenMetadata
from .token_dispatcher import CPTokenDispatcher


class CompressedSparseInnerAttention(FlexAttention):
    """DeepSeek sparse attention core for DeepSeek-V4 (varlen-typed reference).

    The core attends over the concatenated container KV ``[0, S + n_cmp + 1)``,
    where the first ``S`` positions are the uncompressed sliding-window KV
    (``swa_k``), the next ``n_cmp`` positions are the compressed KV in the
    container grid, and the last position is a learned attention sink token.

    The compressed-selection contract is explicit: callers may provide
    ``sparse_indices`` and, for derived attention variants, a materialized
    ``compress_ratio``.  The default V4 path keeps its historical behavior:
    ratio-4 derives Top-K from the local indexer and ratio-128 attends all
    causally reachable compressed blocks.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(  # pyrefly: ignore [bad-override]
        VarlenAttention.Config
    ):
        window_size: int  # pyrefly: ignore [bad-override]
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        kernel_options: dict = field(default_factory=dict)

    def __init__(self, config: Config) -> None:
        super().__init__(config)  # pyrefly: ignore [bad-argument-type]
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.block_size = config.block_size

    def _build_varlen_block_mask(
        self,
        metadata: ReferenceCompressedVarlenMetadata,
        topk_indices: torch.Tensor | None,
        n_cmp: int,
        device,
        *,
        compress_ratio: int | None = None,
    ) -> BlockMask:
        """Build the document-packed mask for a materialized compression ratio."""
        bsz, seqlen = metadata.batch_size, metadata.seq_len
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        kv_len = seqlen + n_cmp + 1
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq
        sink_idx = seqlen + n_cmp
        ratio = self.compress_ratio if compress_ratio is None else compress_ratio
        window_size = self.window_size
        if metadata.plans.get(ratio) is None:
            raise ValueError(f"No compression layout for ratio={ratio}.")
        ref = metadata.reference.ratios[ratio]
        if ref.static_blocks is None:
            raise ValueError(f"Reference layout for ratio={ratio} has no static block mask.")

        bm = ref.static_blocks.expand(  # pyrefly: ignore [missing-attribute]
            bsz, 1, -1, -1
        ).clone()
        if topk_indices is not None:
            cmp_block_of = (seqlen + torch.arange(n_cmp, device=device)) // bk
            block_of_topk = cmp_block_of[topk_indices].reshape(bsz, n_q_blocks, bq * topk_indices.size(-1))
            bm[:, 0].scatter_add_(
                -1,
                block_of_topk.clamp(0, n_kv_blocks - 1),
                torch.ones_like(block_of_topk, dtype=torch.int32),
            )
        bm = (bm > 0).to(torch.int32)  # pyrefly: ignore [missing-attribute]
        kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
        kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(torch.int32)

        cmp_sel = torch.zeros(bsz, seqlen, max(n_cmp, 1), dtype=torch.bool, device=device)
        if topk_indices is not None:
            cmp_sel.scatter_(2, topk_indices.clamp(0, max(n_cmp, 1) - 1), True)

        doc_of_token = metadata.reference.doc_of_token
        pos_in_doc = metadata.reference.pos_in_doc
        if n_cmp > 0:
            cmp_doc = ref.doc_of_block
            cmp_local = ref.block_local
            if cmp_doc is None or cmp_local is None:
                raise ValueError(f"Reference layout for ratio={ratio} has no compressed-slot coordinates.")
        else:
            cmp_doc = torch.full((bsz, 1), -1, dtype=torch.int32, device=device)
            cmp_local = torch.full((bsz, 1), -1, dtype=torch.int32, device=device)

        def csa_varlen_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            doc_q = doc_of_token[b, q_idx]
            kv_safe = kv_idx.clamp(0, seqlen - 1)
            swa = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < window_size)
                & (doc_of_token[b, kv_safe] == doc_q)
            )
            is_sink = kv_idx == sink_idx
            if n_cmp > 0 and ratio > 0:
                c = kv_idx - seqlen
                in_cmp = (c >= 0) & (c < n_cmp)
                c_safe = c.clamp(0, n_cmp - 1)
                same_doc = cmp_doc[b, c_safe] == doc_q  # pyrefly: ignore [unsupported-operation]
                causal = cmp_local[b, c_safe] < torch.div(  # pyrefly: ignore [unsupported-operation]
                    pos_in_doc[b, q_idx] + 1,
                    ratio,
                    rounding_mode="floor",
                )
                if topk_indices is not None:
                    topk_sel = cmp_sel[b, q_idx, c_safe]
                    return swa | (in_cmp & same_doc & causal & topk_sel) | is_sink
                return swa | (in_cmp & same_doc & causal) | is_sink
            return swa | is_sink

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=(bq, bk),
            mask_mod=csa_varlen_mask_mod,
            seq_lengths=(seqlen, kv_len),
        )

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        sparse_indices=None,
        attn_sink: torch.Tensor | None = None,
        *,
        attention_masks: ReferenceCompressedVarlenMetadata | None = None,
        compress_ratio: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(attention_masks, CompressedVarlenMetadata):
            raise TypeError(
                "CompressedSparseInnerAttention requires CompressedVarlenMetadata "
                f"attention masks, got {type(attention_masks)}."
            )
        if attn_sink is None:
            raise ValueError("CompressedSparseInnerAttention requires attn_sink")

        metadata = attention_masks
        bsz, seqlen, _, head_dim = q.size()
        n_cmp = 0 if cmp_k is None else cmp_k.size(1)
        sink_idx = seqlen + n_cmp
        ratio = self.compress_ratio if compress_ratio is None else compress_ratio

        topk_indices = sparse_indices
        if topk_indices is None and compress_ratio is None and self.compress_ratio == 4:
            if idx_q is None or idx_k is None or idx_w is None:
                raise ValueError(
                    "CompressedSparseInnerAttention requires idx_q, idx_k, and idx_w when compress_ratio=4"
                )
            if metadata.plans.get(4) is None:
                raise ValueError(
                    "CompressedSparseInnerAttention requires the ratio-4 compression layout for indexer selection."
                )
            topk_indices, _ = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                metadata.reference.ratios[  # pyrefly: ignore [bad-argument-type]
                    4
                ].dense_mask,
                self.index_topk,
            )

        kv = swa_k.unsqueeze(2)
        if cmp_k is not None:
            kv = torch.cat([kv, cmp_k.unsqueeze(2)], dim=1)
        sink_kv = kv.new_zeros((bsz, 1, 1, head_dim))
        kv = torch.cat([kv, sink_kv], dim=1)

        block_mask = self._build_varlen_block_mask(
            metadata,
            topk_indices,
            n_cmp,
            q.device,
            compress_ratio=ratio,
        )

        def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(
                kv_idx == sink_idx,
                attn_sink[h],  # pyrefly: ignore [unsupported-operation]
                score,
            )

        return super().forward(
            q,
            kv,
            kv,
            attention_masks=block_mask,
            score_mod=v4_sink_score_mod,
            scale=self.softmax_scale,
            enable_gqa=True,
        )


class CompressedSparseAttention(Module):
    """Thin CP boundary containing LI and the sparse attention core."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lightning_indexer: LightningIndexer.Config | None
        inner_attention: CompressedSparseInnerAttention.Config

    def __init__(self, config: Config):
        super().__init__()
        self.lightning_indexer = config.lightning_indexer.build() if config.lightning_indexer is not None else None
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        *,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        sparse_indices=None,
        compress_ratio=None,
        attn_sink=None,
        attention_masks=None,
    ):
        if sparse_indices is None and self.lightning_indexer is not None:
            sparse_indices = self.lightning_indexer(idx_q, idx_k, idx_w, attention_masks=attention_masks)
        return self.inner_attention(
            q,
            swa_k,
            cmp_k,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_w=idx_w,
            sparse_indices=sparse_indices,
            attn_sink=attn_sink,
            attention_masks=attention_masks,
            compress_ratio=compress_ratio,
        )


@dataclass(slots=True)
class LongRangeContext:
    """Materialized long-range inputs consumed by the sparse-attention core."""

    compressed_kv: torch.Tensor | None = None
    index_q: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    index_weight: torch.Tensor | None = None
    sparse_indices: torch.Tensor | None = None
    compress_ratio: int | None = None


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        inner_attention: Module.Config
        rope: RoPE.Config
        head_dim: int
        rope_head_dim: int
        q_lora_rank: int
        n_groups: int
        compress_ratio: int
        norm_eps: float
        post_q_rms_norm: bool = True

        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config

        compressor: Compressor.Config | None
        indexer: Indexer.Config | None
        compressed_sparse_attention: CompressedSparseAttention.Config
        token_dispatcher: CPTokenDispatcher.Config = field(default_factory=CPTokenDispatcher.Config)

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.post_q_rms_norm = cfg.post_q_rms_norm
        self.rope = cfg.rope.build()
        self.token_dispatcher = cfg.token_dispatcher.build()
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        self.attn_sink = torch.nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))
        self.compressor = cfg.compressor.build() if cfg.compressor is not None else None
        self.indexer = cfg.indexer.build() if cfg.indexer is not None else None
        self.compressed_sparse_attention = cfg.compressed_sparse_attention.build()

    @property
    def inner_attention(self):
        """Read-only compatibility access to the wrapped attention module."""
        return self.compressed_sparse_attention.inner_attention

    def parallelize(self, parallel_dims) -> None:
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(cp_mesh=parallel_dims.get_optional_mesh("cp"))

    def _golden_rope(self, x: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False):
        """Apply the reference complex-pair rotation through the float cache."""
        cache = self.rope._reshape_cache(x, positions)
        if isinstance(cache, tuple):
            cos, sin = cache
            freqs = torch.complex(cos[..., ::2], sin[..., ::2])
        else:
            freqs = cache
        if inverse:
            freqs = freqs.conj()
        real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        c, s = freqs.real, freqs.imag
        return torch.stack((real * c - imag * s, imag * c + real * s), dim=-1).flatten(-2).type_as(x)

    def _project_q(self, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, -1, self.head_dim)
        if self.post_q_rms_norm:
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self._golden_rope(q_rope, positions) if golden_enabled() else self.rope(q_rope, positions=positions)
        return qr, torch.cat([q_nope, q_rope], dim=-1)

    def _project_window_kv(self, x, attention_masks, positions) -> torch.Tensor:
        rd = self.rope_head_dim
        swa_k = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(swa_k, [self.head_dim - rd, rd], dim=-1)
        kv_input = kv_rope.unsqueeze(2)
        kv_rope = (
            self._golden_rope(kv_input, positions.reshape(1, -1)).squeeze(2)
            if golden_enabled()
            else self.rope(kv_input, positions=positions.reshape(1, -1)).squeeze(2)
        )
        swa_k = torch.cat([kv_nope, kv_rope], dim=-1)
        return self.token_dispatcher.gather(swa_k, attention_masks.window)

    def _build_long_range_context(self, x, qr, attention_masks, positions) -> LongRangeContext:
        """Build the layer-local V4 compressed KV/indexer inputs."""
        cmp_k = None
        idx_q = idx_k = idx_w = None
        if self.compress_ratio > 1:
            if self.compressor is None:
                raise ValueError("compress_ratio > 1 requires the compressor submodule")
            pooled = self.compressor(x, attention_masks, positions=positions)
            plan = attention_masks.plans[self.compress_ratio]
            cmp_k = self.token_dispatcher.select(pooled, plan)

        if self.indexer is not None and self.compress_ratio > 1:
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                attention_masks=attention_masks,
            )
            plan = attention_masks.plans[self.compress_ratio]
            if plan.gather_indices is not None:
                idx_k = self.token_dispatcher.select(idx_k, plan)

        return LongRangeContext(
            compressed_kv=cmp_k,
            index_q=idx_q,
            index_k=idx_k,
            index_weight=idx_w,
            # None means "use the core's own configured ratio".  This keeps
            # the default V4 ratio-4 path responsible for its local Top-K;
            # derived cross-layer policies set an explicit active ratio.
            compress_ratio=None,
        )

    def _apply_sparse_attention(
        self,
        q: torch.Tensor,
        swa_k: torch.Tensor,
        context: LongRangeContext,
        attention_masks,
    ) -> torch.Tensor:
        return self.compressed_sparse_attention(
            q,
            swa_k,
            context.compressed_kv,
            idx_q=context.index_q,
            idx_k=context.index_k,
            idx_w=context.index_weight,
            sparse_indices=context.sparse_indices,
            compress_ratio=context.compress_ratio,
            attn_sink=self.attn_sink,
            attention_masks=attention_masks,
        )

    def _project_output(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = o.shape[:2]
        rd = self.rope_head_dim
        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = (
            self._golden_rope(o_rope, positions, inverse=True)
            if golden_enabled()
            else self.rope(o_rope, positions=positions, inverse=True)
        )
        o = torch.cat([o_nope, o_rope], dim=-1)
        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        if golden_enabled():
            wo_a = self.wo_a.weight.view(n_local_groups, self.wo_a.out_features, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        else:
            o = self.wo_a(o)
        o = o.reshape(bsz, seqlen, -1)
        return self.wo_b(o)

    def forward(self, x, attention_masks, positions):
        """V4 attention forward through version-neutral extension seams."""
        qr, q = self._project_q(x, positions)
        swa_k = self._project_window_kv(x, attention_masks, positions)
        context = self._build_long_range_context(x, qr, attention_masks, positions)
        o = self._apply_sparse_attention(q, swa_k, context, attention_masks)
        return self._project_output(o, positions)
