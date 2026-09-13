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
from .metadata import CompressedVarlenMetadata, build_index_dense_mask
from .reference import ReferenceCompressedVarlenMetadata
from .token_dispatcher import CPTokenDispatcher


class CompressedSparseInnerAttention(FlexAttention):
    """DeepSeek sparse attention core for DeepSeek-V4 (varlen-typed reference).

    The core attends over the concatenated container KV ``[0, S + n_cmp + 1)``,
    where the first ``S`` positions are the uncompressed sliding-window KV
    (``swa_k``), the next ``n_cmp`` positions are the compressed KV in the
    ``[B, S // ratio, D]`` container grid (``cmp_k``), and the last position is
    a learned attention sink token:

    - sliding window: fixed ``mask_mod`` pattern, restricted to the query
      token's document;
    - compressed blocks: for HCA (``compress_ratio=128``) all causally
      reachable blocks of the same document, also a fixed pattern; for CSA
      (``compress_ratio=4``) each query attends only its top-k selected
      container slots, chosen by ``Indexer.select`` against the dense mask
      from the model's ``build_attention_masks``;
    - attention sink: always attendable via ``score_mod``.

    ``_build_block_mask`` is the single-document container formulation (kept
    for upstream parity and its unit test); ``_build_varlen_block_mask`` is the
    document-packed path driven by ``CompressedVarlenMetadata``.  NPU overrides
    replace the whole ``forward`` (fused SMLA/CSA kernels consume the raw
    ``q / swa_k / cmp_k / idx_q / idx_k / idx_w`` tensors).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(  # pyrefly: ignore [bad-override]
        VarlenAttention.Config
    ):
        # Redeclared as the int DSA window (replaces the inherited varlen
        # ``window_size`` tuple, which is never used by the DSA path).
        window_size: int  # pyrefly: ignore [bad-override]
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        # Consumed by the inherited ``FlexAttention.__init__`` (kernel options
        # for the flex_attention backend of the reference path).
        kernel_options: dict = field(default_factory=dict)

    def __init__(self, config: Config) -> None:
        super().__init__(config)  # pyrefly: ignore [bad-argument-type]
        # Subclasses read ``self.window_size`` as an int.
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
    ) -> BlockMask:
        """Document-packed block mask driven by ``CompressedVarlenMetadata``.

        The block listing is a superset (window range, selected/full compressed
        region, sink); ``mask_mod`` applies the exact per-token predicates
        (same document, per-document causal limit, top-k selection).
        """
        bsz, seqlen = metadata.batch_size, metadata.seq_len
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        kv_len = seqlen + n_cmp + 1
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq
        sink_idx = seqlen + n_cmp
        ratio = getattr(self, "_v41_compress_ratio", self.compress_ratio)
        window_size = self.window_size
        if metadata.plans.get(ratio) is None:
            raise ValueError(f"No compression layout for ratio={ratio}.")
        ref = metadata.reference.ratios[ratio]

        # Static parts (window, sink, HCA range) are hoisted in the metadata;
        # only the CSA top-k blocks are scattered here.
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
        if ratio > 1 and n_cmp > 0:
            cmp_doc = ref.doc_of_block
            cmp_local = ref.block_local
        else:
            # No compressed slots: keep the gather safe with dummy values.
            cmp_doc = torch.full((bsz, max(n_cmp, 1)), -1, dtype=torch.int32, device=device)
            cmp_local = torch.full((bsz, max(n_cmp, 1)), -1, dtype=torch.int32, device=device)

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
            if ratio > 1:
                c = kv_idx - seqlen
                in_cmp = (c >= 0) & (c < n_cmp)
                c_safe = c.clamp(0, max(n_cmp, 1) - 1)
                same_doc = (
                    cmp_doc[  # pyrefly: ignore [unsupported-operation]
                        b, c_safe
                    ]
                    == doc_q
                )
                causal = cmp_local[  # pyrefly: ignore [unsupported-operation]
                    b, c_safe
                ] < torch.div(pos_in_doc[b, q_idx] + 1, ratio, rounding_mode="floor")
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

        topk_indices = getattr(self, "_v41_topk_indices", None)
        if topk_indices is None and self.compress_ratio == 4:
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

        block_mask = self._build_varlen_block_mask(metadata, topk_indices, n_cmp, q.device)

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
        self, q, swa_k, cmp_k=None, *, idx_q=None, idx_k=None, idx_w=None, attn_sink=None, attention_masks=None
    ):
        sparse_indices = None
        if self.lightning_indexer is not None:
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
        )


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
        # V4 re-scales the Q projection by RMS after ``wq_b`` (HF V4:
        # q_norm(wq_a) -> wq_b -> RMS rescale -> RoPE); V4.1 dropped the
        # second rescale (q_norm(wq_a) -> wq_b -> RoPE).
        post_q_rms_norm: bool = True

        # Declare submodule configs as fields so sharding can be assigned before
        # the modules are built.
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config

        # Built only for ``compress_ratio > 1`` layers (``indexer`` only for
        # ratio-4 CSA layers); the registry passes ``None`` otherwise.
        compressor: Compressor.Config | None
        indexer: Indexer.Config | None
        compressed_sparse_attention: CompressedSparseAttention.Config

        # The CP token dispatcher (the RoutedExperts mirror): a submodule of
        # the attention, wired once by ``Attention.parallelize``.
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
        # Bare head-wise sink parameter (fp32), matching the inference
        # reference and the kernels' ``[N1]`` sink contract.
        self.attn_sink = torch.nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))

        self.compressor = cfg.compressor.build() if cfg.compressor is not None else None
        self.indexer = cfg.indexer.build() if cfg.indexer is not None else None

        self.compressed_sparse_attention = cfg.compressed_sparse_attention.build()

    @property
    def inner_attention(self):
        """Read-only compatibility access to the wrapped attention module."""
        return self.compressed_sparse_attention.inner_attention

    def parallelize(self, parallel_dims) -> None:
        """Parallelize the attention, then wire the CP mesh on the
        attention's own token dispatcher (the ``RoutedExperts.parallelize``
        mirror).  The compressors' dispatchers are wired by their owners'
        ``parallelize`` through the framework's ``Module.parallelize``
        recursion."""
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(cp_mesh=parallel_dims.get_optional_mesh("cp"))

    def _golden_rope(self, x: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False):
        """Apply the reference complex-pair rotation through the float cache."""
        cache = self.rope._reshape_cache(x, positions)
        if isinstance(cache, tuple):
            # The workaround RoPE hands back an interleaved cos/sin pair.
            cos, sin = cache
            freqs = torch.complex(cos[..., ::2], sin[..., ::2])
        else:
            # The base RoPE already carries the cache as complex exponentials.
            freqs = cache
        if inverse:
            freqs = freqs.conj()
        real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        c, s = freqs.real, freqs.imag
        return torch.stack((real * c - imag * s, imag * c + real * s), dim=-1).flatten(-2).type_as(x)

    def forward(
        self,
        x,
        attention_masks,
        positions,
        *,
        v41_layer_id: int | None = None,
        v41_plan=None,
        v41_context=None,
    ):
        """The unified attention forward (CP and non-CP).

        The Q side and the swa projection run on the local stream; the
        token dispatcher's ops serve every consumer with no context-
        parallel special-casing: ``gather`` exchanges the post-RoPE
        ``swa_k`` rows (the window plan) into the packed ori stream, the
        compressors gather their own block rows internally, and ``select``
        packs the pooled streams into the padded containers.  The
        containers' all-gather is declarative — the core's
        ``ShardingConfig`` (``cp: S(1) -> R``) emits it at the core
        boundary.
        """
        window = attention_masks.window
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        if self.post_q_rms_norm:
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q = q.view(bsz, seqlen, -1, self.head_dim)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        golden = golden_enabled()
        q_rope = (
            self._golden_rope(q_rope, positions)
            if golden
            else self.rope(q_rope, positions=positions)
        )
        q = torch.cat([q_nope, q_rope], dim=-1)

        # The swa projection + RoPE run on the local rows (the sender's own
        # doc-relative positions — the attention's positions convention
        # resets per document); the window gather exchanges the post-RoPE
        # rows into the packed ori stream.
        swa_k = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(swa_k, [self.head_dim - rd, rd], dim=-1)
        kv_input = kv_rope.unsqueeze(2)
        kv_rope = (
            self._golden_rope(kv_input, positions.reshape(1, -1)).squeeze(2)
            if golden
            else self.rope(kv_input, positions=positions.reshape(1, -1)).squeeze(2)
        )
        swa_k = torch.cat([kv_nope, kv_rope], dim=-1)
        swa_k = self.token_dispatcher.gather(swa_k, window)

        cmp_k = None
        compressor_latent = None
        pooled = None
        idx_q = idx_k = idx_w = None

        shared_kv = None
        shared_topk = None
        if v41_plan is not None and v41_context is not None and v41_layer_id is not None:
            is_v41_source = v41_layer_id in v41_plan.kv_source_layers
            if not is_v41_source:
                shared_kv, _, shared_topk = v41_context.resolve(v41_plan, v41_layer_id)
            if shared_kv is not None:
                cmp_k = shared_kv[0]
            if is_v41_source:
                if self.compressor is None:
                    raise ValueError("V4.1 KV source requires a compressor")
                pooled, compressor_latent = self.compressor(
                    x,
                    attention_masks,
                    positions=positions,
                    return_pre_rope=True,
                )
                if self.compress_ratio == 1:
                    cmp_k = pooled
                else:
                    plan = attention_masks.plans[self.compress_ratio]
                    cmp_k = self.token_dispatcher.select(pooled, plan)

        has_v41_indexer = (
            v41_plan is not None
            and v41_layer_id is not None
            and v41_layer_id in v41_plan.index_source_layers
        )
        if self.indexer is not None and (self.compress_ratio > 1 or has_v41_indexer):
            index_source = (
                None
                if v41_plan is None or v41_layer_id is None
                else v41_plan.index_source_before(v41_layer_id)
            )
            if (
                v41_plan is not None
                and v41_layer_id is not None
                and v41_layer_id in v41_plan.kv_source_layers
            ):
                index_source = None
            shared_index_k = (
                None
                if v41_context is None or index_source is None
                else v41_context.index_keys.get(index_source)
            )
            indexer_kwargs = {
                "positions": positions,
                "attention_masks": attention_masks,
            }
            if shared_index_k is not None:
                indexer_kwargs["key_override"] = shared_index_k
            if compressor_latent is not None:
                indexer_kwargs["latent"] = compressor_latent
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(),
                qr.detach(),
                **indexer_kwargs,
            )
            # The indexer's outputs: idx_q / idx_w (local), idx_k (the
            # pooled stream — packed into the container).
            index_ratio = self.compress_ratio
            if index_source is not None and v41_plan is not None:
                index_ratio = v41_plan.ratios[index_source]
            index_plan = attention_masks.plans[index_ratio]
            if index_plan.gather_indices is not None and shared_index_k is None:
                idx_k = self.token_dispatcher.select(idx_k, index_plan)
            if (
                v41_plan is not None
                and v41_context is not None
                and v41_layer_id is not None
                and v41_layer_id in v41_plan.index_source_layers
            ):
                reference = getattr(attention_masks, "reference", None)
                dense_mask = getattr(attention_masks, "index_dense_masks", {}).get(index_ratio)
                if dense_mask is None and reference is not None:
                    ratio_layout = reference.ratios.get(index_ratio)
                    dense_mask = None if ratio_layout is None else ratio_layout.dense_mask
                if dense_mask is None:
                    dense_mask = build_index_dense_mask(attention_masks, index_ratio)
                if dense_mask is not None:
                    candidate_mask = None
                    if (
                        v41_plan is not None
                        and v41_context is not None
                        and v41_layer_id is not None
                        and v41_layer_id > v41_plan.candidate_source_layer
                    ):
                        candidate_mask = v41_context.candidates
                    shared_topk, index_scores = Indexer.select(
                        idx_q,
                        idx_k,
                        idx_w,
                        dense_mask,
                        getattr(self.compressed_sparse_attention.inner_attention, "index_topk", 512),
                        candidate_mask=candidate_mask,
                    )
                    if (
                        v41_plan is not None
                        and v41_context is not None
                        and v41_layer_id == v41_plan.candidate_source_layer
                    ):
                        from torchtitan_npu.models.deepseek_v4_1.attention import (
                            select_candidate_blocks,
                        )

                        compress_lens = dense_mask.squeeze(1).sum(dim=-1)
                        v41_context.put_candidates(
                            select_candidate_blocks(
                                index_scores,
                                compress_lens,
                                v41_plan.candidate_topk_blocks,
                                v41_plan.candidate_block_size,
                            )
                        )
                    v41_context.put_source(
                        v41_layer_id,
                        index_key=idx_k,
                        topk_indices=shared_topk,
                    )

        needs_local_kv = self.compress_ratio > 1 or (
            v41_plan is not None
            and v41_context is not None
            and v41_layer_id is not None
            and v41_layer_id in v41_plan.kv_source_layers
        )
        if needs_local_kv and cmp_k is None:
            assert self.compressor is not None, "compress_ratio > 1 requires the compressor submodule."
            pooled = self.compressor(
                x,
                attention_masks,
                positions=positions,
            )
            if self.compress_ratio == 1:
                cmp_k = pooled
            else:
                plan = attention_masks.plans[self.compress_ratio]
                cmp_k = self.token_dispatcher.select(pooled, plan)

        if (
            v41_plan is not None
            and v41_context is not None
            and v41_layer_id is not None
            and v41_layer_id in v41_plan.kv_source_layers
        ):
            source_kv = cmp_k if cmp_k is not None else swa_k
            v41_context.put_source(
                v41_layer_id,
                compressed_kv=(source_kv, source_kv),
            )
        if v41_plan is not None:
            self.compressed_sparse_attention.inner_attention._v41_topk_indices = shared_topk
            kv_source = v41_plan.kv_source_for(v41_layer_id)
            self.compressed_sparse_attention.inner_attention._v41_compress_ratio = (
                self.compress_ratio
                if kv_source is None
                else v41_plan.ratios[kv_source]
            )

        # Inner-attention positional contract: absent components are None.
        #   sink + swa_k always; + cmp_k when compress_ratio > 1;
        #   + idx_q/idx_k/idx_w when compress_ratio == 4 (indexer layer).
        o = self.compressed_sparse_attention(
            q,
            swa_k,
            cmp_k,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_w=idx_w,
            attn_sink=self.attn_sink,
            attention_masks=attention_masks,
        )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = (
            self._golden_rope(o_rope, positions, inverse=True)
            if golden
            else self.rope(o_rope, positions=positions, inverse=True)
        )
        o = torch.cat([o_nope, o_rope], dim=-1)

        # ``wo_a`` is a BatchedLinear over the head groups; group the heads
        # before the per-group matmul.
        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        if golden_enabled():
            wo_a = self.wo_a.weight.view(n_local_groups, self.wo_a.out_features, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        else:
            o = self.wo_a(o)
        o = o.reshape(bsz, seqlen, -1)
        return self.wo_b(o)
