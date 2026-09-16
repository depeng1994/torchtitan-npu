# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 attention: the CSA2 source/reuse/reindex orchestration.

Self-contained (no V4 imports).  The projections and the golden rope are
the inference-reference arithmetic; the sparse core is
:class:`~torchtitan_npu.models.deepseek_v41.sparse_attention.V41SparseAttention`
(the golden per-document DSA).  ``USE_GOLDEN=0`` is rejected at the entry
points, so no golden/non-golden branch exists here.

The compression topology policy lives in :class:`V41CompressionSpec` and
the per-forward shared state in :class:`V41AttentionContext` (reset by
the model at every forward; never carried across batches).
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor, Indexer, pack_container
from .rope import V41RoPERotation
from .sparse_attention import V41SparseAttention


@dataclass(frozen=True, slots=True)
class V41CompressionSpec:
    """The explicit source/consumer dependency plan for one crop.

    ``ratios`` is positionally indexed: ``layer_ids`` must be consecutive
    and 0-based.
    """

    layer_ids: tuple[int, ...]
    ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    def __post_init__(self) -> None:
        if len(self.layer_ids) != len(self.ratios):
            raise ValueError("layer_ids and ratios must have the same length")
        if self.layer_ids != tuple(range(len(self.layer_ids))):
            raise ValueError("layer_ids must be consecutive and 0-based; ratios are positionally indexed")
        layers = set(self.layer_ids)
        if not set(self.kv_source_layers) <= layers:
            raise ValueError("KV source layer is outside the crop")
        if not set(self.index_source_layers) <= layers:
            raise ValueError("index source layer is outside the crop")
        if self.candidate_source_layer not in layers:
            raise ValueError("candidate source layer is outside the crop")
        if self.candidate_source_layer not in self.index_source_layers:
            raise ValueError("candidate source layer must also be an index source")
        if self.candidate_topk_blocks <= 0 or self.candidate_block_size <= 0:
            raise ValueError("candidate block parameters must be positive")

    def kv_source_for(self, layer_id: int) -> int | None:
        candidates = [source for source in self.kv_source_layers if source <= layer_id]
        return max(candidates) if candidates else None

    def index_source_for(self, layer_id: int) -> int | None:
        candidates = [source for source in self.index_source_layers if source <= layer_id]
        return max(candidates) if candidates else None

    def index_source_before(self, layer_id: int) -> int | None:
        candidates = [source for source in self.index_source_layers if source < layer_id]
        return max(candidates) if candidates else None


def build_v41_compression_spec(
    *,
    layer_ids: tuple[int, ...],
    ratios: tuple[int, ...],
    kv_source_layers: tuple[int, ...],
    index_source_layers: tuple[int, ...],
    candidate_source_layer: int,
    candidate_topk_blocks: int = 2048,
    candidate_block_size: int = 8,
) -> V41CompressionSpec:
    """Build the explicit source/consumer dependency plan for one crop."""
    return V41CompressionSpec(
        layer_ids=layer_ids,
        ratios=ratios,
        kv_source_layers=kv_source_layers,
        index_source_layers=index_source_layers,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


@dataclass(slots=True)
class V41AttentionContext:
    """Per-forward shared state for V4.1 compression and index stages.

    The model resets it before every forward; layers publish their
    compressed KV / index keys / materialized top-k and consumers resolve
    the most recent source at or below their layer id.
    """

    compressed_kv: dict[int, Any]
    index_keys: dict[int, Any]
    topk_indices: dict[int, Any]
    candidates: Any | None = None

    @classmethod
    def empty(cls) -> "V41AttentionContext":
        return cls(compressed_kv={}, index_keys={}, topk_indices={})

    def reset(self) -> None:
        self.compressed_kv.clear()
        self.index_keys.clear()
        self.topk_indices.clear()
        self.candidates = None

    def put_source(
        self,
        layer_id: int,
        *,
        compressed_kv: Any | None = None,
        index_key: Any | None = None,
        topk_indices: Any | None = None,
    ) -> None:
        if compressed_kv is not None:
            self.compressed_kv[layer_id] = compressed_kv
        if index_key is not None:
            self.index_keys[layer_id] = index_key
        if topk_indices is not None:
            self.topk_indices[layer_id] = topk_indices

    def put_candidates(self, candidates: Any) -> None:
        self.candidates = candidates

    def build_candidates(
        self,
        index_scores: torch.Tensor,
        compress_lens: torch.Tensor | int,
        topk_blocks: int,
        block_size: int,
    ) -> None:
        self.put_candidates(select_candidate_blocks(index_scores, compress_lens, topk_blocks, block_size))

    def resolve(self, plan: V41CompressionSpec, layer_id: int) -> tuple[Any | None, Any | None, Any | None]:
        kv_source = plan.kv_source_for(layer_id)
        index_source = plan.index_source_for(layer_id)
        return (
            None if kv_source is None else self.compressed_kv.get(kv_source),
            None if index_source is None else self.index_keys.get(index_source),
            None if index_source is None else self.topk_indices.get(index_source),
        )


def select_candidate_blocks(
    index_scores: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Reference level-one candidate block mask for V4.1 index selection."""
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("topk_blocks and block_size must be positive")
    width = index_scores.size(-1)
    blocks = F.pad(index_scores, (0, (-width) % block_size), value=-torch.inf)
    blocks = blocks.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last = (compress_lens - 1) // block_size
    block_ids = torch.arange(blocks.size(-1), device=index_scores.device)
    reachable_last = block_ids == last.unsqueeze(-1) if torch.is_tensor(last) else block_ids == last
    blocks = blocks.masked_fill(reachable_last, torch.inf)
    _, selected = blocks.topk(min(topk_blocks, blocks.size(-1)), dim=-1)
    valid = blocks.gather(-1, selected) > -torch.inf
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, selected, valid)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class CompressedSparseAttention(Module):
    """The V4.1 sparse-attention boundary (the golden DSA core).

    V4.1 has no LightningIndexer wrapper, so this is the direct holder of
    the inner attention core (kept as a Module child for placement).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_attention: V41SparseAttention.Config

    def __init__(self, config: Config):
        super().__init__()
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


class DeepSeekV41Attention(BaseAttention):
    """The V4.1 attention: projections, golden rope, CSA2 long-range inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        inner_attention: Module.Config
        rope: RoPE.Config
        rotary: V41RoPERotation.Config = field(default_factory=V41RoPERotation.Config)
        head_dim: int
        rope_head_dim: int
        q_lora_rank: int
        n_groups: int
        compress_ratio: int
        norm_eps: float

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

    def __init__(self, config: "DeepSeekV41Attention.Config"):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.rope = cfg.rope.build()
        self.rotary = cfg.rotary.build()
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
        """Read-only access to the wrapped attention module."""
        return self.compressed_sparse_attention.inner_attention

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False):
        cache = self.rope._reshape_cache(x, positions)
        if isinstance(cache, tuple):
            cos, sin = cache
        else:
            cos, sin = cache.real.repeat_interleave(2, dim=-1), cache.imag.repeat_interleave(2, dim=-1)
        return self.rotary(x, cos, sin, inverse=inverse)

    def _project_q(self, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, -1, self.head_dim)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self._apply_rope(q_rope, positions)
        return qr, torch.cat([q_nope, q_rope], dim=-1)

    def _project_window_kv(self, x, attention_masks, positions) -> torch.Tensor:
        rd = self.rope_head_dim
        swa_k = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(swa_k, [self.head_dim - rd, rd], dim=-1)
        kv_input = kv_rope.unsqueeze(2)
        kv_rope = self._apply_rope(kv_input, positions.reshape(1, -1)).squeeze(2)
        swa_k = torch.cat([kv_nope, kv_rope], dim=-1)
        return swa_k

    def _build_long_range_context(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attention_masks,
        positions: torch.Tensor,
        *,
        layer_id: int,
        plan: V41CompressionSpec,
        context: V41AttentionContext,
    ):
        """Build the layer's materialized long-range inputs (CSA2 policy)."""
        cmp_k = None
        compressor_latent = None
        idx_q = idx_k = idx_w = None
        shared_topk = None

        is_kv_source = layer_id in plan.kv_source_layers
        if not is_kv_source:
            shared_kv, _, shared_topk = context.resolve(plan, layer_id)
            if shared_kv is not None:
                cmp_k = shared_kv[0]

        if is_kv_source:
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
                cmp_k = pack_container(pooled, attention_masks.plans[self.compress_ratio])

        if self.indexer is not None and layer_id in plan.index_source_layers:
            index_source = None if is_kv_source else plan.index_source_before(layer_id)
            shared_index_k = None if index_source is None else context.index_keys.get(index_source)
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

            index_ratio = self.compress_ratio if index_source is None else plan.ratios[index_source]
            index_plan = attention_masks.plans[index_ratio]
            if index_plan.gather_indices is not None and shared_index_k is None:
                idx_k = pack_container(idx_k, index_plan)

            # The reference tier owns the materialized index layouts; V4.1
            # explicitly materializes ratio 1.
            ratio_layout = attention_masks.reference.ratios.get(index_ratio)
            dense_mask = None if ratio_layout is None else ratio_layout.dense_mask
            if dense_mask is None:
                raise ValueError(f"V4.1 requires a materialized reference index mask for ratio={index_ratio}")

            candidate_mask = context.candidates if layer_id > plan.candidate_source_layer else None
            shared_topk, index_scores = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                dense_mask,
                # index_topk is a required field on the sparse core and is set
                # unconditionally in its __init__, so direct access fails
                # loudly on a misconfigured module.
                self.compressed_sparse_attention.inner_attention.index_topk,
                candidate_mask=candidate_mask,
            )
            if layer_id == plan.candidate_source_layer:
                compress_lens = dense_mask.squeeze(1).sum(dim=-1)
                context.build_candidates(
                    index_scores,
                    compress_lens,
                    plan.candidate_topk_blocks,
                    plan.candidate_block_size,
                )
            context.put_source(
                layer_id,
                index_key=idx_k,
                topk_indices=shared_topk,
            )

        if is_kv_source:
            context.put_source(layer_id, compressed_kv=(cmp_k, cmp_k))

        kv_source = plan.kv_source_for(layer_id)
        active_ratio = self.compress_ratio if kv_source is None else plan.ratios[kv_source]
        return {
            "compressed_kv": cmp_k,
            "index_q": idx_q,
            "index_k": idx_k,
            "index_weight": idx_w,
            "sparse_indices": shared_topk,
            "compress_ratio": active_ratio,
        }

    def _apply_sparse_attention(self, q, swa_k, long_range, attention_masks) -> torch.Tensor:
        return self.compressed_sparse_attention(
            q,
            swa_k,
            long_range["compressed_kv"],
            idx_q=long_range["index_q"],
            idx_k=long_range["index_k"],
            idx_w=long_range["index_weight"],
            sparse_indices=long_range["sparse_indices"],
            compress_ratio=long_range["compress_ratio"],
            attn_sink=self.attn_sink,
            attention_masks=attention_masks,
        )

    def _project_output(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = o.shape[:2]
        rd = self.rope_head_dim
        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self._apply_rope(o_rope, positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)
        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        wo_a = self.wo_a.weight.view(n_local_groups, self.wo_a.out_features, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        o = o.reshape(bsz, seqlen, -1)
        return self.wo_b(o)

    def forward(
        self,
        x,
        attention_masks,
        positions,
        *,
        layer_id: int,
        plan: V41CompressionSpec,
        context: V41AttentionContext,
    ):
        qr, q = self._project_q(x, positions)
        swa_k = self._project_window_kv(x, attention_masks, positions)
        long_range = self._build_long_range_context(
            x,
            qr,
            attention_masks,
            positions,
            layer_id=layer_id,
            plan=plan,
            context=context,
        )
        o = self._apply_sparse_attention(q, swa_k, long_range, attention_masks)
        return self._project_output(o, positions)
