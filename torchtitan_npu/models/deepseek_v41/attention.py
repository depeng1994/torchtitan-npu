from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan_npu.models.deepseek_v4.attention import Attention as DeepSeekV4Attention
from torchtitan_npu.models.deepseek_v4.attention import LongRangeContext
from torchtitan_npu.models.deepseek_v4.compressor import Indexer
from torchtitan_npu.models.deepseek_v4.metadata import build_index_dense_mask


@dataclass(frozen=True, slots=True)
class V41CompressionSpec:
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
    """Per-forward shared state for V4.1 compression and index stages."""

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


class DeepSeekV41Attention(DeepSeekV4Attention):
    """V4.1 attention specialization that owns CSA2 source/reuse/reindex semantics."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Attention.Config):
        pass

    def _build_v41_long_range_context(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attention_masks,
        positions: torch.Tensor,
        *,
        layer_id: int,
        plan: V41CompressionSpec,
        context: V41AttentionContext,
    ) -> LongRangeContext:
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
                cmp_k = self.token_dispatcher.select(pooled, attention_masks.plans[self.compress_ratio])

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
                idx_k = self.token_dispatcher.select(idx_k, index_plan)

            reference = getattr(attention_masks, "reference", None)
            dense_mask = getattr(attention_masks, "index_dense_masks", {}).get(index_ratio)
            if dense_mask is None and reference is not None:
                ratio_layout = reference.ratios.get(index_ratio)
                dense_mask = None if ratio_layout is None else ratio_layout.dense_mask
            if dense_mask is None:
                dense_mask = build_index_dense_mask(attention_masks, index_ratio)

            candidate_mask = context.candidates if layer_id > plan.candidate_source_layer else None
            shared_topk, index_scores = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                dense_mask,
                getattr(self.compressed_sparse_attention.inner_attention, "index_topk", 512),
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

        if is_kv_source and cmp_k is None:
            if self.compressor is None:
                raise ValueError("V4.1 KV source requires a compressor")
            pooled = self.compressor(x, attention_masks, positions=positions)
            cmp_k = (
                pooled
                if self.compress_ratio == 1
                else self.token_dispatcher.select(pooled, attention_masks.plans[self.compress_ratio])
            )

        if is_kv_source:
            context.put_source(layer_id, compressed_kv=(cmp_k, cmp_k))

        kv_source = plan.kv_source_for(layer_id)
        active_ratio = self.compress_ratio if kv_source is None else plan.ratios[kv_source]
        return LongRangeContext(
            compressed_kv=cmp_k,
            index_q=idx_q,
            index_k=idx_k,
            index_weight=idx_w,
            sparse_indices=shared_topk,
            compress_ratio=active_ratio,
        )

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
        long_range = self._build_v41_long_range_context(
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


class V41GoldenAttention(nn.Module):
    """Training-safe reference attention for the V4.1 source/consumer contract."""

    def __init__(self, *, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        layer_id: int,
        plan: V41CompressionSpec,
        context: V41AttentionContext,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        ratio = plan.ratios[layer_id]
        source_kv, _, _ = context.resolve(plan, layer_id)
        if ratio != 0 and source_kv is not None:
            source_key, source_value = source_kv
            key, value = source_key, source_value
        elif layer_id in plan.kv_source_layers:
            context.put_source(layer_id, compressed_kv=(key, value))

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal and attn_mask is None,
        )


class V41GoldenAttentionModule(nn.Module):
    """Pure Torch reference decoder attention for V4.1 integration tests."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.golden_attention = V41GoldenAttention()

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        layer_id: int,
        plan: V41CompressionSpec,
        context: V41AttentionContext,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden).view(shape).transpose(1, 2)
        key = self.k_proj(hidden).view(shape).transpose(1, 2)
        value = self.v_proj(hidden).view(shape).transpose(1, 2)
        output = self.golden_attention(
            query,
            key,
            value,
            layer_id=layer_id,
            plan=plan,
            context=context,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


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
