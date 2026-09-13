from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# The plan/context classes below are built at class-creation time and shared
# across the V4.1 model, so these imports have to stay at the top.


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
        """Drop tensors from the previous forward before starting a new one."""
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
        """Select and store the level-one candidate block mask.

        Kept inside the V4.1 context so the V4 decoder only needs the seam
        (``put_candidates`` / ``candidates``) and never imports this package.
        """
        self.put_candidates(
            select_candidate_blocks(index_scores, compress_lens, topk_blocks, block_size)
        )

    def resolve(self, plan: V41CompressionSpec, layer_id: int) -> tuple[Any | None, Any | None, Any | None]:
        kv_source = plan.kv_source_for(layer_id)
        index_source = plan.index_source_for(layer_id)
        return (
            None if kv_source is None else self.compressed_kv.get(kv_source),
            None if index_source is None else self.index_keys.get(index_source),
            None if index_source is None else self.topk_indices.get(index_source),
        )


# existing definitions are kept above this implementation


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
            # Store tensors, not detached copies, so source gradients flow to
            # consumers in the reference implementation.
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
