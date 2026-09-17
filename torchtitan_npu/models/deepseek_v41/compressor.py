# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 key compression and index projection (golden reference arithmetic).

The V4.1 compressor covers exactly the two KV-source shapes of the topology:

- **ratio 1** — the true shared/global-KV container: the per-token latent
  (``wkv`` + norm) with the compressed-domain RoPE applied per token;
- **ratio 2** — the strict golden pooling: the projected kv/score rows are
  softmax-pooled over each 2-token block in FP32, golden-RMS-normed, and
  rotated at the block-start position.

The golden arithmetic (adjacent-pair complex rotation, FP32 RMS norm,
softmax pooling) matches the inference reference op-for-op; the dtype cast
nodes and their order are pinned by the frozen training trajectory.

The indexer owns the query/weight projections.  Source-key indexers (the
KV-source layers 2/8/14/20) consume the compressor's pre-RoPE latent
through their own ``wk``/``k_norm``; external-key indexers (24/28/32/36)
own no key projection at all and must receive ``key_override`` from the
attention context.
"""

from dataclasses import dataclass

import torch
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .metadata import CompressedBlockLayout


def _golden_complex_rope(rope, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Reference adjacent-pair complex rotation using the expanded float cache."""
    cos, sin = rope._reshape_cache(x, positions.reshape(1, -1))
    freqs = torch.complex(cos[..., ::2], sin[..., ::2])
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs).flatten(-2).type_as(x)


def _golden_rms_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)  # pyrefly: ignore [unsupported-operation]
    return (norm.weight * x).to(dtype)


def pack_container(x: torch.Tensor, plan: CompressedBlockLayout) -> torch.Tensor:
    """Pack the pooled stream into the uniform container grid ``[1, out_width, D]``.

    A CP1 plan keeps every complete block row, so this is the pooled stream
    zero-padded to the container width ``seq_len // ratio``.
    """
    x2 = x.flatten(0, 1) if x.ndim > 2 else x
    if plan.out_width is None:
        raise ValueError("pack_container requires the container width")
    pad = x.new_zeros((plan.out_width - x2.shape[0], x.shape[-1]))
    return torch.cat([x2, pad], dim=0).unsqueeze(0)


class Compressor(Module):
    """Document-packed key compression for the V4.1 KV-source layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wkv: Linear.Config
        wgate: Linear.Config | None
        norm: RMSNorm.Config
        head_dim: int
        rope_head_dim: int
        compress_ratio: int

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.rope = cfg.rope.build()
        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build() if cfg.wgate is not None else None
        self.norm = cfg.norm.build()

    def _apply_rope(self, kv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply the compressed-domain RoPE to a batched stream ``[B, N, D]``."""
        if kv.ndim != 3:
            raise ValueError(f"V4.1 compressor rope expects [B, N, D], got {tuple(kv.shape)}")
        rd = self.rope_head_dim
        nope_dim = self.head_dim - rd
        kv_nope, kv_rope = torch.split(kv, [nope_dim, rd], dim=-1)
        rotated = _golden_complex_rope(
            self.rope,
            kv_rope.unsqueeze(2),
            positions,
        ).squeeze(2)
        return torch.cat([kv_nope, rotated], dim=-1)

    def forward(
        self,
        x,
        attention_masks,
        *,
        positions: torch.Tensor | None = None,
        return_pre_rope: bool = False,
    ):
        """Compress the local stream; ``attention_masks`` is accepted for
        call-site symmetry but the golden ratios derive their layouts from
        the stream geometry alone.

        Returns the pooled key stream ``[B, N, head_dim]`` plus the
        pre-RoPE latent when ``return_pre_rope`` (the source-key indexer's
        key input).
        """
        if self.compress_ratio == 1:
            latent = self.norm(self.wkv(x))
            output = latent if positions is None else self._apply_rope(latent, positions)
            return (output, latent) if return_pre_rope else output
        if self.compress_ratio != 2:
            raise NotImplementedError("the V4.1 golden compressor covers ratio=1 and ratio=2 only")
        batch, seqlen, _ = x.shape
        ratio = self.compress_ratio
        dtype = x.dtype
        rows = seqlen // ratio
        projected_kv = self.wkv(x)
        projected_score = self.wgate(x)  # pyrefly: ignore [not-callable]
        kv = projected_kv[:, : rows * ratio].reshape(batch, rows, ratio, -1).float()
        score = projected_score[:, : rows * ratio].reshape(batch, rows, ratio, -1).float()
        latent = (kv * score.softmax(dim=2)).sum(dim=2)
        latent = _golden_rms_norm(self.norm, latent.to(dtype))
        block_positions = torch.arange(rows, device=x.device, dtype=torch.long) * ratio
        output = self._apply_rope(latent, block_positions)
        return (output, latent) if return_pre_rope else output


class Indexer(Module):
    """V4.1 index projection: queries, keys, and per-head weights.

    Key modes: a source-key indexer owns ``wk``/``k_norm`` and consumes the
    KV-source compressor's latent; an external-key indexer owns no key
    projection and requires ``key_override`` (the cross-layer reused key).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wq_b: Linear.Config
        weights_proj: Linear.Config
        num_index_heads: int
        index_head_dim: int
        rope_head_dim: int
        compress_ratio: int = 1
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.rope = cfg.rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        if (cfg.wk is None) != (cfg.k_norm is None):
            raise ValueError("Indexer source-key projection requires both wk and k_norm")
        self.wk = cfg.wk.build() if cfg.wk is not None else None
        self.k_norm = cfg.k_norm.build() if cfg.k_norm is not None else None

    def forward(
        self,
        x,
        qr,
        *,
        positions,
        attention_masks,
        key_override=None,
        latent=None,
    ):
        """Project the indexer queries, keys, and per-head weights.

        Returns:
            idx_q: ``[B, L, num_index_heads, index_head_dim]`` (RoPE applied).
            idx_k: keys in the container grid ``[B, L // ratio, index_head_dim]``.
            idx_w: per-head weights ``[B, L, num_index_heads]``.
        """
        bsz, seqlen, _ = qr.size()
        rd = self.rope_head_dim
        idx_q = self.wq_b(qr)
        idx_q = idx_q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(idx_q, [self.head_dim - rd, rd], dim=-1)
        q_rope = _golden_complex_rope(self.rope, q_rope, positions)
        idx_q = torch.cat([q_nope, q_rope], dim=-1)
        if key_override is None:
            if self.wk is None:
                raise ValueError("V4.1 external-key indexer requires key_override")
            if latent is None:
                raise ValueError("source-key indexer requires the compressor latent")
            idx_k = self.k_norm(self.wk(latent))  # pyrefly: ignore [not-callable]
            ratio = self.compress_ratio
            if ratio > 1:
                plan = attention_masks.plans.get(ratio)
                if plan is None or plan.block_positions is None:
                    raise ValueError(f"missing compressed positions for index ratio={ratio}")
                idx_k = self._apply_key_rope(idx_k, plan.block_positions)
            else:
                idx_k = self._apply_key_rope(idx_k, positions)
        else:
            idx_k = key_override
        idx_w = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return idx_q, idx_k, idx_w

    def _apply_key_rope(self, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply the index-key RoPE to a batched key stream ``[B, N, D]``."""
        rd = self.rope_head_dim
        if rd == 0:
            return key
        key_nope, key_rope = torch.split(key, [self.head_dim - rd, rd], dim=-1)
        rotated = _golden_complex_rope(self.rope, key_rope.unsqueeze(2), positions).reshape_as(key_rope)
        return torch.cat([key_nope, rotated], dim=-1)

    @staticmethod
    def select(
        idx_q,
        idx_k,
        idx_w,
        dense_mask: torch.Tensor,
        topk: int,
        candidate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the top-k compressed container slots per query.

        Scores are masked by the precomputed attendability dense mask (same
        document and causally reachable), so all returned indices are
        attendable.  Unused slots hold ``-1``.  The sorted-then-stabilized
        top-k order is part of the frozen reference digests.

        Returns:
            topk_indices: ``[B, L, K]`` container-grid slots, ``K =
                min(topk, S // ratio)``.
            index_scores: ``[B, L, S // ratio]`` masked indexer scores.
        """
        q, k, weights = idx_q, idx_k, idx_w
        index_score = torch.einsum("bshd,btd->bsht", q, k)
        index_score = index_score.relu_() * weights.unsqueeze(-1)
        index_score = index_score.sum(dim=2)

        k = min(topk, idx_k.shape[1])
        index_score = index_score.where(dense_mask.squeeze(1), float("-inf"))
        if candidate_mask is not None:
            if candidate_mask.ndim == 4:
                candidate_mask = candidate_mask.squeeze(1)
            if candidate_mask.shape != index_score.shape:
                raise ValueError(
                    "candidate mask shape must match index score shape: "
                    f"{tuple(candidate_mask.shape)} vs {tuple(index_score.shape)}"
                )
            index_score = index_score.where(candidate_mask, float("-inf"))
        selected = index_score.topk(k, dim=-1, sorted=False).indices
        # aten sort does not survive dynamo fake-eval under the spmd patch
        # stack; the selected slot ids are distinct, so a full-width
        # topk(largest=False, sorted=True) is the identical ascending sort.
        indices = selected.topk(k, dim=-1, largest=False, sorted=True).values
        valid = dense_mask.squeeze(1).gather(-1, indices)
        return indices.where(valid, -1), index_score
