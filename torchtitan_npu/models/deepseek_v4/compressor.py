# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from functools import cache
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.models.common.metadata_extension import LightningIndexerKernelConfig

from torchtitan_npu.models.deepseek_v4.golden import golden_enabled

from .metadata import CompressedKernelContract
from .token_dispatcher import CPTokenDispatcher


def _golden_complex_rope(rope, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Reference adjacent-pair complex rotation using the expanded float cache."""
    cos, sin = rope._reshape_cache(x, positions.reshape(1, -1))
    freqs = torch.complex(cos[..., ::2], sin[..., ::2])
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs).flatten(-2).type_as(x)


def _golden_rms_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
    return (norm.weight * x).to(dtype)


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class Compressor(Module):
    """Document-packed key compression.

    Consumes the unified per-ratio compressor contract from the metadata's
    ``plans[ratio]`` — ``gather_indices`` / ``block_positions`` /
    ``first_indices`` — and returns the pooled key stream ``[n_blocks, D]``.
    The contract is provided identically under context parallel and
    without: the projection (``wkv`` / ``wgate``) is per-token and runs on
    the local stream, and the compressor's own token dispatcher gathers
    the plan-block rows — without context parallel a plain local gather
    (the doc-major ``gather_indices`` over the local stream), under
    context parallel the remote gather + permute (the exchange plus the
    pooled-order ``gather_indices``).  The container packing and the CP
    strip are the call sites' concern.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wkv: Linear.Config
        wgate: Linear.Config | None
        norm: RMSNorm.Config
        head_dim: int
        rope_head_dim: int
        compress_ratio: int
        use_ape: bool = True
        # The CP token dispatcher (the RoutedExperts mirror): owned by the
        # compressor, wired once by ``Compressor.parallelize``.
        token_dispatcher: CPTokenDispatcher.Config = field(default_factory=CPTokenDispatcher.Config)

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.use_ape = cfg.use_ape
        self.overlap = cfg.compress_ratio == 4
        self.rope = cfg.rope.build()
        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build() if cfg.wgate is not None else None
        self.norm = cfg.norm.build()
        self.token_dispatcher = cfg.token_dispatcher.build()
        # ``ape`` is a plain score bias on the compression block, not a
        # projection; own it directly like the upstream implementation.  Its
        # param_init is declared on the Config (registry-side, ``_APE_INIT``).
        self.ape = (
            nn.Parameter(torch.empty(cfg.compress_ratio, self.wkv.out_features, dtype=torch.float32))
            if cfg.compress_ratio > 1 and cfg.use_ape
            else None
        )

    def parallelize(self, parallel_dims) -> None:
        """Wire the compressor's own CP dispatcher (invoked automatically
        by the framework's ``Module.parallelize`` recursion — the
        compressor is a Module child of the Attention / Indexer)."""
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(cp_mesh=parallel_dims.get_optional_mesh("cp"))

    @staticmethod
    def _overlap_transform(
        state: torch.Tensor,
        first_indices: torch.Tensor,
        *,
        value: float,
    ) -> torch.Tensor:
        """The C4A overlap window of one projected state.

        The window's left half is the previous plan block's a-series, rolled
        one block; the right half is the current block's b-series (the
        paper's C^a/C^b halves).  The first-block rows of the left half are
        filled with ``value``: the score state passes ``-inf`` (exactly zero
        softmax weight — which also annihilates the rolled-in block-0 rows),
        the kv state passes ``0`` (defensive, like the reference
        implementations: its masked rows are multiplied by those exact-zero
        weights anyway).  (The CP borrow exchange is a different mechanism —
        the token-level augmented stream.)
        """
        assert state.size(-1) % 2 == 0, "the overlap window needs the 2*head_dim split"
        head_dim = state.size(-1) // 2
        n_blocks = state.size(0)
        # Functional roll/fill (gather + ``torch.where``) instead of
        # ``torch.roll`` + in-place scatter: the in-place write is dropped by
        # aot_autograd recompute and its data-dependent guard trips make_fx.
        prev_idx = (torch.arange(n_blocks, device=state.device) - 1).clamp_min(0)
        state_a = state[prev_idx, :, :head_dim]
        first_mask = torch.index_fill(
            torch.zeros(n_blocks, dtype=torch.bool, device=state.device),
            0,
            first_indices,
            True,
        )
        state_a = torch.where(first_mask.view(-1, 1, 1), state.new_full((1,), value), state_a)
        state_b = state[:, :, head_dim:]
        return torch.cat([state_a, state_b], dim=1)

    def _apply_rope(self, kv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply compressed-key RoPE to either a packed or batched stream."""
        rd = self.rope_head_dim
        nope_dim = self.head_dim - rd
        kv_nope, kv_rope = torch.split(kv, [nope_dim, rd], dim=-1)
        golden = golden_enabled()
        if kv.ndim == 2:
            if golden:
                rotated = (
                    _golden_complex_rope(
                        self.rope,
                        kv_rope.unsqueeze(0).unsqueeze(2),
                        positions,
                    )
                    .squeeze(0)
                    .squeeze(1)
                )
            else:
                rotated = (
                    self.rope(
                        kv_rope.unsqueeze(0).unsqueeze(2),
                        positions=positions.reshape(1, -1),
                    )
                    .squeeze(0)
                    .squeeze(1)
                )
        elif golden:
            rotated = _golden_complex_rope(
                self.rope,
                kv_rope.unsqueeze(2),
                positions,
            ).squeeze(2)
        else:
            rotated = self.rope(
                kv_rope.unsqueeze(2),
                positions=positions.reshape(kv.shape[0], kv.shape[1]),
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
        """Project kv/score locally, gather the plan-block rows through the
        dispatcher, pool, and RoPE.

        Returns the pooled key stream ``[n_blocks, head_dim]`` (all plan
        blocks, borrow-source blocks included — the strip and the container
        packing happen at the call sites).  ``first_indices`` are the
        doc/segment-first block positions whose borrowed (previous-block)
        rows are filled by ``_overlap_transform`` (``-inf`` on the score —
        exactly zero softmax weight — and ``0`` on the kv).  The projection
        is per-token, so it commutes with the exchange (bitwise): without
        context parallel the dispatcher's gather is a plain local gather,
        under context parallel it exchanges the projected rows of the
        plan's ``[A, B)`` block region.
        """
        if self.compress_ratio == 1:
            latent = self.norm(self.wkv(x))
            output = latent if positions is None else self._apply_rope(latent, positions)
            return (output, latent) if return_pre_rope else output
        if golden_enabled():
            if self.compress_ratio != 2:
                raise NotImplementedError("strict golden compressor currently covers ratio=2")
            batch, seqlen, _ = x.shape
            ratio = self.compress_ratio
            dtype = x.dtype
            rows = seqlen // ratio
            projected_kv = self.wkv(x)
            projected_score = self.wgate(x)
            kv = projected_kv[:, : rows * ratio].reshape(batch, rows, ratio, -1).float()
            score = projected_score[:, : rows * ratio].reshape(batch, rows, ratio, -1).float()
            latent = (kv * score.softmax(dim=2)).sum(dim=2)
            latent = _golden_rms_norm(self.norm, latent.to(dtype))
            block_positions = torch.arange(rows, device=x.device, dtype=torch.long) * ratio
            output = self._apply_rope(latent, block_positions)
            return (output, latent) if return_pre_rope else output
        if not isinstance(attention_masks, CompressedKernelContract):
            raise TypeError(
                "DSV4 compression requires a CompressedKernelContract (the "
                "model-dir CompressedVarlenMetadata or the NPU slim type), "
                f"got {type(attention_masks)}."
            )
        ratio = self.compress_ratio
        plan = attention_masks.plans.get(ratio)
        if plan is None or plan.gather_indices is None:
            raise ValueError(f"No compressor plan for ratio={ratio}")

        # -- project the local stream (BF16 weights, FP32 compute via
        #    autocast); the dispatcher's gather then collects the
        #    plan-block rows --
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv_rows = self.token_dispatcher.gather(self.wkv(x), plan).flatten(0, 1)
            assert self.wgate is not None
            score_rows = self.token_dispatcher.gather(self.wgate(x), plan).flatten(0, 1)
        # Block count from the plan's (dynamic) gather_indices, not the
        # gathered rows' runtime shape, which would specialize the trace.
        # An empty plan flows through reshape(0, ...) without a numel() guard.
        n_blocks = plan.gather_indices.numel() // ratio
        # Explicit last dim: a ``-1`` fold trips a data-dependent guard.
        kv = kv_rows.reshape(n_blocks, ratio, kv_rows.size(-1))
        score = score_rows.reshape(n_blocks, ratio, score_rows.size(-1))
        if self.ape is not None:
            score = score + self.ape
        first_indices = plan.first_indices
        block_positions = plan.block_positions
        assert first_indices is not None and block_positions is not None, (
            "the compressor contract requires first_indices and block_positions"
        )
        # -- overlap (ratio=4 only) --
        if self.overlap:
            score = self._overlap_transform(score, first_indices, value=float("-inf"))
            kv = self._overlap_transform(kv, first_indices, value=0.0)

        # -- softmax pool + norm + RoPE --
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        latent = self.norm(kv.to(x.dtype))
        output = self._apply_rope(latent, block_positions)
        return (output, latent) if return_pre_rope else output


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wq_b: Linear.Config
        weights_proj: Linear.Config
        num_index_heads: int
        index_head_dim: int
        rope_head_dim: int
        compress_ratio: int = 1
        compressor: "Compressor.Config | None" = None
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None
        rotation: Literal["hadamard", "none"] = "hadamard"

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.rotation = cfg.rotation
        self.rope = cfg.rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        if cfg.wk is not None or cfg.k_norm is not None:
            if cfg.wk is None or cfg.k_norm is None:
                raise ValueError("Indexer source-key projection requires both wk and k_norm")
            self.wk = cfg.wk.build()
            self.k_norm = cfg.k_norm.build()
            self.compressor = None
        elif cfg.compressor is not None:
            self.wk = None
            self.k_norm = None
            self.compressor = cfg.compressor.build()
        else:
            self.wk = None
            self.k_norm = None
            self.compressor = None

    def _rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation == "none":
            return x
        d = x.size(-1)
        H = _hadamard(d, dtype=x.dtype, device=x.device)
        return F.linear(x, H) * (d**-0.5)

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
        """Project raw indexer queries, keys, and per-head weights.

        ``x`` is the caller's **local** stream (never the augmented
        stream): ``idx_q`` / ``idx_w`` derive from the local ``qr`` and
        the local rows, and the indexer's compressor borrows its own
        block rows internally.

        Returns:
            idx_q: Indexer queries ``[B, L, num_index_heads, index_head_dim]``
                with RoPE applied and Hadamard-rotated.
            idx_k: Indexer compressed keys in the container grid
                ``[B, L // ratio, index_head_dim]``, Hadamard-rotated.
            idx_w: Per-head indexer weights ``[B, L, num_index_heads]``
                (the local rows).
        """
        bsz, seqlen, _ = qr.size()
        rd = self.rope_head_dim
        idx_q = self.wq_b(qr)
        idx_q = idx_q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(idx_q, [self.head_dim - rd, rd], dim=-1)
        q_rope = (
            _golden_complex_rope(self.rope, q_rope, positions)
            if golden_enabled()
            else self.rope(q_rope, positions=positions)
        )
        idx_q = torch.cat([q_nope, q_rope], dim=-1)
        idx_q = self._rotate_activation(idx_q)
        if key_override is None:
            if self.wk is not None:
                if latent is None:
                    raise ValueError("source-key indexer requires the compressor latent")
                idx_k = self.k_norm(self.wk(latent))
                ratio = self.compress_ratio
                if ratio > 1:
                    plan = attention_masks.plans.get(ratio)
                    if plan is None or plan.block_positions is None:
                        raise ValueError(f"missing compressed positions for index ratio={ratio}")
                    idx_k = self._apply_key_rope(idx_k, plan.block_positions)
                else:
                    idx_k = self._apply_key_rope(idx_k, positions)
                idx_k = self._rotate_activation(idx_k)
            else:
                if self.compressor is None:
                    raise ValueError("indexer key projection requires key_override")
                idx_k = self._rotate_activation(self.compressor(x, attention_masks))
        else:
            idx_k = key_override
        idx_w = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return idx_q, idx_k, idx_w

    def _apply_key_rope(self, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply the index-key RoPE to a packed or batched key stream."""
        rd = self.rope_head_dim
        if rd == 0:
            return key
        key_nope, key_rope = torch.split(key, [self.head_dim - rd, rd], dim=-1)
        if golden_enabled():
            shaped = key_rope.unsqueeze(0).unsqueeze(2) if key.ndim == 2 else key_rope.unsqueeze(2)
            rotated = _golden_complex_rope(self.rope, shaped, positions).reshape_as(key_rope)
            return torch.cat([key_nope, rotated], dim=-1)
        if key.ndim == 2:
            rotated = (
                self.rope(
                    key_rope.unsqueeze(0).unsqueeze(2),
                    positions=positions.reshape(1, -1),
                )
                .squeeze(0)
                .squeeze(1)
            )
        else:
            rotated = self.rope(
                key_rope.unsqueeze(2),
                positions=positions.reshape(key.shape[0], key.shape[1]),
            ).squeeze(2)
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
        attendable.  Unused slots hold ``-1``.

        Returns:
            topk_indices: ``[B, L, K]`` container-grid slots, ``K =
                min(topk, S // ratio)``.
            index_scores: ``[B, L, S // ratio]`` masked indexer scores.
        """
        golden = golden_enabled()
        q, k, weights = (idx_q, idx_k, idx_w) if golden else (idx_q.float(), idx_k.float(), idx_w.float())
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
        if golden:
            indices = index_score.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
            valid = dense_mask.squeeze(1).gather(-1, indices)
            return indices.where(valid, -1), index_score
        topk_scores, topk_indices = index_score.topk(k, dim=-1)
        return topk_indices.where(topk_scores.isfinite(), -1), index_score


class LightningIndexer(Module):
    """Reference sparse-index selector; kernel configuration is reserved for fused integration."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int
        li_kernel_config: LightningIndexerKernelConfig = field(default_factory=LightningIndexerKernelConfig)

    def __init__(self, config: Config):
        super().__init__()
        self.index_topk = config.index_topk
        self.li_kernel_config = config.li_kernel_config

    def forward(self, idx_q, idx_k, idx_w, *, attention_masks):
        dense_mask = attention_masks.reference.ratios[4].dense_mask
        return Indexer.select(idx_q, idx_k, idx_w, dense_mask, self.index_topk)[0]
