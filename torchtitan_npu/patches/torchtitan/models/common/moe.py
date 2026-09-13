# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4095

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hash routing and SwiGLU clamping for the common MoE (DeepSeek-V4).

Adds ``hash`` / ``vocab_size`` to ``TokenChoiceTopKRouter.Config``, an
``input_ids`` argument to ``MoE.forward``, and the ``"sqrtsoftplus"`` score
function; adds an optional ``swiglu_limit`` to ``GroupedExperts`` (routed
experts) and ``FeedForward`` (shared experts) with the DeepSeek-V4 clamp
(``up`` in ``[-limit, limit]``, ``gate`` capped at ``limit`` — the same
clamp as transformers / the inference repo; 0 disables).  The config
factories gain a ``swiglu_limit`` passthrough.  Swaps the originals at
import time so downstream code keeps using the upstream names; non-hash
configs and the default 0.0 clamp take the byte-identical upstream path.
``config_utils`` is deliberately imported inside ``apply()``, after the
class swaps, so its module-level imports bind the patched classes — keep
the swap-then-import order.
"""

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    RoutedExperts,
    TokenChoiceTopKRouter,
)


# ``torchtitan_npu/__init__.py`` imports this module, so reaching back into the
# models package for the shared helper would re-enter it while it is still
# initialising and the class swaps below would never run.
def _golden_enabled() -> bool:
    """True when the golden operator path is selected (single USE_GOLDEN switch)."""
    return (
        os.getenv(
            "USE_GOLDEN",
            os.getenv("TORCHTITAN_NPU_VISION_GOLDEN", os.getenv("TORCHTITAN_NPU_GOLDEN_TRAINING", "0")),
        )
        == "1"
    )


__all__ = ["HashMoE", "HashRouter"]


def _build_hash_routing_table(vocab_size, num_experts, top_k, device=None, chunk_size=8192):
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        tid2eid[start:end] = torch.rand((end - start, num_experts), device=device).topk(top_k, dim=-1).indices
    return tid2eid


class HashRouter(TokenChoiceTopKRouter):
    """TokenChoiceTopKRouter with optional DSV4 hash routing.

    When ``hash`` is set, tokens are routed by a fixed tid->expert table
    (``tid2eid``) instead of score top-k; ``expert_bias_E`` is not applied on
    hash layers.  ``vocab_size`` is required iff ``hash``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        hash: bool = False
        vocab_size: int | None = None
        vision_enabled: bool = False
        score_func: Literal[  # pyrefly: ignore [bad-override]
            "softmax", "sigmoid", "sqrtsoftplus"
        ] = "sigmoid"

    def __init__(self, config: Config):
        super().__init__(config)
        self.hash = config.hash
        self.vocab_size = config.vocab_size
        self.bias_vl = (
            torch.nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32)) if config.vision_enabled else None
        )
        if self.hash:
            if config.vocab_size is None:
                raise ValueError("hash routing requires vocab_size.")
            self.register_buffer(
                "tid2eid",
                _build_hash_routing_table(self.vocab_size, self.num_experts, self.top_k),
                persistent=True,
            )

    def reset_parameters(self):
        bias_vl = getattr(self, "bias_vl", None)
        if bias_vl is not None:
            torch.nn.init.zeros_(bias_vl)

    def _init_self_buffers(self, *, buffer_device=None):
        if self.hash:
            if buffer_device is None:
                buffer_device = self.tid2eid.device
            with torch.device(buffer_device):  # pyrefly: ignore [no-matching-overload]
                self.tid2eid = _build_hash_routing_table(
                    self.vocab_size,
                    self.num_experts,
                    self.top_k,
                    device=buffer_device,
                )

    def _select_experts(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        # Golden mode requires the sorted top-k order: the frozen reference
        # digests depend on it, and the index order feeds the fp32 route_norm
        # sum, so it is visible in the final logits.  Non-golden keeps the
        # upstream unsorted behaviour; the TileLang router override replaces
        # this seam with the fused topk_gate kernel.
        return scores_for_choice.topk(self.top_k, dim=-1, sorted=_golden_enabled())[1]

    def forward(self, x_BLD, expert_bias_E=None, *, input_ids=None, image_mask=None):
        # Compute gate in float32 to help stability of expert load balancing
        # (torchtitan TokenChoiceTopKRouter pattern).
        gate_weight = getattr(self.gate, "weight", None)
        if gate_weight is not None and not isinstance(gate_weight, DTensor):
            # The reference computes router scores from FP32 activations and
            # weights even when the model storage dtype is FP16/BF16.
            scores = F.linear(x_BLD.float(), gate_weight.float())
        else:
            with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
                scores = self.gate(x_BLD)
            # Some eager backends do not implement float32 autocast.  Promote
            # the gate result explicitly before score transforms and top-k.
            scores = scores.float()
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            # Use the baseline's ``F.softplus`` expression directly. Equivalent
            # formulas can round differently near top-k decision boundaries.
            scores = F.softplus(scores).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            if input_ids is None:
                raise ValueError("input_ids is required for DSV4 hash routing.")
            selected_experts_indices = self.tid2eid[input_ids]
        else:
            choice_bias = scores.new_zeros(self.num_experts) if expert_bias_E is None else expert_bias_E
            if image_mask is not None and self.bias_vl is not None:
                choice_bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, choice_bias)
            scores_for_choice = scores + choice_bias
            # Apply node-limited routing if configured (upstream behavior).
            if self.num_expert_groups is not None:
                scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
            selected_experts_indices = self._select_experts(scores_for_choice)

        top_scores = scores.gather(dim=-1, index=selected_experts_indices)

        if self._debug_force_load_balance:
            selected_experts_indices, top_scores = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        return top_scores, selected_experts_indices, scores


class HashMoE(MoE):
    """MoE whose forward threads an optional ``input_ids`` to the router.

    ``Config`` is redefined so ``build()`` instantiates this class rather than
    the upstream ``MoE`` (the inherited alias would construct the base).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # [TODO] need to add https://github.com/pytorch/torchtitan/pull/3634
        if getattr(self.router, "hash", False):
            self.expert_bias_E = None

    def forward(
        self,
        x_BLD: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward through the router (with optional ``input_ids``) and experts.

        The body mirrors upstream ``MoE.forward``; ``input_ids`` is only
        consumed by the router's hash path.
        """
        if _golden_enabled():
            return self._golden_forward(x_BLD, input_ids=input_ids, image_mask=image_mask)
        return self._routed_forward(x_BLD, input_ids=input_ids, image_mask=image_mask)

    def _routed_forward(self, x_BLD, *, input_ids=None, image_mask=None):
        _B, L, _D = x_BLD.shape
        sp_size = getattr(self.routed_experts.token_dispatcher, "sp_size", 1)
        if not isinstance(x_BLD, DTensor) and getattr(self, "seq_dim_tp_sharded", False):
            seq_pad = 0
            seq_dim_pad_tokens = 0
        else:
            seq_pad = sp_size - L if sp_size > L else 0
            if seq_pad:
                x_BLD = F.pad(x_BLD, (0, 0, 0, seq_pad))
                L = L + seq_pad
            seq_dim_pad_tokens = (-L) % sp_size

        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        ) = self.router(
            x_BLD,
            getattr(self, "expert_bias_E", None),
            input_ids=input_ids,
            image_mask=image_mask,
        )

        routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_BLK,
            True,
        )
        num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))

        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_BLD = self.routed_experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            num_local_tokens_per_expert_E,
        )

        shared_out_BLD = self.shared_experts(x_BLD) if self.shared_experts is not None else None

        if shared_out_BLD is not None:
            out_BLD = out_BLD + shared_out_BLD

        if seq_dim_pad_tokens:
            out_BLD = out_BLD[:, :L, :]

        if seq_pad:
            out_BLD = out_BLD[:, : L - seq_pad, :]

        return out_BLD

    @staticmethod
    def _golden_expert(x, w1, w2, w3, route_weights=None, limit=0.0):
        dtype = x.dtype
        gate = F.linear(x, w1).float()
        up = F.linear(x, w3).float()
        if limit > 0:
            up = up.clamp(min=-limit, max=limit)
            gate = gate.clamp(max=limit)
        hidden = F.silu(gate) * up
        if route_weights is not None:
            hidden = route_weights * hidden
        return F.linear(hidden.to(dtype), w2)

    def _golden_forward(self, x_BLD, *, input_ids=None, image_mask=None):
        shape = x_BLD.shape
        x = x_BLD.view(-1, shape[-1])
        weights, indices, _ = self.router(
            x_BLD,
            getattr(self, "expert_bias_E", None),
            input_ids=input_ids,
            image_mask=image_mask,
        )
        weights = weights.view(-1, weights.shape[-1])
        indices = indices.view(-1, indices.shape[-1])
        y = torch.zeros_like(x, dtype=torch.float32)
        routed = self.routed_experts.inner_experts
        num_experts = self.router.num_experts
        # The loop below indexes the expert tensors with global expert ids,
        # which is only valid when every expert lives here as a plain tensor;
        # EP/FSDP runs must use the golden_moe.golden override instead.
        if isinstance(routed.w1_EFD, DTensor) or routed.w1_EFD.shape[0] != num_experts:
            raise ValueError(
                "_golden_forward requires plain replicated expert tensors; "
                "enable torchtitan_npu.override.deepseek_v41.golden_moe.golden "
                "for EP/FSDP runs"
            )
        counts = torch.bincount(indices.flatten(), minlength=num_experts).tolist()
        for expert_id in range(num_experts):
            if counts[expert_id] == 0:
                continue
            token_ids, top_ids = torch.where(indices == expert_id)
            y[token_ids] += self._golden_expert(
                x[token_ids],
                routed.w1_EFD[expert_id],
                routed.w2_EDF[expert_id],
                routed.w3_EFD[expert_id],
                weights[token_ids, top_ids, None],
                routed.swiglu_limit,
            )
        if self.shared_experts is not None:
            shared = self.shared_experts
            y += self._golden_expert(x, shared.w1.weight, shared.w2.weight, shared.w3.weight, None, shared.swiglu_limit)
        return y.type_as(x_BLD).view(shape)


class _RouterScoreAbsorbingRoutedExperts(RoutedExperts):
    """RoutedExperts that absorbs dispatcher-aligned router scores before w2."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.size(-1)
        T = B * L
        x_TD = x_BLD.view(T, D)

        topk_scores_TK = topk_scores_BLK.view(T, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)
        dispatcher = self.token_dispatcher
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        # Patch override: consume scores carried by dispatcher metadata and pass
        # them into grouped experts before the down projection.
        routed_scores_R = getattr(metadata, "routed_scores_R", None)

        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD,
                num_global_tokens_per_local_expert_e,
                routed_scores_R=routed_scores_R,
            )

        out_TD = dispatcher.combine(routed_output_RD, metadata, x_TD.to(routed_output_RD.dtype))
        return out_TD.view(B, -1, D)


class _ClampGroupedExperts(GroupedExperts):
    """GroupedExperts with the optional DeepSeek-V4 SwiGLU clamp."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        # SwiGLU limit on the gate/up activations; 0 disables the clamp.
        # DeepSeek-V4 uses 10.0 (transformers config default).
        swiglu_limit: float = 0.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def _grouped_mm(self, *, A: torch.Tensor, B_t: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        if A.device.type == "cpu" and A.dtype == torch.float16:
            ends = offs.to(device="cpu", dtype=torch.long).tolist()
            start = 0
            chunks = []
            for end, weight in zip(ends, B_t, strict=True):
                chunks.append(A[start:end] @ weight)
                start = end
            if chunks:
                return torch.cat(chunks, dim=0)
            return A.new_empty((0, B_t.shape[-1]))
        return super()._grouped_mm(A=A, B_t=B_t, offs=offs)

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw expert computation; the gate/up grouped-mms are clamped
        before the SiLU when ``swiglu_limit > 0``."""
        if isinstance(self.w1_EFD, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        compute_dtype = x_RD.dtype if x_RD.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        g_RF = self._grouped_mm(
            A=x_RD.to(compute_dtype),
            B_t=w1_EFD.to(compute_dtype).transpose(-2, -1),
            offs=offsets_E,
        )
        u_RF = self._grouped_mm(
            A=x_RD.to(compute_dtype),
            B_t=w3_EFD.to(compute_dtype).transpose(-2, -1),
            offs=offsets_E,
        )
        if self.swiglu_limit > 0:
            u_RF = torch.clamp(u_RF, min=-self.swiglu_limit, max=self.swiglu_limit)
            g_RF = torch.clamp(g_RF, max=self.swiglu_limit)
        h_RF = F.silu(g_RF) * u_RF
        if routed_scores_R is not None:
            h_RF = (h_RF.float() * routed_scores_R.float().reshape(-1, 1)).to(h_RF.dtype)
        return self._grouped_mm(
            A=h_RF,
            B_t=w2_EDF.to(compute_dtype).transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)


class _ClampFeedForward(FeedForward):
    """FeedForward (shared expert) with the optional SwiGLU clamp."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        swiglu_limit: float = 0.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        return self.w2(F.silu(gate) * up)


def _clamp_make_routed_experts_config(
    *,
    swiglu_limit: float = 0.0,
    absorb_router_scores: bool = True,
    **kwargs,
):
    """``make_routed_experts_config`` with the clamp passthrough."""
    cfg = _original_make_routed_experts_config(**kwargs)
    dispatcher_config = cfg.token_dispatcher
    dispatcher_config = dataclasses.replace(
        dispatcher_config,
        absorb_router_scores=absorb_router_scores,
    )
    return dataclasses.replace(
        cfg,
        inner_experts=dataclasses.replace(cfg.inner_experts, swiglu_limit=swiglu_limit),
        token_dispatcher=dispatcher_config,
    )


def _clamp_make_ffn_config(*, swiglu_limit: float = 0.0, **kwargs):
    """``make_ffn_config`` with the clamp passthrough."""
    cfg = _original_make_ffn_config(**kwargs)
    return dataclasses.replace(cfg, swiglu_limit=swiglu_limit)


# Assigned in ``apply()`` before the clamp factories are ever called; declared
# at module scope so type checkers see the names.
_original_make_routed_experts_config: Callable[..., Any] = lambda **kwargs: None
_original_make_ffn_config: Callable[..., Any] = lambda **kwargs: None


# Swap the originals so ``torchtitan.models.common.moe`` resolves to the
# extended classes (see the module docstring).  Modules imported after this
# patch binds the new classes; ``hash=False`` and the default 0.0 clamp keep
# the upstream behavior.


def apply() -> None:
    import torchtitan.models.common.feed_forward
    import torchtitan.models.common.moe

    # Swap the classes BEFORE ``config_utils`` is imported: its module-level
    # ``from torchtitan.models.common.moe import GroupedExperts`` (and
    # ``from ...feed_forward import FeedForward``) then binds the patched
    # classes, so the factories build the clamped configs.  Importing it
    # earlier would freeze the originals into its namespace and the
    # factories would keep building unpatched configs.
    torchtitan.models.common.moe.TokenChoiceTopKRouter = HashRouter
    torchtitan.models.common.moe.MoE = HashMoE
    torchtitan.models.common.moe.GroupedExperts = _ClampGroupedExperts
    torchtitan.models.common.moe.RoutedExperts = _RouterScoreAbsorbingRoutedExperts
    torchtitan.models.common.feed_forward.FeedForward = _ClampFeedForward

    import torchtitan.models.common.config_utils

    torchtitan.models.common.config_utils.GroupedExperts = _ClampGroupedExperts
    torchtitan.models.common.config_utils.RoutedExperts = _RouterScoreAbsorbingRoutedExperts

    global _original_make_routed_experts_config, _original_make_ffn_config
    _original_make_routed_experts_config = torchtitan.models.common.config_utils.make_routed_experts_config
    _original_make_ffn_config = torchtitan.models.common.config_utils.make_ffn_config
    torchtitan.models.common.config_utils.make_routed_experts_config = _clamp_make_routed_experts_config
    torchtitan.models.common.config_utils.make_ffn_config = _clamp_make_ffn_config


apply()
