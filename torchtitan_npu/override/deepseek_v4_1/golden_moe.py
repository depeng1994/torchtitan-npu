# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the BSD-style license in the repository LICENSE file.

"""Dense Golden expert arithmetic through the standard EP/FSDP boundaries."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override

from torchtitan_npu.patches.torchtitan.models.common.moe import (
    HashMoE,
    _ClampFeedForward,
    _ClampGroupedExperts,
)


class GoldenGroupedExperts(_ClampGroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(_ClampGroupedExperts.Config):
        pass

    def forward(self, x_RD, num_tokens_per_expert_E, *, routed_scores_R=None):
        weights = [
            weight.to_local() if isinstance(weight, DTensor) else weight
            for weight in (self.w1_EFD, self.w2_EDF, self.w3_EFD)
        ]
        counts = num_tokens_per_expert_E.tolist()
        parts = []
        start = 0
        for expert, count in enumerate(counts):
            end = start + count
            # Keep empty expert operations in autograd so every shard gets a gradient.
            score = None if routed_scores_R is None else routed_scores_R[start:end, None]
            parts.append(
                HashMoE._golden_expert(
                    x_RD[start:end],
                    *(weight[expert] for weight in weights),
                    route_weights=score,
                    limit=self.swiglu_limit,
                ).float()
            )
            start = end
        return torch.cat(parts, dim=0)


class GoldenFeedForward(_ClampFeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(_ClampFeedForward.Config):
        pass

    def forward(self, x):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.w2((F.silu(gate) * up).to(dtype)).float()


class GoldenMoE(HashMoE):
    @dataclass(kw_only=True, slots=True)
    class Config(HashMoE.Config):
        pass

    def forward(self, x_BLD, *, input_ids=None, image_mask=None):
        return self._routed_forward(
            x_BLD,
            input_ids=input_ids,
            image_mask=image_mask,
        ).to(x_BLD.dtype)


@override(
    target=HashMoE.Config, exact=True, description="Golden dense experts with standard EP dispatch and FSDP hooks"
)
def golden(cfg: HashMoE.Config) -> GoldenMoE.Config:
    routed = cfg.routed_experts
    if not routed.token_dispatcher.absorb_router_scores:
        raise ValueError("Golden experts require pre-W2 router score absorption")
    return derive(
        cfg,
        GoldenMoE.Config,
        routed_experts=derive(
            routed,
            type(routed),
            inner_experts=derive(routed.inner_experts, GoldenGroupedExperts.Config),
        ),
        shared_experts=(
            derive(cfg.shared_experts, GoldenFeedForward.Config) if cfg.shared_experts is not None else None
        ),
    )
