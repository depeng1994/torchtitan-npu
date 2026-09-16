# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in V4.1 grouped GEMM, retaining the reference expert arithmetic."""

from dataclasses import dataclass
from typing import cast

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend

from torchtitan_npu.models.deepseek_v41.moe import V41GroupedExperts


class AscV41GroupedExperts(V41GroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(V41GroupedExperts.Config):
        pass

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Retain the reference autograd edges for ranks receiving no tokens.
        if x_RD.shape[0] == 0:
            return super().forward(x_RD, num_tokens_per_expert_E, routed_scores_R=routed_scores_R)
        w1, w2, w3 = (
            weight.to_local() if isinstance(weight, DTensor) else weight
            for weight in (self.w1_EFD, self.w2_EDF, self.w3_EFD)
        )
        offsets = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets, axis, src=spmd.P, dst=spmd.V)

        gate = self._grouped_mm(A=x_RD, B_t=w1.transpose(-2, -1), offs=offsets).float()
        up = self._grouped_mm(A=x_RD, B_t=w3.transpose(-2, -1), offs=offsets).float()
        limit = cast("float", self.swiglu_limit)
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        hidden = F.silu(gate) * up
        if routed_scores_R is not None:
            hidden = routed_scores_R.reshape(-1, 1) * hidden
        return self._grouped_mm(A=hidden.to(x_RD.dtype), B_t=w2.transpose(-2, -1), offs=offsets).float()


@override(
    target=V41GroupedExperts.Config,
    exact=True,
    description="V4.1 routed experts with Ascend grouped GEMM",
)
def ascendc(cfg: V41GroupedExperts.Config) -> AscV41GroupedExperts.Config:
    return derive(cfg, AscV41GroupedExperts.Config)
