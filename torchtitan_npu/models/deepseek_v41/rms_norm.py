# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 RMSNorm contracts, independently selectable from other models."""

from dataclasses import dataclass
from typing import cast

import torch
from torchtitan.models.common.nn_modules import RMSNorm


class V41RMSNorm(RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        # Pooled compressor and vision multiply the weight before the final cast.
        reference_fp32: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.reference_fp32 = config.reference_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.reference_fp32:
            return super().forward(x)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + cast("float", self.eps))
        return (self.weight * x).to(dtype)
