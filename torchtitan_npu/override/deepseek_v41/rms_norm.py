# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in AscendC RMSNorm for the independent V4.1 model."""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v41.rms_norm import V41RMSNorm


class AscV41RMSNorm(V41RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(V41RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reference_fp32:
            return torch_npu.npu_rms_norm(x.float(), self.weight.float(), self.eps)[0].to(x.dtype)
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(target=V41RMSNorm.Config, exact=True, description="V4.1 AscendC RMSNorm")
def ascendc(cfg: V41RMSNorm.Config) -> AscV41RMSNorm.Config:
    return derive(cfg, AscV41RMSNorm.Config)
