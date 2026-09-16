# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in V4.1 rotary multiplication, retaining FP32 position tables."""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v41.rope import V41RoPERotation


class AscV41RoPERotation(V41RoPERotation):
    @dataclass(kw_only=True, slots=True)
    class Config(V41RoPERotation.Config):
        pass

    def forward(self, x, cos, sin, *, inverse=False):
        if inverse:
            sin = -sin
        if self.mode == "half":
            cos, sin = torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
        values = x.float()
        if x.ndim == 3:
            values = values.unsqueeze(0)
        if cos.ndim == 3:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        shape = values.shape
        # CANN interleaved RoPE requires a shared batch axis for position tables.
        # Fold batch into sequence to retain distinct positions for every sample.
        if cos.shape[0] != 1:
            values = values.flatten(0, 1).unsqueeze(0)
            cos = cos.expand(shape[0], shape[1], -1, -1).flatten(0, 1).unsqueeze(0)
            sin = sin.expand(shape[0], shape[1], -1, -1).flatten(0, 1).unsqueeze(0)
        output = torch_npu.npu_rotary_mul(
            values.contiguous(),
            cos.float().contiguous(),
            sin.float().contiguous(),
            rotary_mode="half" if self.mode == "half" else "interleave",
        )
        output = output.reshape(shape)
        if x.ndim == 3:
            output = output.squeeze(0)
        return output.to(x.dtype)


@override(target=V41RoPERotation.Config, exact=True, description="V4.1 AscendC rotary multiplication")
def ascendc(cfg: V41RoPERotation.Config) -> AscV41RoPERotation.Config:
    return derive(cfg, AscV41RoPERotation.Config)
