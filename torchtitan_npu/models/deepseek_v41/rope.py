# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 rotary arithmetic; positions and caches remain owned by callers."""

from dataclasses import dataclass
from typing import Literal

import torch
from torchtitan.protocols.module import Module


class V41RoPERotation(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mode: Literal["interleave", "complex", "half"] = "interleave"

    def __init__(self, config: Config):
        super().__init__()
        if config.mode not in ("interleave", "complex", "half"):
            raise ValueError(f"Unsupported V4.1 RoPE mode: {config.mode}")
        self.mode = config.mode

    def forward(self, x, cos, sin, *, inverse=False):
        if inverse:
            sin = -sin
        values = x.float()
        if self.mode == "half":
            width = values.shape[-1] // 2
            real, imag = values.narrow(-1, 0, width), values.narrow(-1, width, width)
            return torch.cat((real * cos - imag * sin, imag * cos + real * sin), dim=-1).to(x.dtype)
        freqs = torch.complex(cos[..., ::2], sin[..., ::2])
        pairs = values.reshape(*x.shape[:-1], -1, 2)
        if self.mode == "complex":
            return torch.view_as_real(torch.view_as_complex(pairs) * freqs).flatten(-2).to(x.dtype)
        real, imag = pairs.unbind(-1)
        return (
            torch.stack((real * freqs.real - imag * freqs.imag, imag * freqs.real + real * freqs.imag), dim=-1)
            .flatten(-2)
            .to(x.dtype)
        )
