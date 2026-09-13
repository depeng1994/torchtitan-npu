# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 transformer block: Single-Pass mHC on top of the V4 block."""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType

from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4TransformerBlock
from torchtitan_npu.models.deepseek_v4.mhc import _make_identity_pre_mix


class DeepSeekV41TransformerBlock(DeepSeekV4TransformerBlock):
    """V4.1 block with Single-Pass mHC.

    Pre-mix flows between sublayers instead of being consumed within each
    sublayer (V4 classic semantics). The forward_with_pre_mix extension
    seam inherited from the V4 block is used internally; callers invoke
    this block through __call__ so FSDP hooks are properly triggered.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4TransformerBlock.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        pre_mix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pre_mix is None:
            pre_mix = _make_identity_pre_mix(x, self.hc_attn_pre.hc_mult)
        hidden, next_pre_mix = self.forward_with_pre_mix(
            x,
            input_ids,
            attention_masks,
            positions,
            pre_mix=pre_mix,
        )
        return hidden, next_pre_mix