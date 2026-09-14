# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 transformer block: Single-Pass mHC and CSA2 orchestration."""

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import AttentionMasksType

from torchtitan_npu.models.deepseek_v4.mhc import _make_identity_pre_mix
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4TransformerBlock

from .attention import DeepSeekV41Attention, V41AttentionContext, V41CompressionSpec


class DeepSeekV41TransformerBlock(DeepSeekV4TransformerBlock):
    """V4.1 block with Single-Pass mHC and explicit V4.1 attention inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4TransformerBlock.Config):
        attention: DeepSeekV41Attention.Config  # pyrefly: ignore [bad-override]

    def __init__(self, config: Config):
        super().__init__(config)
        self.compression_plan: V41CompressionSpec | None = None
        self.attention_context: V41AttentionContext | None = None
        self.image_mask: torch.Tensor | None = None

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
        if self.compression_plan is None or self.attention_context is None:
            raise RuntimeError("V4.1 block requires a compression plan and per-forward attention context")

        residual = x
        x, post, comb, attn_pre = self.hc_attn_pre.forward_with_pre_mix(x, pre_mix)
        x = self.attention(
            self.attention_norm(x),
            attention_masks,
            positions,
            layer_id=self.layer_id,
            plan=self.compression_plan,
            context=self.attention_context,
        )
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb, ffn_pre = self.hc_ffn_pre.forward_with_pre_mix(x, attn_pre)
        x = self.moe(self.ffn_norm(x), input_ids=input_ids, image_mask=self.image_mask)
        x = self.hc_post(x, residual, post, comb)
        return x, ffn_pre

    def collapse_pre_mix(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        return self.hc_attn_pre.collapse(x, pre_mix)
