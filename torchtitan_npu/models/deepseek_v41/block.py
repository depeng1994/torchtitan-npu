# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 transformer block: Single-Pass mHC and CSA2 orchestration."""

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.moe import MoE

from . import attention as _v41_attention
from .attention import (
    DeepSeekV41Attention,
    V41AttentionContext,
    V41CompressionSpec,
    V41SharedInputs,
)
from .mhc import HcPost, HcPre, _make_identity_pre_mix


class DeepSeekV41TransformerBlock(TransformerBlock):
    """V4.1 block with Single-Pass mHC and explicit V4.1 attention inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attention: DeepSeekV41Attention.Config  # pyrefly: ignore [bad-override]
        moe: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config
        layer_id: int = -1

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.moe_enabled = True
        self.layer_id = config.layer_id
        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        self.moe = cfg.moe.build()
        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()
        self.compression_plan: V41CompressionSpec | None = None
        self.attention_context: V41AttentionContext | None = None

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        pre_mix: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        shared: V41SharedInputs | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, tuple]:
        if pre_mix is None:
            pre_mix = _make_identity_pre_mix(x, self.hc_attn_pre.hc_mult)
        if _v41_attention._SHARED_INPUTS and shared is None:
            raise RuntimeError("V4.1 block requires per-forward shared inputs (V41SharedInputs)")

        residual = x
        x, post, comb, attn_pre = self.hc_attn_pre.forward_with_pre_mix(x, pre_mix)
        published = None
        if _v41_attention._SHARED_INPUTS:
            assert shared is not None
            x, published = self.attention(self.attention_norm(x), attention_masks, positions, shared=shared)
        else:
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
        x = self.moe(self.ffn_norm(x), input_ids=input_ids, image_mask=image_mask)
        x = self.hc_post(x, residual, post, comb)
        if _v41_attention._SHARED_INPUTS:
            return x, ffn_pre, published  # pyrefly: ignore [bad-return]
        return x, ffn_pre

    def collapse_pre_mix(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        return self.hc_attn_pre.collapse(x, pre_mix)
