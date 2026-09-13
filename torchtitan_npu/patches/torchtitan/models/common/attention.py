# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route dense GQA attention to Ascend ``npu_fusion_attention``.

TorchTitan's language models use ``FlexAttention`` for packed-document
training.  On NPU, PyTorch FlexAttention has no privateuse1 backend, so it
falls back to the dense math implementation (``sdpa_dense_backward``) that
materializes the full ``[B, N, S, S]`` score/softmax tensor.

This patch replaces the inner-attention call for the standard
``GQAttention`` + ``FlexAttention`` pair used by Llama3, Qwen3, etc. with
``torch.ops.npu.npu_fusion_attention``.  The upstream model graph, FSDP2
wrapping and activation checkpointing remain unchanged.
"""

import torch

from torchtitan.models.common.attention import FlexAttention, GQAttention

_ORIG_FLEX_FORWARD = FlexAttention.forward
_ORIG_GQA_FORWARD = GQAttention.forward
_PATCHED = False


def _packed_causal_mask(positions, B, L, device):
    """Return ``[B, 1, L, L]`` mask; True means MASKED."""
    q_idx = torch.arange(L, device=device).view(1, L, 1).expand(B, L, L)
    kv_idx = torch.arange(L, device=device).view(1, 1, L).expand(B, L, L)
    causal = q_idx < kv_idx
    if positions is not None:
        doc_ids = torch.cumsum((positions == 0).to(torch.int64), dim=1) - 1
        doc = doc_ids.unsqueeze(2) != doc_ids.unsqueeze(1)
        causal = causal | doc
    return causal.unsqueeze(1)


def _patched_gqa_forward(self, x_BLD, attention_masks, positions=None):
    xq_BLNH, xk_BLNH, xv_BLNH = self.qkv_linear(x_BLD)

    # Optional QK normalization (Qwen3-style)
    if self.q_norm is not None or self.k_norm is not None:
        assert self.q_norm is not None and self.k_norm is not None
        xq_BLNH = self.q_norm(xq_BLNH)
        xk_BLNH = self.k_norm(xk_BLNH)

    # Apply rotary embeddings
    xq_BLNH, xk_BLNH = self.rope(xq_BLNH, xk_BLNH, positions)

    out_BLNH = self.inner_attention(
        xq_BLNH,
        xk_BLNH,
        xv_BLNH,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
    ).contiguous()
    # Fold from out_BLNH's own shape (identical to upstream stock GQAttention).
    out_BLD = out_BLNH.flatten(2)
    return self.wo(out_BLD)


def _patched_flex_forward(
    self,
    q_BLNH,
    k_BLNH,
    v_BLNH,
    *,
    attention_masks,
    score_mod=None,
    scale=None,
    enable_gqa=False,
    positions=None,
    out_transform=None,
    **kwargs,
):
    # Keep the stock path for cases this NPU fast path cannot represent:
    #   - no positions (inference / non-LM callers)
    #   - non-trivial score_mod or out_transform epilogue
    #   - subset of the sequence on this rank (e.g. context parallelism)
    B, L, N, H = q_BLNH.shape
    if (
        positions is None
        or score_mod is not None
        or out_transform is not None
        or positions.size(1) != L
    ):
        return _ORIG_FLEX_FORWARD(
            self,
            q_BLNH,
            k_BLNH,
            v_BLNH,
            attention_masks=attention_masks,
            score_mod=score_mod,
            scale=scale,
            enable_gqa=enable_gqa,
            positions=positions,
            out_transform=out_transform,
            **kwargs,
        )

    dtype = q_BLNH.dtype
    scale = H**-0.5 if scale is None else scale

    # npu_fusion_attention layouts: BNSD / BSH.  Use BNSD.
    query = q_BLNH.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    key = k_BLNH.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    value = v_BLNH.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()

    # GQA: repeat KV heads to match Q heads, as done by Qwen3.
    if enable_gqa and key.size(1) < query.size(1):
        repeats = query.size(1) // key.size(1)
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)

    atten_mask = _packed_causal_mask(positions, B, L, query.device)

    out_BNSD = torch.ops.npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=query.size(1),
        input_layout="BNSD",
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=1.0,
        pre_tockens=L,
        next_tockens=0,
        sparse_mode=1,
    )[0]

    return out_BNSD.permute(0, 2, 1, 3).to(dtype)


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    FlexAttention.forward = _patched_flex_forward
    GQAttention.forward = _patched_gqa_forward
    _PATCHED = True


apply()
