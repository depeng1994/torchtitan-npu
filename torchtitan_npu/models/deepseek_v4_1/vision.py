# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["DeepSeekV41VisionEncoder"]

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.models.common import Linear
from torchtitan.protocols.module import Module, ModuleList


class ImageMarkerEmbeddings(Module):
    """Learned embeddings for the three non-feature image protocol tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.image_start = torch.nn.Parameter(torch.empty(config.dim))
        self.image_newline = torch.nn.Parameter(torch.empty(config.dim))
        self.image_end = torch.nn.Parameter(torch.empty(config.dim))

    def forward(self, hidden, token_types):
        output = hidden.clone()
        output[token_types == 0] = self.image_start.to(output.dtype)
        output[token_types == 2] = self.image_newline.to(output.dtype)
        output[token_types == 3] = self.image_end.to(output.dtype)
        return output


class _ReferenceRMSNorm(Module):
    """Pure Torch RMSNorm with the reference FP32 accumulation contract."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def _vision_rope(n_h: int, n_w: int, rope_dim: int, theta: float, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().view(1, -1, 1, rope_dim), freqs.sin().view(1, -1, 1, rope_dim)


def _vision_rope_batch(grids: torch.Tensor, max_tokens: int, rope_dim: int, theta: float, device):
    """Build one 2D RoPE table per image instead of sharing the largest grid."""
    tables = [_vision_rope(int(h), int(w), rope_dim, theta, device) for h, w in grids.tolist()]
    cos = torch.ones((grids.shape[0], max_tokens, 1, rope_dim), device=device, dtype=torch.float32)
    sin = torch.zeros((grids.shape[0], max_tokens, 1, rope_dim), device=device, dtype=torch.float32)
    for index, ((image_cos, image_sin), (height, width)) in enumerate(zip(tables, grids.tolist(), strict=True)):
        tokens = min(int(height) * int(width), max_tokens)
        cos[index, :tokens] = image_cos[0, :tokens]
        sin[index, :tokens] = image_sin[0, :tokens]
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    pairs = x.float()
    width = pairs.shape[-1] // 2
    x1, x2 = pairs.narrow(-1, 0, width), pairs.narrow(-1, width, width)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class PatchEmbed(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        patch_size: int
        dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.proj = Linear.Config(
            in_features=3 * config.patch_size * config.patch_size,
            out_features=config.dim,
            bias=True,
        ).build()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)


class VisionAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int

    def __init__(self, config: Config):
        super().__init__()
        if config.dim % config.num_heads != 0:
            raise ValueError("vision dim must be divisible by vision heads")
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.wqkv = Linear.Config(
            in_features=config.dim,
            out_features=3 * config.dim,
            bias=True,
        ).build()
        self.wo = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=True,
        ).build()

    def forward(self, x, cos, sin, valid):
        unbatched = x.ndim == 2
        if unbatched:
            p, dim = x.shape
            qkv = self.wqkv(x)
            width = qkv.shape[-1] // 3
            q, k, v = (qkv.narrow(-1, i * width, width) for i in range(3))
            q = q.view(p, self.num_heads, self.head_dim)
            k = k.view(p, self.num_heads, self.head_dim)
            v = v.view(p, self.num_heads, self.head_dim)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                dropout_p=0.0,
            )
            return self.wo(out.transpose(0, 1).reshape(p, dim))

        n, p, dim = x.shape
        qkv = self.wqkv(x)
        width = qkv.shape[-1] // 3
        q, k, v = (qkv.narrow(-1, i * width, width) for i in range(3))
        q = q.view(n, p, self.num_heads, self.head_dim)
        k = k.view(n, p, self.num_heads, self.head_dim)
        v = v.view(n, p, self.num_heads, self.head_dim)
        q = _apply_rope(q, cos, sin).transpose(1, 2)
        k = _apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        mask = valid[:, None, :, None] & valid[:, None, None, :]
        # The reference's tensor contract is [heads, patches, dim] with no batch
        # dimension: passing an extra batch dimension or an all-true mask can
        # select a different SDPA kernel and change BF16 rounding even though the
        # result is mathematically the same.
        if n == 1 and bool(valid[0].all()):
            out = F.scaled_dot_product_attention(q[0], k[0], v[0], dropout_p=0.0)
            return self.wo(out.transpose(0, 1).reshape(p, dim)).unsqueeze(0)
        # On this backend F.scaled_dot_product_attention already dispatches to the
        # Ascend fused attention kernel, so there is no separate fused call to make.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.wo(out.transpose(1, 2).reshape(n, p, dim))


class VisionMLP(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        inter_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = Linear.Config(
            in_features=config.dim,
            out_features=2 * config.inter_dim,
            bias=False,
        ).build()
        self.w2 = Linear.Config(
            in_features=config.inter_dim,
            out_features=config.dim,
            bias=False,
        ).build()

    def forward(self, x):
        hidden = self.w1(x)
        width = hidden.shape[-1] // 2
        gate, up = hidden.narrow(-1, 0, width), hidden.narrow(-1, width, width)
        return self.w2(F.silu(gate) * up)


class VisionBlock(Module):
    attention_type = VisionAttention

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        inter_dim: int
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = _ReferenceRMSNorm(config.dim, config.norm_eps)
        self.attn = self.attention_type.Config(dim=config.dim, num_heads=config.num_heads).build()
        self.norm2 = _ReferenceRMSNorm(config.dim, config.norm_eps)
        self.mlp = VisionMLP.Config(dim=config.dim, inter_dim=config.inter_dim).build()

    def forward(self, x, cos, sin, valid):
        x = x + self.attn(self.norm1(x), cos, sin, valid)
        return x + self.mlp(self.norm2(x))


class ImageAligner(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        text_dim: int
        downsample_ratio: int

    def __init__(self, config: Config):
        super().__init__()
        self.downsample_ratio = config.downsample_ratio
        in_dim = config.dim * self.downsample_ratio**2
        self.w1 = Linear.Config(in_features=in_dim, out_features=config.text_dim, bias=True).build()
        self.w2 = Linear.Config(in_features=config.text_dim, out_features=config.text_dim, bias=True).build()

    def forward(self, x, grids):
        ratio = self.downsample_ratio
        if x.shape[0] == 1:
            height, width = (int(value) for value in grids[0].tolist())
            sample = x[0, : height * width].view(height, width, -1).permute(2, 0, 1)
            sample = F.pad(sample, (0, -width % ratio, 0, -height % ratio))
            sample = F.unfold(sample.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
            return self.w2(F.gelu(self.w1(sample))).unsqueeze(0)

        outputs = []
        max_tokens = 0
        grid_values = grids.to(device="cpu").tolist()
        for sample, (height, width) in zip(x, grid_values, strict=True):
            height = int(height)
            width = int(width)
            sample = sample[: height * width].view(height, width, -1).permute(2, 0, 1)
            pad_h = (-height) % ratio
            pad_w = (-width) % ratio
            if pad_h or pad_w:
                sample = F.pad(sample, (0, pad_w, 0, pad_h))
            out_h, out_w = sample.shape[-2] // ratio, sample.shape[-1] // ratio
            sample = sample.view(sample.shape[0], out_h, ratio, out_w, ratio)
            sample = sample.permute(1, 3, 0, 2, 4).reshape(out_h * out_w, -1)
            sample = self.w2(F.gelu(self.w1(sample)))
            outputs.append(sample)
            max_tokens = max(max_tokens, sample.shape[0])
        result = x.new_zeros((len(outputs), max_tokens, self.w2.out_features))
        for idx, sample in enumerate(outputs):
            result[idx, : sample.shape[0]] = sample
        return result


class DeepSeekV41VisionEncoder(Module):
    block_type = VisionBlock

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 1024
        num_layers: int = 32
        num_heads: int = 16
        inter_dim: int = 2816
        patch_size: int = 14
        rope_theta: float = 10000.0
        text_dim: int = 4096
        downsample_ratio: int = 3
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.rope_theta = config.rope_theta
        self.patch_embed = PatchEmbed.Config(patch_size=config.patch_size, dim=config.dim).build()
        self.blocks = ModuleList(
            [
                self.block_type.Config(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    inter_dim=config.inter_dim,
                    norm_eps=config.norm_eps,
                ).build()
                for _ in range(config.num_layers)
            ]
        )
        self.norm = _ReferenceRMSNorm(config.dim, config.norm_eps)
        self.aligner = ImageAligner.Config(
            dim=config.dim,
            text_dim=config.text_dim,
            downsample_ratio=config.downsample_ratio,
        ).build()

    def forward(self, patches: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
        if patches.shape[0] == 1:
            height, width = (int(value) for value in grids[0].tolist())
            patch_count = height * width
            x = self.patch_embed(patches[0, :patch_count])
            rope_dim = self.dim // self.num_heads // 2
            cos, sin = _vision_rope(height, width, rope_dim, self.rope_theta, x.device)
            cos, sin = cos.squeeze(0), sin.squeeze(0)
            valid = torch.ones(patch_count, dtype=torch.bool, device=x.device)
            for block in self.blocks:
                x = block(x, cos, sin, valid)
            x = self.norm(x)
            return self.aligner(x.unsqueeze(0), grids).clone()

        x = self.patch_embed(patches)
        num_patches = (grids[:, 0] * grids[:, 1]).to(torch.long)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < num_patches[:, None]
        rope_dim = self.dim // self.num_heads // 2
        cos, sin = _vision_rope_batch(grids, x.shape[1], rope_dim, self.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin, valid)
        return self.aligner(self.norm(x), grids)
