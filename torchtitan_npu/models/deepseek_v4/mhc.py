# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.protocols.module import Module


def _make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Return the one-hot stream mix used at the input of the main stack."""
    pre_mix = torch.zeros(
        (*x.shape[:2], hc_mult),
        device=x.device,
        dtype=torch.float32,
    )
    pre_mix[..., 0] = 1.0
    return pre_mix


class HcPre(Module):
    """Head-collaboration pre step; owns its mixing parameters."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        sinkhorn_iters: int = 20
        eps: float = 1e-6
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_mult = config.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.dim
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.sinkhorn_iters
        self.eps = config.eps
        self.norm_eps = config.norm_eps
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _sinkhorn(self, mixes, hc_scale, hc_base):
        hc_mult = self.hc_mult
        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + self.eps
        post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult].unsqueeze(0).unsqueeze(0))
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

        row_max = comb.max(dim=-1, keepdim=True).values
        comb = torch.exp(comb - row_max).clone()
        comb = comb / comb.sum(dim=-1, keepdim=True) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb

    def _mixes(self, x):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_fn.float()) * rsqrt
        return self._sinkhorn(mixes, self.hc_scale.float(), self.hc_base.float())

    @staticmethod
    def collapse(x, pre_mix):
        shape, dtype = x.size(), x.dtype
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float().reshape(shape), dim=2)
        return y.to(dtype)

    def forward_with_pre_mix(self, x, pre_mix=None):
        """Collapse with a caller-supplied stream mix (V4.1 Single-Pass).

        V4 classic calls the parameter-less :meth:`forward` — each sub-block
        collapses with its own freshly-computed pre mix and the result is
        independent of neighbouring sub-layers.  V4.1 instead carries the
        pre mix between sub-layers (Single-Pass mHC) and calls this method.
        ``pre_mix=None`` falls back to the V4 classic behaviour (uses the
        mix generated from ``x`` itself).
        """
        pre, post, comb = self._mixes(x)
        collapse_mix = pre if pre_mix is None else pre_mix
        return self.collapse(x, collapse_mix), post, comb, pre

    def forward(self, x):
        y, post, comb, _ = self.forward_with_pre_mix(x)
        return y, post, comb


class HcPost(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)


class HcHead(Module):
    """Head-collaboration head; owns its mixing parameters."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        norm_eps: float = 1e-6
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_dim = config.hc_mult * config.dim
        self.norm_eps = config.norm_eps
        self.eps = config.eps
        self.hc_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, x):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)
