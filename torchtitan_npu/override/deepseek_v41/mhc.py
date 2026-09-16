# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent mHC stages; preserve V4.1's cross-layer pre-mix graph."""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override

import torchtitan_npu.ops.ascendc.mhc  # noqa: F401
from torchtitan_npu.models.deepseek_v41.mhc import HcPost, HcPre


class AscV41HcPre(HcPre):
    """Fuse only Sinkhorn; inherited collapse still uses the caller's pre mix."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if self.hc_mult not in (4, 6, 8) or not 1 <= self.sinkhorn_iters <= 100:
            raise ValueError("Native mHC Sinkhorn requires hc_mult in (4, 6, 8) and 1..100 iterations")

    def _sinkhorn(self, mixes, hc_scale, hc_base):
        n = self.hc_mult
        pre, post, comb = mixes.split([n, n, n * n], dim=-1)
        pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:n]) + self.eps
        post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[n : 2 * n])
        comb = (comb * hc_scale[2] + hc_base[2 * n :]).unflatten(-1, (n, n)).contiguous()
        comb, _, _ = torch_npu.npu_mhc_sinkhorn(comb, eps=self.eps, num_iters=self.sinkhorn_iters, out_flag=1)
        return pre, post, comb


class AscV41HcPost(HcPost):
    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(self, x, residual, post, comb):
        return torch.ops.cann_ops_transformer.mhc_post(residual, comb, x, post)


@override(target=HcPre.Config, exact=True, description="V4.1 single-pass mHC with native Sinkhorn")
def asc_sinkhorn(cfg: HcPre.Config) -> AscV41HcPre.Config:
    return derive(cfg, AscV41HcPre.Config)


@override(target=HcPost.Config, exact=True, description="V4.1 AscendC mHC post")
def asc_hc_post(cfg: HcPost.Config) -> AscV41HcPost.Config:
    return derive(cfg, AscV41HcPost.Config)
