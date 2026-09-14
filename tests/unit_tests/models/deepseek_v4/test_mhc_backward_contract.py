# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression guard for the DeepSeek-V4 classic HcPre autograd graph."""

import torch
import torch.nn.functional as F

from torchtitan_npu.models.deepseek_v4.mhc import HcPre


def _legacy_v4_hc_pre_forward(module: HcPre, x: torch.Tensor):
    """The pre-V4.1 V4 HcPre graph: one FP32 cast shared by both branches."""
    shape, dtype = x.size(), x.dtype
    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + module.norm_eps)
    mixes = F.linear(flat, module.hc_fn.float()) * rsqrt
    pre, post, comb = module._sinkhorn(  # noqa: SLF001 - intentional regression oracle
        mixes,
        module.hc_scale.float(),
        module.hc_base.float(),
    )
    y = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
    return y.to(dtype), post, comb


def _loss(outputs) -> torch.Tensor:
    y, post, comb = outputs
    return y.float().square().mean() + post.square().mean() + comb.square().mean()


def test_v4_hc_pre_bf16_backward_matches_legacy_graph_bitwise():
    """V4 extension seams must not change BF16 backward accumulation order."""
    torch.manual_seed(42)
    config = HcPre.Config(hc_mult=2, dim=4, sinkhorn_iters=4, eps=1e-6, norm_eps=1e-6)
    actual = HcPre(config)
    reference = HcPre(config)
    reference.load_state_dict(actual.state_dict())

    x_actual = torch.randn(2, 3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    x_reference = x_actual.detach().clone().requires_grad_(True)

    actual_outputs = actual(x_actual)
    reference_outputs = _legacy_v4_hc_pre_forward(reference, x_reference)
    for got, expected in zip(actual_outputs, reference_outputs, strict=True):
        assert torch.equal(got, expected)

    _loss(actual_outputs).backward()
    _loss(reference_outputs).backward()

    assert x_actual.grad is not None
    assert x_reference.grad is not None
    assert torch.equal(x_actual.grad, x_reference.grad)
    for (actual_name, actual_param), (reference_name, reference_param) in zip(
        actual.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        assert actual_name == reference_name
        assert actual_param.grad is not None
        assert reference_param.grad is not None
        assert torch.equal(actual_param.grad, reference_param.grad), actual_name
