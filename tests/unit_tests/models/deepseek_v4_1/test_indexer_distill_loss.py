# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The indexer distillation loss: a marginal-weighted KL with the full CSA denominator.

The teacher is the head-averaged attention mass on the selected compressed entries with
the window, entry and sink terms all in the denominator, so its row sum ``Z <= 1``.  The
loss is ``sum_j p_j (log t_j - log Y_j)`` with ``t = p / Z``, whose gradient w.r.t. the
student logits is ``Z * Y - p`` -- the closed form the NPU kernel implements.
"""

import torch

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1 import (
    V41_FULL_INDEX_SOURCE_LAYERS,
    deepseek_v4_1_debugmodel_config,
)
from torchtitan_npu.models.deepseek_v4_1.indexer import IndexerKLLoss


def _loss(coeff: float = 1.0) -> IndexerKLLoss:
    config = IndexerKLLoss.Config(
        coeff=coeff,
        reduce_mesh="batch",
        global_batch_size=1,
        softmax_scale=1.0,
    )
    return build_cpu_model(config)


def test_student_gradient_is_z_times_y_minus_p() -> None:
    """One head, two uniform-logit entries: dI = Z * Y - p, and the loss value matches."""
    loss = _loss()
    q_BLHD = torch.zeros(1, 1, 1, 1)
    cmp_k_BND = torch.zeros(1, 2, 1)
    topk_indices_BLK = torch.tensor([[[0, 1]]])
    # Both compressed logits are zero, so the conditional teacher is uniform and the mass
    # is exp(log 2 - lse); lse = log 2 makes it 1, i.e. p = [0.5, 0.5] and Z = 1.
    log_two = torch.log(torch.tensor(2.0))
    lse_BHL = log_two.expand(1, 1, 1).clone()
    # Student logits log 3 / 0 give Y = softmax(log 3, 0) = [0.75, 0.25].
    student_logits = torch.tensor([[[torch.log(torch.tensor(3.0)), 0.0]]], requires_grad=True)
    carrier = torch.zeros(1, 1, 1)

    returned = loss(q_BLHD, cmp_k_BND, topk_indices_BLK, lse_BHL, student_logits, carrier=carrier)
    # The carrier's values pass through unchanged; the loss hangs off its graph.
    torch.testing.assert_close(returned, carrier, rtol=0, atol=0)
    assert returned.grad_fn is not None
    returned.sum().backward()

    torch.testing.assert_close(
        student_logits.grad,
        torch.tensor([[[0.25, -0.25]]]),
        rtol=1e-6,
        atol=1e-7,
    )
    # L = sum_j p_j (log t_j - log Y_j) = log 2 - 0.5 log 3 for this case.
    expected_loss = float(log_two - 0.5 * torch.log(torch.tensor(3.0)))
    torch.testing.assert_close(loss.read(), torch.tensor(expected_loss), rtol=1e-6, atol=1e-7)


def test_invalid_slots_contribute_nothing() -> None:
    """A row whose slots are all unused produces no gradient on that row."""
    loss = _loss()
    q_BLHD = torch.zeros(1, 2, 1, 1)
    cmp_k_BND = torch.zeros(1, 2, 1)
    topk_indices_BLK = torch.tensor([[[0, 1]], [[-1, -1]]])
    log_two = torch.log(torch.tensor(2.0))
    lse_BHL = log_two.expand(1, 1, 2).clone()
    student_logits = torch.zeros(1, 2, 2, requires_grad=True)
    carrier = torch.zeros(1, 2, 1)

    loss(q_BLHD, cmp_k_BND, topk_indices_BLK, lse_BHL, student_logits, carrier=carrier).sum().backward()

    # The all-invalid row has no teacher mass, so its slots receive no gradient.
    torch.testing.assert_close(student_logits.grad[0, 1], torch.zeros(2), rtol=0, atol=0)


def test_distill_loss_is_attached_only_where_a_selection_exists() -> None:
    """The loss follows upstream's rule: every consumer, and only index sources score."""
    config = deepseek_v4_1_debugmodel_config()
    index_sources = set(V41_FULL_INDEX_SOURCE_LAYERS)

    for layer_id, layer in enumerate(config.layers):
        attention = layer.attention
        aux_loss = attention.inner_attention.aux_loss
        consumes_selection = attention.compress_ratio > 0 and any(
            source <= layer_id for source in V41_FULL_INDEX_SOURCE_LAYERS
        )
        assert (aux_loss is not None) == consumes_selection
        assert attention.indexer.needs_selection_scores == (aux_loss is not None and layer_id in index_sources)
        if aux_loss is not None:
            assert aux_loss.coeff == 0.01
            assert aux_loss.reduce_mesh == "batch"
            assert aux_loss.softmax_scale == attention.inner_attention.softmax_scale

def test_per_layer_losses_sum_to_the_pooled_teacher() -> None:
    """One pooled-teacher backward equals the sum of two consumer backwards.

    ``dI = Z * Y - p`` is affine in the teacher, so the per-consumer losses of a shared
    indexer accumulate exactly the gradient of the single pooled objective.
    """
    loss = _loss()
    q_BLHD = torch.zeros(1, 1, 1, 1)
    cmp_k_BND = torch.zeros(1, 2, 1)
    topk_indices_BLK = torch.tensor([[[0, 1]]])
    log_two = torch.log(torch.tensor(2.0))
    student_logits = torch.tensor([[[torch.log(torch.tensor(3.0)), 0.0]]])

    def student_grad(lse_value: torch.Tensor) -> torch.Tensor:
        logits = student_logits.clone().requires_grad_(True)
        carrier = torch.zeros(1, 1, 1)
        out = loss(q_BLHD, cmp_k_BND, topk_indices_BLK, lse_value.expand(1, 1, 1).clone(), logits, carrier=carrier)
        out.sum().backward()
        return logits.grad

    # Two consumers with the same uniform conditional and masses 1.0 and 0.5.
    first = student_grad(log_two)
    second = student_grad(torch.log(torch.tensor(4.0)))
    pooled = student_grad(log_two - torch.log(torch.tensor(1.5)))

    torch.testing.assert_close(pooled, first + second, rtol=1e-6, atol=1e-7)
