# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license in the repository.

# TODO: backward path of _SparseMLA (document-local indices gradient handling)
# requires NPU integration testing; CPU cannot load cann_ops_transformer.
from itertools import pairwise

import pytest
import torch
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v41.metadata import build_compressed_varlen_metadata
from torchtitan_npu.override.deepseek_v41.sparse_attn import ascendc


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("bounds", [[0, 8], [0, 3, 8], [0, 4, 8]], ids=["single", "odd-packed", "even-packed"])
def test_kernel_receives_document_local_indices(monkeypatch, ratio, bounds):
    cu = torch.tensor(bounds, dtype=torch.int32)
    common = build_compressed_varlen_metadata(VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=8, max_k=8), (0, 1, 2))
    metadata = ascendc.AscV41MetadataExtension(
        ascendc.AscV41MetadataExtension.Config(window_size=2, materialized_ratios=(1,))
    )(common)
    cmp_cu = cu if ratio == 1 else metadata.plans[2].cu_seqlens_cmp_k
    indices = torch.empty((1, 8, 5), dtype=torch.int32)
    expected = torch.empty((8, 1, 5), dtype=torch.int32)
    for doc, (begin, end) in enumerate(pairwise(bounds)):
        start, stop = int(cmp_cu[doc]), int(cmp_cu[doc + 1])
        # First/last valid entries, another document (or out of range), padding,
        # and an exclusive-end index. Model indices use container coordinates.
        foreign = 0 if start else int(cmp_cu[-1])
        indices[0, begin:end] = torch.tensor([start, stop - 1, foreign, -1, stop])
        last = stop - start - 1 if ratio == 1 or stop - start > 1 else -1
        expected[begin:end, 0] = torch.tensor([0, last, -1, -1, -1])

    calls = []

    def kernel(q, original, shared, local_indices, *args):
        calls.append(local_indices)
        return torch.zeros_like(q)

    monkeypatch.setattr(ascendc._SparseMLA, "apply", kernel)
    attention = ascendc.AscV41SparseAttention.Config(
        window_size=2, compress_ratio=ratio, softmax_scale=0.5, index_topk=5
    ).build()
    q = torch.zeros((1, 8, 2, 4), dtype=torch.bfloat16)
    output = attention(
        q,
        torch.zeros((1, 8, 4), dtype=torch.bfloat16),
        torch.zeros((1, int(cmp_cu[-1]), 4), dtype=torch.bfloat16),
        sparse_indices=indices,
        attn_sink=torch.zeros(2),
        attention_masks=metadata,
    )
    assert output.shape == q.shape
    assert len(calls) == 1
    assert calls[0].is_contiguous()
    torch.testing.assert_close(calls[0], expected, rtol=0, atol=0)
