# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1 import (
    V41_CANDIDATE_SOURCE_LAYER,
    V41_FULL_COMPRESS_RATIOS,
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
    _make_v41_config,
    deepseek_v4_1_debugmodel_config,
)
from torchtitan_npu.models.deepseek_v4_1.attention import Attention
from torchtitan_npu.models.deepseek_v4_1.indexer import Indexer


def test_index_selection_mask_is_document_isolated_and_causal() -> None:
    """Entry j is visible to query t of the same document iff group j is complete at t."""
    indexer = Indexer.Config(
        num_index_heads=1,
        index_head_dim=1,
        index_topk=1,
        compress_ratio=2,
        is_source=False,
        owns_k=False,
    ).build()
    # Two documents of four tokens each; entry j covers tokens [2j, 2j + 2).
    doc_ids_BL = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
    visible, newest, newest_valid = indexer._selection_mask(doc_ids_BL, num_cmp=4, device=torch.device("cpu"))

    expected_visible = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, True, False],
            [False, False, True, False],
            [False, False, True, True],
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        newest.squeeze(-1),
        torch.tensor([[-1, 0, 0, 1, 1, 2, 2, 3]]),
        rtol=0,
        atol=0,
    )
    # A fresh document with no complete group yet pins no block.
    torch.testing.assert_close(
        newest_valid.squeeze(-1),
        torch.tensor([[False, True, True, True, False, True, True, True]]),
        rtol=0,
        atol=0,
    )


def test_v41_config_uses_specialized_attention_with_explicit_ownership() -> None:
    config = deepseek_v4_1_debugmodel_config()

    assert not hasattr(config, "hc_head")
    assert config.kv_source_layers == V41_KV_SOURCE_LAYERS
    assert config.index_source_layers == V41_FULL_INDEX_SOURCE_LAYERS

    kv_sources = set(V41_KV_SOURCE_LAYERS)
    index_sources = set(V41_FULL_INDEX_SOURCE_LAYERS)
    key_owners = kv_sources & index_sources
    for layer_id, layer in enumerate(config.layers):
        assert isinstance(layer.attention, Attention.Config)
        # Every layer compresses and indexes; only a source carries the weights.
        assert layer.attention.compressor.is_source == (layer_id in kv_sources)
        assert (layer.attention.compressor.wkv is not None) == (layer_id in kv_sources)
        assert layer.attention.indexer.is_source == (layer_id in index_sources)
        assert layer.attention.indexer.owns_k == (layer_id in key_owners)
        assert (layer.attention.indexer.wq_b is not None) == (layer_id in index_sources)

    ratio_one_source = config.layers[20].attention
    assert ratio_one_source.compress_ratio == 1
    assert ratio_one_source.compressor.is_source
    assert ratio_one_source.compressor.wgate is None
    assert not hasattr(ratio_one_source.compressor, "use_ape")
    assert ratio_one_source.indexer.is_source
    assert ratio_one_source.indexer.owns_k
    assert ratio_one_source.indexer.wk is not None
    assert ratio_one_source.indexer.k_norm is not None

    reindex = config.layers[24].attention
    assert not reindex.compressor.is_source
    # A reusing compressor holds no weights at all, so the checkpoint keys are the ones
    # the sources publish.
    assert reindex.compressor.wkv is None
    assert reindex.compressor.wgate is None
    assert reindex.compressor.norm is None
    assert reindex.compressor.rope is None
    assert reindex.indexer.is_source
    # A re-indexing layer scores the shared keys, so it owns no key projection.
    assert not reindex.indexer.owns_k
    assert reindex.indexer.wk is None
    assert reindex.indexer.k_norm is None
    assert not hasattr(reindex.indexer, "compressor")

    reuse = config.layers[21].attention
    assert not reuse.indexer.is_source
    assert reuse.indexer.wq_b is None
    assert not reuse.indexer.owns_k


def test_candidate_pool_roles_are_declared_per_indexer_layer() -> None:
    """The pool source and its consumers are config policy, not layer-id lookups."""
    config = deepseek_v4_1_debugmodel_config()
    # The frozen debug recipe: one pool built at layer 20, consumed by every later
    # index source, over blocks of 8 slots.
    pool_consumers = {24, 28, 32, 36}

    for layer_id, layer in enumerate(config.layers):
        indexer = layer.attention.indexer
        if not indexer.is_source:
            continue
        assert indexer.is_candidate_source == (layer_id == V41_CANDIDATE_SOURCE_LAYER)
        assert indexer.uses_candidates == (layer_id in pool_consumers)
        assert indexer.candidate_topk_blocks == 4
        assert indexer.candidate_block_size == 8

    assert config.layers[V41_CANDIDATE_SOURCE_LAYER].attention.indexer.is_candidate_source
    assert config.layers[24].attention.indexer.uses_candidates


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_layers": 20}, "supported V4.1 layer counts"),
        ({"compress_ratios": V41_FULL_COMPRESS_RATIOS[:-1]}, "compress_ratios must match n_layers"),
        (
            {"compress_ratios": (0, 0, 2, 1) + (2,) * 16 + (1,) * 20},
            "consumes the compressed KV of layer 2",
        ),
        ({"kv_source_layers": (8, 14, 20)}, "compress_ratio=2 but no KV source precedes it"),
        (
            {"index_source_layers": tuple(layer for layer in V41_FULL_INDEX_SOURCE_LAYERS if layer != 20)},
            "scores the shared index keys of layer 14",
        ),
        ({"index_source_layers": (24, 28, 32, 36)}, "re-indexing layer 24 has no preceding index source"),
        (
            {"candidate_source_layer": 24, "index_source_layers": (2, 8, 14, 20, 36)},
            "candidate-pool source must also be an index source",
        ),
        ({"candidate_source_layer": 24}, "candidate-pool source must own the compressed KV"),
    ],
    ids=[
        "layer_count",
        "ratio_table_length",
        "kv_ratio_mismatch",
        "no_kv_source",
        "index_ratio_mismatch",
        "index_source_missing",
        "candidate_source_not_index",
        "candidate_source_owns_no_kv",
    ],
)
def test_invalid_reuse_topology_is_rejected(overrides, match) -> None:
    """A consumer derives its plan from its own ratio, so a mismatched source fails at build."""
    topology = dict(
        n_layers=40,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
    )
    topology.update(overrides)
    with pytest.raises(ValueError, match=match):
        _make_v41_config(**topology)
