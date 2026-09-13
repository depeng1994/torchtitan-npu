import torch
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4.metadata import build_compressed_varlen_metadata
from torchtitan_npu.models.deepseek_v4.reference import ReferenceMetadataExtension
from torchtitan_npu.models.deepseek_v41.attention import DeepSeekV41Attention
from torchtitan_npu.models.deepseek_v41.config import (
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
)
from torchtitan_npu.models.deepseek_v41.model_registry import deepseek_v41_debugmodel_config


def test_ratio_one_materialization_is_explicit_policy() -> None:
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    varlen = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=3, max_k=3)
    metadata = build_compressed_varlen_metadata(varlen, (0, 1, 2))

    default_reference = ReferenceMetadataExtension(
        ReferenceMetadataExtension.Config(window_size=2, block_size=(1, 1))
    )(metadata)
    assert default_reference.reference.ratios[1].doc_of_block is None

    materialized_reference = ReferenceMetadataExtension(
        ReferenceMetadataExtension.Config(
            window_size=2,
            block_size=(1, 1),
            materialized_ratios=(1,),
        )
    )(metadata)
    ratio_one = materialized_reference.reference.ratios[1]

    torch.testing.assert_close(
        ratio_one.doc_of_block,
        torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ratio_one.block_local,
        torch.tensor([[0, 1, 0, 1, 2]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    expected = torch.tensor(
        [
            [
                [
                    [True, False, False, False, False],
                    [True, True, False, False, False],
                    [False, False, True, False, False],
                    [False, False, True, True, False],
                    [False, False, True, True, True],
                ]
            ]
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(ratio_one.dense_mask, expected, rtol=0, atol=0)


def test_v41_config_uses_specialized_attention_with_explicit_ownership() -> None:
    config = deepseek_v41_debugmodel_config()

    assert config.hc_head is None
    assert config.metadata_extension.materialized_ratios == (1,)
    assert config.kv_source_layers == V41_KV_SOURCE_LAYERS
    assert config.index_source_layers == V41_FULL_INDEX_SOURCE_LAYERS

    kv_sources = set(V41_KV_SOURCE_LAYERS)
    index_sources = set(V41_FULL_INDEX_SOURCE_LAYERS)
    for layer_id, layer in enumerate(config.layers):
        assert isinstance(layer.attention, DeepSeekV41Attention.Config)
        assert (layer.attention.compressor is not None) == (layer_id in kv_sources)
        assert (layer.attention.indexer is not None) == (layer_id in index_sources)

    ratio_one_source = config.layers[20].attention
    assert ratio_one_source.compress_ratio == 1
    assert ratio_one_source.compressor is not None
    assert ratio_one_source.compressor.wgate is None
    assert ratio_one_source.compressor.use_ape is False
    assert ratio_one_source.indexer is not None
    assert ratio_one_source.indexer.wk is not None
    assert ratio_one_source.indexer.k_norm is not None

    reindex = config.layers[24].attention
    assert reindex.compressor is None
    assert reindex.indexer is not None
    assert reindex.indexer.wk is None
    assert reindex.indexer.k_norm is None
    assert reindex.indexer.compressor is None
