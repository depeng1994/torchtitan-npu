# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class EngramArgs:
    """Model-flavor arguments for attaching Engram to selected layers.

    ``vocab_size_per_ngram`` gives the target rows per hash head for each
    N-gram order; the actual sizes are distinct primes at least that large.
    ``n_embed_per_ngram`` is the concatenated width of all hash heads for one
    order. For example, 640 with 8 heads produces 80 values per fetched row
    and a total memory width of 1280 for orders 2 and 3.
    """

    layer_ids: tuple[int, ...]
    vocab_size_per_ngram: tuple[int, ...]
    n_embed_per_ngram: int
    ngram_orders: tuple[int, ...] = (2, 3)
    num_heads_per_ngram: int = 8
    pad_id: int = 0
    hash_seed: int = 0
    norm_eps: float = 1e-5
    # A fixed model-shape alignment. The runtime EP degree must divide it;
    # this avoids changing checkpoint tensor shapes when EP changes. Published
    # V4.1 table sizes are logical row counts; this training layout adds padding
    # without changing the hash bucket ranges.
    table_padding_multiple: int = 2048
    token_id_map_path: str | None = None
    require_token_id_map: bool = True
    # When set, the compressed vocabulary the tokenizer map must produce. The
    # compressed size bounds the hash multipliers, so a map from a different
    # tokenizer silently changes every row ID; checking it turns that into a
    # startup error.
    compressed_vocab_size: int | None = None


# Published DeepSeek-V4.1-Flash text_config, revision 2bc89ac599031fa673cab993f1df02fc4a98c673:
# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/2bc89ac599031fa673cab993f1df02fc4a98c673
_ENGRAM_V41_FLASH_GEOMETRY = EngramArgs(
    layer_ids=(1, 14),
    ngram_orders=(2, 3, 4),
    vocab_size_per_ngram=(16_000_000, 16_000_000, 16_000_000),
    n_embed_per_ngram=2048,
    num_heads_per_ngram=8,
    pad_id=2,
    norm_eps=1e-20,
    require_token_id_map=True,
    compressed_vocab_size=99_092,
)


V41_LAYER_IDS = tuple(range(30))
V41_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 10
V41_KV_SOURCE_LAYERS = (2, 8, 14, 20)
V41_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28)
V41_CANDIDATE_SOURCE_LAYER = 20
V41_FULL_LAYER_IDS = tuple(range(40))
V41_FULL_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
# The reference config carries 43 compressed-layer entries; the V4.1 backbone
# uses the first 40 and this tree does not build the trailing depths.
V41_FULL_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20


@dataclass(frozen=True, slots=True)
class DeepSeekV41CropConfig:
    """Single-node 8-card V4.1 configuration with a continuous layer range."""

    engram: EngramArgs = field(default_factory=lambda: _ENGRAM_V41_FLASH_GEOMETRY)
    hidden_size: int = 5120
    num_hidden_layers: int = 30
    num_experts: int = 16
    expert_parallel_degree: int = 8
    fsdp_shard_degree: int = 8
    context_parallel_degree: int = 1
    sequence_length: int = 512
    vision_layers: int = 32
    compress_ratios: tuple[int, ...] = V41_COMPRESS_RATIOS
    layer_ids: tuple[int, ...] = V41_LAYER_IDS
    kv_source_layers: tuple[int, ...] = V41_KV_SOURCE_LAYERS
    index_source_layers: tuple[int, ...] = V41_INDEX_SOURCE_LAYERS
    candidate_source_layer: int = V41_CANDIDATE_SOURCE_LAYER
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    def __post_init__(self) -> None:
        if self.vision_layers < 4:
            raise ValueError("vision_layers must be at least 4")
        if self.num_hidden_layers != len(self.layer_ids):
            raise ValueError("layer_ids must describe every configured decoder layer")
        if self.layer_ids != tuple(range(self.num_hidden_layers)):
            raise ValueError("layer_ids must be the continuous decoder range 0..num_hidden_layers-1")
        if self.num_hidden_layers not in (30, 40):
            raise ValueError("the supported V4.1 layer configurations are 30 or 40 continuous layers")
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError("compress_ratios must match num_hidden_layers")
        if self.num_experts != 16 or self.expert_parallel_degree != 8:
            raise ValueError("the initial single-node crop requires 16 experts and EP8")
        if self.fsdp_shard_degree != 8:
            raise ValueError("the initial single-node crop requires FSDP shard degree 8")
        if self.context_parallel_degree != 1:
            raise ValueError("the first V4.1 implementation is CP1-only")
        if self.candidate_topk_blocks <= 0 or self.candidate_block_size <= 0:
            raise ValueError("candidate block parameters must be positive")

    def build_compression_spec(self):
        from .attention import build_v41_compression_spec

        return build_v41_compression_spec(
            layer_ids=self.layer_ids,
            ratios=self.compress_ratios,
            kv_source_layers=self.kv_source_layers,
            index_source_layers=self.index_source_layers,
            candidate_source_layer=self.candidate_source_layer,
            candidate_topk_blocks=self.candidate_topk_blocks,
            candidate_block_size=self.candidate_block_size,
        )


@dataclass(frozen=True, slots=True)
class DeepSeekV41FullLayerConfig(DeepSeekV41CropConfig):
    """Full 40-layer single-node shape with the 16-expert resource crop."""

    num_hidden_layers: int = 40
    sequence_length: int = 512
    compress_ratios: tuple[int, ...] = V41_FULL_COMPRESS_RATIOS
    layer_ids: tuple[int, ...] = V41_FULL_LAYER_IDS
    index_source_layers: tuple[int, ...] = V41_FULL_INDEX_SOURCE_LAYERS


@dataclass(frozen=True, slots=True)
class DeepSeekV41DebugConfig(DeepSeekV41FullLayerConfig):
    """Reduced-width full-structure shape for deterministic golden trajectories.

    Identical 40-layer decoder, compression ratios, KV/index sources,
    vision depth and expert layout to the full shape. Hidden width, candidate
    selection budget, and Engram bucket capacity and memory width are reduced;
    Engram layer positions and n-gram orders remain unchanged.
    """

    engram: EngramArgs = field(
        default_factory=lambda: EngramArgs(
            layer_ids=(1, 14),
            ngram_orders=(2, 3, 4),
            vocab_size_per_ngram=(1024, 1024, 1024),
            n_embed_per_ngram=256,
            num_heads_per_ngram=2,
            pad_id=2,
            norm_eps=1e-20,
            require_token_id_map=False,
        )
    )
    hidden_size: int = 512
    candidate_topk_blocks: int = 4


@dataclass(frozen=True, slots=True)
class DeepSeekV41FullScaleProfile:
    """Reference-scale shape metadata; never selected by the 8-card launcher."""

    engram: EngramArgs = field(default_factory=lambda: _ENGRAM_V41_FLASH_GEOMETRY)
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_experts: int = 384
    layer_ids: tuple[int, ...] = V41_FULL_LAYER_IDS
    index_source_layers: tuple[int, ...] = V41_FULL_INDEX_SOURCE_LAYERS
    candidate_source_layer: int = V41_CANDIDATE_SOURCE_LAYER
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    def __post_init__(self) -> None:
        if self.num_hidden_layers != len(self.layer_ids):
            raise ValueError("full-scale layer_ids must cover 40 layers")
        if self.num_experts != 384:
            raise ValueError("the full-scale V4.1 profile requires 384 experts")
        if not set(self.index_source_layers) <= set(self.layer_ids):
            raise ValueError("full-scale index source is outside the 40-layer profile")
