from dataclasses import dataclass

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
    vision depth and expert layout to the full shape; only the hidden
    width (and the candidate selection budget, which only makes sense
    relative to the sequence length) is scaled down.
    """

    hidden_size: int = 512
    candidate_topk_blocks: int = 4


@dataclass(frozen=True, slots=True)
class DeepSeekV41FullScaleProfile:
    """Reference-scale shape metadata; never selected by the 8-card launcher."""

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
