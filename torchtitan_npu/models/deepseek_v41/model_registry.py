from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model_spec import ModelSpec

from .model import V41Model
from .vision import DeepSeekV4VisionEncoder, ImageMarkerEmbeddings

# Marker embedding init mirrors the reference tower.
_MARKER_INIT = {
    name: partial(torch.nn.init.normal_, std=1.0)
    for name in ("image_start", "image_newline", "image_end")
}

if TYPE_CHECKING:
    from torchtitan.protocols.model import ModelConfigConverter

from .config import (
    V41_COMPRESS_RATIOS,
    V41_FULL_COMPRESS_RATIOS,
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
)


@dataclass(frozen=True, slots=True)
class _V41Widths:
    """Per-flavor width set.

    The debug widths keep the real 40-layer compression/source structure
    while scaling only the widths (and the selection knobs that only make
    sense at the 512-token debug sequence length, so the sparse index and
    candidate selections stay active rather than clamping to dense).
    """

    dim: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    n_groups: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    moe_inter_dim: int
    candidate_topk_blocks: int
    vision_dim: int
    vision_heads: int
    vision_inter_dim: int
    max_seq_len: int


_FLASH_WIDTHS = _V41Widths(
    dim=5120,
    n_heads=64,
    head_dim=512,
    rope_head_dim=64,
    q_lora_rank=1280,
    o_lora_rank=1024,
    n_groups=8,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=512,
    moe_inter_dim=2304,
    candidate_topk_blocks=2048,
    vision_dim=1024,
    vision_heads=16,
    vision_inter_dim=2816,
    max_seq_len=4096,
)

_DEBUG_WIDTHS = _V41Widths(
    dim=512,
    n_heads=8,
    head_dim=64,
    rope_head_dim=16,
    q_lora_rank=128,
    o_lora_rank=128,
    n_groups=4,
    index_n_heads=8,
    index_head_dim=32,
    index_topk=32,
    moe_inter_dim=256,
    candidate_topk_blocks=4,
    vision_dim=128,
    vision_heads=8,
    vision_inter_dim=256,
    max_seq_len=512,
)


def _make_v41_config(
    *,
    n_layers: int,
    compress_ratios: tuple[int, ...],
    index_source_layers: tuple[int, ...],
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    widths: _V41Widths = _FLASH_WIDTHS,
):
    # Imported lazily: the V4 base package builds this package's vision
    # module, so a module-level import would form a cycle.
    from torchtitan_npu.models.deepseek_v4 import (
        _make_v4_config,
    )

    vocab_size = 129280
    config = _make_v4_config(
        dim=widths.dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        n_heads=widths.n_heads,
        head_dim=widths.head_dim,
        rope_head_dim=widths.rope_head_dim,
        q_lora_rank=widths.q_lora_rank,
        o_lora_rank=widths.o_lora_rank,
        n_groups=widths.n_groups,
        compress_ratios=compress_ratios,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=index_source_layers,
        candidate_source_layer=20,
        candidate_topk_blocks=widths.candidate_topk_blocks,
        candidate_block_size=8,
        window_size=128,
        norm_eps=1e-20,
        index_n_heads=widths.index_n_heads,
        index_head_dim=widths.index_head_dim,
        index_topk=widths.index_topk,
        moe_inter_dim=widths.moe_inter_dim,
        num_experts=16,
        num_shared_experts=1,
        top_k=6,
        n_hash_layers=0,
        route_norm=True,
        route_scale=1.5,
        load_balance_coeff=1e-3,
        hc_mult=4,
        sinkhorn_iters=20,
        hc_eps=1e-6,
        max_seq_len=widths.max_seq_len,
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16.0,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        post_q_rms_norm=False,
        rotation="none",
    )

    # The V4 base builder is decoder-only, so promote its config to the V4.1
    # multimodal config (which carries the vision fields) before injecting the
    # ViT and marker embeddings.
    config = V41Model.Config(**{f.name: getattr(config, f.name) for f in dataclasses.fields(config)})
    config.vision_encoder = DeepSeekV4VisionEncoder.Config(
        dim=widths.vision_dim,
        num_layers=32,
        num_heads=widths.vision_heads,
        inter_dim=widths.vision_inter_dim,
        patch_size=14,
        text_dim=widths.dim,
        downsample_ratio=3,
    )
    config.image_marker_embeddings = ImageMarkerEmbeddings.Config(dim=widths.dim, param_init=_MARKER_INIT)
    return config


def deepseek_v41_flash_30layers_16experts_vision_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    return _make_v41_config(
        n_layers=30,
        compress_ratios=V41_COMPRESS_RATIOS,
        index_source_layers=V41_INDEX_SOURCE_LAYERS,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def deepseek_v41_flash_40layers_16experts_vision_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    """Full 40-layer V4.1 backbone with the single-node 16-expert crop."""
    return _make_v41_config(
        n_layers=40,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def deepseek_v41_debugmodel_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    """Reduced-width V4.1 shape for the deterministic golden-trajectory tests.

    The real 40-layer structure (compression ratios, KV/index sources,
    16 experts, vision depth, real vocabulary) with debug widths, so the
    golden loss guard exercises every V4.1 code path quickly.
    """
    return _make_v41_config(
        n_layers=40,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        widths=_DEBUG_WIDTHS,
    )


def model_registry(
    flavor: str = "deepseek_v41_flash_30layers_16experts_vision",
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    # Lazy: the V4 base package imports this package's vision module.
    from torchtitan_npu.models.deepseek_v4 import (
        _register_step_pre_hooks,
        parallelize_deepseek_v4,
    )
    from .vision_state_dict import DeepSeekV41StateDictAdapter

    config_factories = {
        "deepseek_v41_flash_30layers_16experts_vision": deepseek_v41_flash_30layers_16experts_vision_config,
        "deepseek_v41_flash_40layers_16experts_vision": deepseek_v41_flash_40layers_16experts_vision_config,
        "deepseek_v41_debugmodel": deepseek_v41_debugmodel_config,
    }
    if flavor not in config_factories:
        raise ValueError(f"Unknown DeepSeek V4.1 flavor: {flavor}")
    config = config_factories[flavor](
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    for layer in config.layers:
        layer.moe.router.vision_enabled = True
    if converters is not None:
        validate_converter_order(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    return ModelSpec(
        name="deepseek_v41",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_v4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=_register_step_pre_hooks,
        state_dict_adapter=DeepSeekV41StateDictAdapter,
    )
