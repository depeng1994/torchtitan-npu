# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 model registry: direct construction of the V4.1 config.

No ``_make_v4_config`` detour — every layer, projection, MoE block and
init strategy is constructed here against the local V4.1 classes.  The
width sets and topology constants match the frozen
``dsv41_golden_2p_ep2_fsdp2`` baseline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn
from torchtitan.config import derive
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_routed_experts_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .attention import CompressedSparseAttention, DeepSeekV41Attention
from .block import DeepSeekV41TransformerBlock
from .compressor import Compressor, Indexer
from .mhc import HcPost, HcPre
from .model import V41Model
from .moe import V41MoE, V41Router
from .reference import ReferenceMetadataExtension
from .sparse_attention import V41SparseAttention
from .vision import DeepSeekV41VisionEncoder, ImageMarkerEmbeddings

# std=1.0 mirrors the reference debug initialization; the marker values
# are pinned by the frozen dsv41_golden_2p_ep2_fsdp2 trajectory, so changing
# the std re-anchors the guard.
_MARKER_INIT = {name: partial(nn.init.normal_, std=1.0) for name in ("image_start", "image_newline", "image_end")}

if TYPE_CHECKING:
    from torchtitan.protocols.model import ModelConfigConverter


@dataclass(frozen=True, slots=True)
class _V41Widths:
    """Per-flavor width set."""

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

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_SINK_INIT = {"attn_sink": partial(nn.init.trunc_normal_, std=0.02)}
_HC_PARAM_INIT = {
    "hc_fn": partial(nn.init.trunc_normal_, std=0.02),
    "hc_base": partial(nn.init.trunc_normal_, std=0.02),
    "hc_scale": partial(nn.init.trunc_normal_, std=0.02),
}
_SWIGLU_LIMIT = 10.0


def _output_linear_init(dim: int) -> dict:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _make_compressor_config(
    *,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    norm_eps: float,
    coff: int,
    rope: WorkaroundComplexRoPE.Config,
) -> Compressor.Config:
    return Compressor.Config(
        rope=dataclasses.replace(rope),
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        wkv=Linear.Config(
            in_features=dim,
            out_features=coff * head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wgate=(
            None
            if compress_ratio == 1
            else Linear.Config(
                in_features=dim,
                out_features=coff * head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
        ),
        norm=RMSNorm.Config(
            normalized_shape=head_dim,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
    )


def _make_indexer_config(
    *,
    dim: int,
    num_index_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    compress_ratio: int,
    norm_eps: float,
    rope: WorkaroundComplexRoPE.Config,
    source_key: bool,
    source_head_dim: int | None,
) -> Indexer.Config:
    config_kwargs = dict(
        rope=dataclasses.replace(rope),
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=num_index_heads * index_head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        weights_proj=Linear.Config(
            in_features=dim,
            out_features=num_index_heads,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
    )
    if source_key:
        if source_head_dim is None:
            raise ValueError("source-key indexer requires source_head_dim")
        config_kwargs.update(  # pyrefly: ignore [no-matching-overload]
            wk=Linear.Config(
                in_features=source_head_dim,
                out_features=index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            ),
            k_norm=RMSNorm.Config(
                normalized_shape=index_head_dim,
                eps=norm_eps,
                param_init=_NORM_INIT,
            ),
        )
    return Indexer.Config(**config_kwargs)


def _make_v41_attn_config(
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratio: int,
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope: WorkaroundComplexRoPE.Config,
    owns_compressor: bool,
    owns_indexer: bool,
    source_key: bool,
    external_key: bool,
) -> DeepSeekV41Attention.Config:
    if source_key and external_key:
        raise ValueError("an indexer cannot own its source-key projection and consume an external key at the same time")

    hd = head_dim
    per_group_in = (n_heads * hd) // n_groups
    per_group_out = n_groups * o_lora_rank
    softmax_scale = head_dim**-0.5
    compressor_cfg = None
    indexer_cfg = None

    if owns_compressor:
        compressor_cfg = _make_compressor_config(
            dim=dim,
            head_dim=hd,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            coff=1,
            rope=rope,
        )
    if owns_indexer:
        indexer_cfg = _make_indexer_config(
            dim=dim,
            num_index_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            rope=rope,
            source_key=source_key,
            source_head_dim=hd if (source_key or external_key) else None,
        )

    inner_attention_cfg = V41SparseAttention.Config(
        window_size=window_size,
        compress_ratio=compress_ratio,
        softmax_scale=softmax_scale,
        index_topk=index_topk,
    )
    return DeepSeekV41Attention.Config(
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        n_groups=n_groups,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        inner_attention=inner_attention_cfg,
        compressed_sparse_attention=CompressedSparseAttention.Config(
            inner_attention=inner_attention_cfg,
        ),
        rope=dataclasses.replace(rope),
        wq_a=Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=dim,
            out_features=hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=hd,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wo_a=BatchedLinear.Config(
            n_heads=n_groups,
            in_features=per_group_in,
            out_features=o_lora_rank,
            param_init=_LINEAR_INIT,
        ),
        wo_b=Linear.Config(
            in_features=per_group_out,
            out_features=dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        param_init=_SINK_INIT,
        compressor=compressor_cfg,
        indexer=indexer_cfg,
    )


def _make_v41_moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
    route_norm: bool,
    load_balance_coeff: float,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
):
    from .moe import V41FeedForward, V41GroupedExperts, V41RoutedExperts

    # The golden experts are the native V4.1 expert computation: derive the
    # factory configs onto the golden classes (the frozen baseline ran with
    # exactly these through the golden_moe override; here they are the
    # default, no class swap involved).
    routed = make_routed_experts_config(
        dim=dim,
        hidden_dim=moe_inter_dim,
        num_experts=num_experts,
        top_k=top_k,
        param_init=_depth_experts_init(layer_id),
        comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
    )
    routed = derive(
        routed,
        V41RoutedExperts.Config,
        inner_experts=derive(routed.inner_experts, V41GroupedExperts.Config),
    )
    shared = (
        make_ffn_config(
            dim=dim,
            hidden_dim=moe_inter_dim * num_shared_experts,
            w1_param_init=_LINEAR_INIT,
            w2w3_param_init=_depth_init(layer_id),
            swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
        )
        if num_shared_experts > 0
        else None
    )
    if shared is not None:
        shared = derive(shared, V41FeedForward.Config)
    return V41MoE.Config(
        num_experts=num_experts,
        router=V41Router.Config(
            num_experts=num_experts,
            gate=Linear.Config(
                in_features=dim,
                out_features=num_experts,
                bias=False,
                param_init=_depth_init(layer_id),
            ),
            top_k=top_k,
            score_func="sqrtsoftplus",
            route_scale=route_scale,
            route_norm=route_norm,
            vision_enabled=True,
        ),
        routed_experts=routed,
        shared_experts=shared,
        load_balance_coeff=load_balance_coeff,
    )


def _make_v41_config(
    *,
    n_layers: int,
    compress_ratios: tuple[int, ...],
    kv_source_layers: tuple[int, ...],
    index_source_layers: tuple[int, ...],
    candidate_source_layer: int,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    widths: _V41Widths = _FLASH_WIDTHS,
) -> V41Model.Config:
    vocab_size = 129280
    source_key_indexer_layers = tuple(layer_id for layer_id in index_source_layers if layer_id in kv_source_layers)
    external_key_indexer_layers = tuple(
        layer_id for layer_id in index_source_layers if layer_id not in kv_source_layers
    )

    window_size = 128
    norm_eps = 1e-20
    hc_mult = 4

    rope = WorkaroundComplexRoPE.Config(
        dim=widths.rope_head_dim,
        max_seq_len=widths.max_seq_len,
        theta=10000.0,
        scaling="none",
    )
    rope_compress = WorkaroundComplexRoPE.Config(
        dim=widths.rope_head_dim,
        max_seq_len=widths.max_seq_len,
        theta=160000.0,
        scaling="yarn",
        rope_factor=16.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=65536,
    )

    layers = []
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id]
        attn_cfg = _make_v41_attn_config(
            dim=widths.dim,
            n_heads=widths.n_heads,
            head_dim=widths.head_dim,
            rope_head_dim=widths.rope_head_dim,
            q_lora_rank=widths.q_lora_rank,
            o_lora_rank=widths.o_lora_rank,
            n_groups=widths.n_groups,
            compress_ratio=cr,
            window_size=window_size,
            norm_eps=norm_eps,
            index_n_heads=widths.index_n_heads,
            index_head_dim=widths.index_head_dim,
            index_topk=widths.index_topk,
            rope=rope_compress if cr >= 1 else rope,
            owns_compressor=layer_id in kv_source_layers,
            owns_indexer=layer_id in index_source_layers,
            source_key=layer_id in source_key_indexer_layers,
            external_key=layer_id in external_key_indexer_layers,
        )
        moe_cfg = _make_v41_moe_config(
            layer_id=layer_id,
            dim=widths.dim,
            moe_inter_dim=widths.moe_inter_dim,
            num_experts=16,
            num_shared_experts=1,
            top_k=6,
            route_scale=1.5,
            route_norm=True,
            load_balance_coeff=1e-3,
            moe_comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        )
        layers.append(
            DeepSeekV41TransformerBlock.Config(
                layer_id=layer_id,
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=widths.dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=widths.dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                moe=moe_cfg,
                hc_attn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=widths.dim,
                    sinkhorn_iters=20,
                    eps=1e-6,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_ffn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=widths.dim,
                    sinkhorn_iters=20,
                    eps=1e-6,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_post=HcPost.Config(),
            )
        )

    return V41Model.Config(
        dim=widths.dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=widths.dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=widths.dim, eps=norm_eps, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=widths.dim,
            out_features=vocab_size,
            param_init=_output_linear_init(widths.dim),
        ),
        layers=layers,
        window_size=window_size,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
        kv_source_layers=kv_source_layers,
        index_source_layers=index_source_layers,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=widths.candidate_topk_blocks,
        candidate_block_size=8,
        metadata_extension=ReferenceMetadataExtension.Config(
            window_size=window_size,
            num_heads=widths.n_heads,
            head_dim=widths.head_dim,
            index_n_heads=widths.index_n_heads,
            index_head_dim=widths.index_head_dim,
            index_topk=widths.index_topk,
            materialized_ratios=(1,),
        ),
        vision_encoder=DeepSeekV41VisionEncoder.Config(
            dim=widths.vision_dim,
            num_layers=32,
            num_heads=widths.vision_heads,
            inter_dim=widths.vision_inter_dim,
            patch_size=14,
            text_dim=widths.dim,
            downsample_ratio=3,
        ),
        image_marker_embeddings=ImageMarkerEmbeddings.Config(
            dim=widths.dim,
            param_init=_MARKER_INIT,
        ),
    )


def deepseek_v41_flash_30layers_16experts_vision_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    from .config import (
        V41_COMPRESS_RATIOS,
        V41_INDEX_SOURCE_LAYERS,
        V41_KV_SOURCE_LAYERS,
        DeepSeekV41CropConfig,
    )

    config = _make_v41_config(
        n_layers=30,
        compress_ratios=V41_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_INDEX_SOURCE_LAYERS,
        candidate_source_layer=20,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )

    return _attach_engram(config, DeepSeekV41CropConfig().engram)


def deepseek_v41_flash_40layers_16experts_vision_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    """Full 40-layer V4.1 backbone with the single-node 16-expert crop."""
    from .config import (
        V41_FULL_COMPRESS_RATIOS,
        V41_FULL_INDEX_SOURCE_LAYERS,
        V41_KV_SOURCE_LAYERS,
        DeepSeekV41FullLayerConfig,
    )

    config = _make_v41_config(
        n_layers=40,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=20,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )

    return _attach_engram(config, DeepSeekV41FullLayerConfig().engram)


def deepseek_v41_debugmodel_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    """Reduced-width shape retaining the real 40-layer compression topology."""
    from .config import (
        V41_FULL_COMPRESS_RATIOS,
        V41_FULL_INDEX_SOURCE_LAYERS,
        V41_KV_SOURCE_LAYERS,
        DeepSeekV41DebugConfig,
    )

    config = _make_v41_config(
        n_layers=40,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=20,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        widths=_DEBUG_WIDTHS,
    )

    return _attach_engram(config, DeepSeekV41DebugConfig().engram)


def _attach_engram(config, engram):
    from .engram_config import _make_engram_configs

    if any(layer_id >= len(config.layers) or layer_id < 0 for layer_id in engram.layer_ids):
        raise ValueError("Engram layer IDs must lie inside the decoder")
    configs = _make_engram_configs(
        hidden_size=config.dim,
        hc_mult=config.hc_mult,
        vocab_size=config.vocab_size,
        engram=engram,
    )
    for layer_id, engram_config in configs.items():
        config.layers[layer_id].engram = engram_config
    return config


def model_registry(
    flavor: str = "deepseek_v41_flash_30layers_16experts_vision",
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    from .parallelize import parallelize_deepseek_v41
    from .state_dict_adapter import DeepSeekV41StateDictAdapter

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
    if converters is not None:
        validate_converter_order(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    return ModelSpec(
        name="deepseek_v41",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_v41,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=_register_step_pre_hooks,
        state_dict_adapter=DeepSeekV41StateDictAdapter,
    )


def _register_step_pre_hooks(optimizers, model_parts, parallel_dims) -> None:
    """Register MoE balancing and auxiliary-loss step hooks."""
    from torchtitan.components.optimizer import register_moe_load_balancing_hook

    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)
