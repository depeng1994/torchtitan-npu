# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 model configuration: builders, flavors and the model registry.

Following torchtitan's model-directory layout, the model configuration lives in
this package entry point: the per-layer `Config` builders, the real V4.1
topology constants, the flavor factories and `model_registry` are all defined
here.  The trainer-facing recipes stay in `config_registry.py`, and every
component keeps its own `Config` in its own module.

The width sets and topology constants match the frozen
the registered 40-layer V4.1 shape.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .attention import Attention, CompressedSparseInnerAttention2
from .compressor import Compressor
from .indexer import Indexer, IndexerKLLoss
from .mhc import HcPost, HcPre
from .model import DeepSeekV41TransformerBlock, V41Model
from .state_dict_adapter import DeepSeekV41StateDictAdapter
from .vision import DeepSeekV41VisionEncoder, ImageMarkerEmbeddings

# std=1.0 mirrors the reference debug initialization; the marker values
# are the reference marker values, so changing the std changes the marker
# parameterization.
_MARKER_INIT = {name: partial(nn.init.normal_, std=1.0) for name in ("image_start", "image_newline", "image_end")}

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE
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
# Positions per candidate block of the hierarchical indexer.
_CANDIDATE_BLOCK_SIZE = 8


# The real V4.1 text-backbone topology: the compression ratio of every layer
# and the layers that own the shared compressed KV, the index selection and
# the candidate pool.  The 30-layer crop keeps the same source layers and
# truncates the trailing ratio-1 group; the 40-layer shapes are the frozen
# registered 40-layer V4.1 topology.
V41_CANDIDATE_SOURCE_LAYER = 20
V41_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 10
V41_FULL_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
V41_KV_SOURCE_LAYERS = (2, 8, 14, 20)
V41_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28)
V41_FULL_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)


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
    compress_ratio: int,
    norm_eps: float,
    is_source: bool,
    rope: WorkaroundComplexRoPE.Config | None,
) -> Compressor.Config:
    """Main-KV compressor config. A reusing layer holds no weights of its own.

    ``head_dim`` sizes the projections and the norm; the rotated span of the rope is
    carried by the rope config's own ``dim``/``split``.
    """
    return Compressor.Config(
        compress_ratio=compress_ratio,
        is_source=is_source,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wkv=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        # The softmax gate only exists when there is more than one token to pool.
        wgate=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source and compress_ratio > 1
            else None
        ),
        norm=(
            RMSNorm.Config(
                normalized_shape=head_dim,
                eps=norm_eps,
                param_init=_NORM_INIT,
            )
            if is_source
            else None
        ),
    )


def _make_indexer_config(
    *,
    dim: int,
    q_lora_rank: int,
    head_dim: int,
    index_head_dim: int,
    num_index_heads: int,
    index_topk: int,
    compress_ratio: int,
    norm_eps: float,
    is_source: bool,
    owns_k: bool,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    needs_selection_scores: bool,
    rope: WorkaroundComplexRoPE.Config | None,
) -> Indexer.Config:
    """Lightning-indexer config. Only an index source carries the projections; its keys
    come from its own compressor latent, or from the shared ones otherwise."""
    owns_index_k = owns_k and is_source
    return Indexer.Config(
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        is_source=is_source,
        owns_k=owns_index_k,
        is_candidate_source=is_candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        needs_selection_scores=needs_selection_scores,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wq_b=(
            Linear.Config(
                in_features=q_lora_rank,
                out_features=num_index_heads * index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        weights_proj=(
            Linear.Config(
                in_features=dim,
                out_features=num_index_heads,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        wk=(
            Linear.Config(
                in_features=head_dim,
                out_features=index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if owns_index_k
            else None
        ),
        k_norm=(
            RMSNorm.Config(
                normalized_shape=index_head_dim,
                eps=norm_eps,
                param_init=_NORM_INIT,
            )
            if owns_index_k
            else None
        ),
    )


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
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    layer_id: int,
    index_source_layers: tuple[int, ...],
    indexer_loss_coeff: float | None,
) -> Attention.Config:
    if source_key and external_key:
        raise ValueError("an indexer cannot own its source-key projection and consume an external key at the same time")
    if is_candidate_source and not owns_compressor:
        raise ValueError("the candidate-pool source must own the compressed KV it scores")

    hd = head_dim
    # Every rope config rotates the trailing ``rope_head_dim`` channels of its site's
    # head, so each site carries the un-rotated prefix width of that head.
    attention_rope = dataclasses.replace(rope, split=hd - rope_head_dim)
    indexer_rope = dataclasses.replace(rope, split=index_head_dim - rope_head_dim)
    per_group_in = (n_heads * hd) // n_groups
    per_group_out = n_groups * o_lora_rank
    softmax_scale = head_dim**-0.5
    # Every layer carries a compressor; only the sources carry its weights and its rope.
    compressor_cfg = _make_compressor_config(
        dim=dim,
        head_dim=hd,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        is_source=owns_compressor,
        rope=attention_rope if owns_compressor else None,
    )
    # The distillation loss is attached on every layer that consumes the selection:
    # the teacher is rebuilt from that layer's own attention mass, and because ``dI`` is
    # affine in the teacher, the per-layer losses sum to the pooled-teacher objective.
    compresses = compress_ratio > 0
    aux_loss = None
    if indexer_loss_coeff is not None and compresses and any(source <= layer_id for source in index_source_layers):
        aux_loss = IndexerKLLoss.Config(
            coeff=indexer_loss_coeff,
            reduce_mesh="batch",
            softmax_scale=softmax_scale,
        )
    needs_selection_scores = owns_indexer and aux_loss is not None

    # Every layer carries an indexer; only a source carries its weights and its rope.
    indexer_cfg = _make_indexer_config(
        dim=dim,
        q_lora_rank=q_lora_rank,
        head_dim=hd,
        index_head_dim=index_head_dim,
        num_index_heads=index_n_heads,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        is_source=owns_indexer,
        owns_k=source_key,
        is_candidate_source=is_candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        needs_selection_scores=needs_selection_scores,
        rope=indexer_rope if owns_indexer else None,
    )

    inner_attention_cfg = CompressedSparseInnerAttention2.Config(
        window_size=window_size,
        softmax_scale=softmax_scale,
        aux_loss=aux_loss,
    )
    return Attention.Config(
        n_heads=n_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        compress_ratio=compress_ratio,
        inner_attention=inner_attention_cfg,
        rope=attention_rope,
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
) -> MoE.Config:
    # Every block carries the same MoE stack, so one config serves them all; the common
    # factories build it exactly as upstream's V4.1 does (the plugin patch gives them the
    # ``sqrtsoftplus`` score function and the ``swiglu_limit`` passthrough).
    router = make_router_config(
        dim=dim,
        num_experts=num_experts,
        gate_param_init=_depth_init(layer_id),
        top_k=top_k,
        score_func="sqrtsoftplus",  # pyrefly: ignore [bad-argument-type]
        route_norm=route_norm,
        route_scale=route_scale,
    )
    # The plugin's two router extensions: the image-masked vision bias and the
    # reference's descending selection order.
    router = dataclasses.replace(router, vision_enabled=True, sorted_topk=True)
    return make_moe_config(
        num_experts=num_experts,
        router=router,
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=moe_inter_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=_depth_experts_init(layer_id),
            comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
        ),
        shared_experts=(
            make_ffn_config(
                dim=dim,
                hidden_dim=moe_inter_dim * num_shared_experts,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
                swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
            )
            if num_shared_experts > 0
            else None
        ),
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
    indexer_loss_coeff: float | None = 0.01,
    widths: _V41Widths = _FLASH_WIDTHS,
) -> V41Model.Config:
    # Per-layer source invariants. A reusing layer derives its container grid and its
    # index mask from its own ``compress_ratio``, so the ratio of a layer must equal the
    # ratio of the source it consumes; a topology that breaks this would silently read
    # another ratio's plan.
    if n_layers not in (30, 40):
        raise ValueError(f"the supported V4.1 layer counts are 30 or 40, got {n_layers}")
    if len(compress_ratios) != n_layers:
        raise ValueError(f"compress_ratios must match n_layers ({n_layers}), got {len(compress_ratios)}")
    for name, sources in (("kv_source_layers", kv_source_layers), ("index_source_layers", index_source_layers)):
        outside = sorted({source for source in sources if source < 0 or source >= n_layers})
        if outside:
            raise ValueError(f"{name} outside the {n_layers}-layer crop: {outside}")
    for layer_id, ratio in enumerate(compress_ratios):
        kv_source = max((source for source in kv_source_layers if source <= layer_id), default=None)
        if ratio > 0 and kv_source is None:
            raise ValueError(f"layer {layer_id} has compress_ratio={ratio} but no KV source precedes it")
        if kv_source is not None and compress_ratios[kv_source] != ratio:
            raise ValueError(
                f"layer {layer_id} (compress_ratio={ratio}) consumes the compressed KV of layer "
                f"{kv_source} (compress_ratio={compress_ratios[kv_source]}); the ratios must match"
            )
        if layer_id in index_source_layers and layer_id not in kv_source_layers:
            index_source = max((source for source in index_source_layers if source < layer_id), default=None)
            if index_source is None:
                raise ValueError(f"re-indexing layer {layer_id} has no preceding index source")
            if compress_ratios[index_source] != ratio:
                raise ValueError(
                    f"layer {layer_id} (compress_ratio={ratio}) scores the shared index keys of layer "
                    f"{index_source} (compress_ratio={compress_ratios[index_source]}); the ratios must match"
                )
    if candidate_source_layer not in set(index_source_layers):
        raise ValueError(
            "the candidate-pool source must also be an index source: "
            f"layer {candidate_source_layer} is not in index_source_layers"
        )
    if widths.candidate_topk_blocks <= 0 or _CANDIDATE_BLOCK_SIZE <= 0:
        raise ValueError("candidate block parameters must be positive")

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
            is_candidate_source=layer_id == candidate_source_layer,
            # Only an index source selects, so only it can consult the pool.
            uses_candidates=(
                layer_id in index_source_layers
                and layer_id != candidate_source_layer
                and 0 <= candidate_source_layer < layer_id
            ),
            candidate_topk_blocks=widths.candidate_topk_blocks,
            candidate_block_size=_CANDIDATE_BLOCK_SIZE,
            layer_id=layer_id,
            index_source_layers=index_source_layers,
            indexer_loss_coeff=indexer_loss_coeff,
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
                    hc_eps=1e-6,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_ffn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=widths.dim,
                    sinkhorn_iters=20,
                    hc_eps=1e-6,
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
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
        kv_source_layers=kv_source_layers,
        index_source_layers=index_source_layers,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=widths.candidate_topk_blocks,
        candidate_block_size=_CANDIDATE_BLOCK_SIZE,
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


def deepseek_v4_1_flash_30layers_16experts_vision_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    """Thirty of the forty decoder layers, for fast single-node validation."""
    return _make_v41_config(
        n_layers=30,
        compress_ratios=V41_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def deepseek_v4_1_flash_40layers_16experts_vision_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    """Full 40-layer V4.1 backbone with the single-node 16-expert crop."""
    return _make_v41_config(
        n_layers=40,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def deepseek_v4_1_debugmodel_config(
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
):
    """Reduced-width shape retaining the real 40-layer compression topology."""
    return _make_v41_config(
        n_layers=40,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        widths=_DEBUG_WIDTHS,
    )


def model_registry(
    flavor: str = "deepseek_v4_1_flash_30layers_16experts_vision",
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    from .parallelize import parallelize_deepseek_v4_1

    config_factories = {
        "deepseek_v4_1_flash_30layers_16experts_vision": deepseek_v4_1_flash_30layers_16experts_vision_config,
        "deepseek_v4_1_flash_40layers_16experts_vision": deepseek_v4_1_flash_40layers_16experts_vision_config,
        "deepseek_v4_1_debugmodel": deepseek_v4_1_debugmodel_config,
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
        name="deepseek_v4_1",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_v4_1,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=_register_step_pre_hooks,
        state_dict_adapter=DeepSeekV41StateDictAdapter,
    )


def _register_step_pre_hooks(optimizers, model_parts, parallel_dims) -> None:
    """Register MoE balancing and auxiliary-loss step hooks."""
    from torchtitan.components.optimizer import register_moe_load_balancing_hook

    from torchtitan_npu.patches.torchtitan.models.common.aux_loss import register_aux_loss_zero_hook

    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)
    register_aux_loss_zero_hook(optimizers, model_parts, parallel_dims)


# Public surface: the model config, its component classes and the model-config
# factories.  ``config_registry`` imports this module (never the reverse), so
# the trainer recipes are not re-exported here.
__all__ = [
    "V41_CANDIDATE_SOURCE_LAYER",
    "V41_COMPRESS_RATIOS",
    "V41_FULL_COMPRESS_RATIOS",
    "V41_FULL_INDEX_SOURCE_LAYERS",
    "V41_INDEX_SOURCE_LAYERS",
    "V41_KV_SOURCE_LAYERS",
    "Attention",
    "DeepSeekV41StateDictAdapter",
    "V41Model",
    "deepseek_v4_1_debugmodel_config",
    "deepseek_v4_1_flash_30layers_16experts_vision_config",
    "deepseek_v4_1_flash_40layers_16experts_vision_config",
    "model_registry",
]
