# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 model layer: the standalone decoder with multimodal input handling.

A plain :class:`torchtitan.models.common.decoder.Decoder` 鈥?no MTP depths,
no context-parallel sharding (both rejected in ``update_from_config``).
The golden reference arithmetic is the only path; ``USE_GOLDEN=0`` is
rejected by the config registry entry, and the model output follows the
golden FP32 contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata
from torchtitan.models.common.decoder import Decoder

from .attention import V41AttentionContext, build_v41_compression_spec
from .metadata import build_compressed_varlen_metadata
from .mhc import _make_identity_pre_mix
from .reference import ReferenceMetadataExtension
from .vision import DeepSeekV41VisionEncoder, ImageMarkerEmbeddings  # noqa: TC001
from .vision_data import scatter_image_features

if TYPE_CHECKING:
    from torchtitan_npu.models.common.metadata_extension import MetadataExtension


def scatter_image_embeddings(
    hidden: torch.Tensor,
    visual: torch.Tensor,
    spans: torch.Tensor,
) -> torch.Tensor:
    """Scatter padded visual features into ``[batch, seq, hidden]`` embeddings."""
    if spans.ndim != 2 or spans.shape[-1] != 3:
        raise ValueError(f"image spans must have shape [num_images, 3], got {tuple(spans.shape)}")
    if visual.ndim != 3 or visual.shape[0] != spans.shape[0]:
        raise ValueError(
            "visual features must have shape [num_images, tokens, hidden] "
            f"matching spans, got visual={tuple(visual.shape)}, spans={tuple(spans.shape)}"
        )
    output = hidden.clone()
    for image_idx, (sample_idx, start, length) in enumerate(spans.detach().cpu().tolist()):
        sample_idx, start, length = int(sample_idx), int(start), int(length)
        if sample_idx < 0 or sample_idx >= hidden.shape[0]:
            raise ValueError(f"invalid image span {(sample_idx, start, length)}")
        if start < 0 or length < 0:
            raise ValueError(f"invalid image span {(sample_idx, start, length)}")
        if start + length > hidden.shape[1] or length > visual.shape[1]:
            raise ValueError(
                f"image span {(sample_idx, start, length)} exceeds hidden/visual shapes "
                f"{tuple(hidden.shape)}/{tuple(visual.shape)}"
            )
        output[sample_idx, start : start + length] = visual[image_idx, :length].to(output.dtype)
    return output


class V41Model(Decoder):
    """DeepSeek-V4.1 backbone with CSA2 policy and the vision tower wired in."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        n_layers: int
        hc_mult: int = 4
        compress_ratios: tuple[int, ...]
        window_size: int
        kv_source_layers: tuple[int, ...] = ()
        index_source_layers: tuple[int, ...] = ()
        candidate_source_layer: int = 20
        candidate_topk_blocks: int = 2048
        candidate_block_size: int = 8
        vision_encoder: DeepSeekV41VisionEncoder.Config | None = None
        image_marker_embeddings: ImageMarkerEmbeddings.Config | None = None
        metadata_extension: MetadataExtension.Config = field(default_factory=ReferenceMetadataExtension.Config)

        def update_from_config(self, *, config, **kwargs):
            parallelism = config.parallelism
            if parallelism.tensor_parallel_degree != 1:
                raise NotImplementedError("DeepSeek V4.1 currently supports TP=1 only")
            cp = parallelism.context_parallel_degree
            pp = parallelism.pipeline_parallel_degree
            if cp != 1:
                raise NotImplementedError(f"DeepSeek V4.1 currently supports CP=1 only; got CP={cp}")
            if pp != 1:
                raise NotImplementedError(f"DeepSeek V4.1 does not support pipeline parallelism; got PP={pp}")
            # compile (aot_eager / inductor) is now supported; the rejection
            # has been removed to allow torch.compile on V4.1.
            Decoder.Config.update_from_config(self, config=config, **kwargs)

            if hasattr(config, "training"):
                from torchtitan.models.common.rope import RoPE

                seq_len = config.training.seq_len
                for _, rope_cfg, _, _ in self.traverse(RoPE.Config):
                    setattr(rope_cfg, "max_seq_len", seq_len)  # noqa: B010

            from .sharding import set_deepseek_v41_sharding_config

            set_deepseek_v41_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            from typing import cast

            from torchtitan.models.utils import get_moe_model_nparams_and_flops

            deepseek_v41_model = cast("V41Model", model)
            first_attention = self.layers[0].attention
            head_dims = 2 * first_attention.head_dim
            nparams, num_flops_per_token = get_moe_model_nparams_and_flops(
                self,
                deepseek_v41_model,
                first_attention.n_heads,
                head_dims,
                seq_len,
            )

            # Subtract the upstream per-token full-attention estimate and add
            # the window + compressed-container attention actually performed.
            num_flops_per_token -= 6 * len(self.layers) * first_attention.n_heads * head_dims * seq_len
            for layer in self.layers:
                attention = layer.attention
                inner_attention = attention.inner_attention
                num_flops_per_token += (
                    6 * attention.n_heads * (2 * attention.head_dim) * min(seq_len, inner_attention.window_size)
                )
                if attention.compress_ratio > 1:
                    compressed_seq_len = seq_len // attention.compress_ratio
                    if attention.indexer is not None:
                        num_flops_per_token += (
                            6
                            * attention.indexer.num_index_heads
                            * attention.indexer.index_head_dim
                            * compressed_seq_len
                        )
                        compressed_seq_len = min(compressed_seq_len, inner_attention.index_topk)
                    num_flops_per_token += 6 * attention.n_heads * (2 * attention.head_dim) * compressed_seq_len
            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config
        self.hc_mult = cfg.hc_mult
        self.compress_ratios = tuple(cfg.compress_ratios)
        self.window_size = cfg.window_size
        self.vocab_size = cfg.vocab_size
        self._metadata_extension = cfg.metadata_extension.build()

        self.vision_encoder = config.vision_encoder.build() if config.vision_encoder is not None else None
        self.image_marker_embeddings = (
            config.image_marker_embeddings.build() if config.image_marker_embeddings is not None else None
        )

        self.compression_plan = build_v41_compression_spec(
            layer_ids=tuple(range(config.n_layers)),
            ratios=self.compress_ratios[: config.n_layers],
            kv_source_layers=config.kv_source_layers,
            index_source_layers=config.index_source_layers,
            candidate_source_layer=config.candidate_source_layer,
            candidate_topk_blocks=config.candidate_topk_blocks,
            candidate_block_size=config.candidate_block_size,
        )
        self.attention_context = V41AttentionContext.empty()
        # Per-layer shared-input specs resolved once, outside any compiled
        # region: (kv_source, index_source, index_before, after_candidate,
        # compute_candidates, active_ratio, index_key_ratio,
        # candidate_topk_blocks, candidate_block_size).
        self._layer_shared_spec: dict[int, tuple] = {}
        for layer_id in range(config.n_layers):
            layer = self.layers[str(layer_id)]
            kv_source = self.compression_plan.kv_source_for(layer_id)
            index_source = self.compression_plan.index_source_for(layer_id)
            # KV-source layers consume their own key projection (no override):
            # mirror the reference `index_source = None if is_kv_source else before`.
            index_before = (
                None
                if layer_id in self.compression_plan.kv_source_layers
                else self.compression_plan.index_source_before(layer_id)
            )
            self._layer_shared_spec[layer_id] = (
                kv_source,
                index_source,
                index_before,
                layer_id > config.candidate_source_layer,
                layer_id == config.candidate_source_layer,
                self.compression_plan.ratios[kv_source]
                if kv_source is not None
                else layer.attention.compress_ratio,  # pyrefly: ignore [missing-attribute]
                self.compression_plan.ratios[index_before] if index_before is not None else 1,
                self.compression_plan.candidate_topk_blocks,
                self.compression_plan.candidate_block_size,
            )
            layer.compression_plan = self.compression_plan  # pyrefly: ignore [bad-argument-type]
            layer.attention_context = self.attention_context  # pyrefly: ignore [bad-argument-type]
            layer.attention.publish_layer_id = layer_id  # pyrefly: ignore [missing-attribute, bad-argument-type]

    def apply_activation_checkpointing_extensions(self, policy) -> None:
        """Apply the shared AC policy to the V4.1 vision blocks."""
        if self.vision_encoder is None:
            return
        for name, block in self.vision_encoder.blocks.named_children():
            self.vision_encoder.blocks.register_module(
                name,
                policy._wrap_block(block, base_fqn=f"vision_encoder.blocks.{name}"),
            )

    def apply_fsdp_extensions(self, *, dp_mesh, training, parallelism, parallel_dims) -> None:
        """FSDP-wrap the V4.1 vision tower before the shared decoder wrapper."""
        if self.vision_encoder is None:
            return
        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.distributed.fsdp import apply_fsdp_to_vision_encoder

        apply_fsdp_to_vision_encoder(
            self.vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallel_dims.pp_enabled,
        )

    def build_attention_masks(self, inputs, labels, extra_kwargs, *, cp_mesh=None, load_balancer_type=None):
        """Build the model-owned per-batch compressed-attention metadata."""
        if cp_mesh is not None:
            raise NotImplementedError("DeepSeek V4.1 currently supports CP=1 only")
        positions = extra_kwargs.get("positions")
        masks = self.get_attention_masks(positions=positions)
        if not isinstance(masks, VarlenMetadata):
            raise TypeError(
                "DeepSeek-V4.1 compression requires a varlen stream (the "
                "inner attention is varlen-typed), got "
                f"{type(masks)}."
            )
        common = build_compressed_varlen_metadata(masks, self.compress_ratios)
        if self._metadata_extension is not None:
            common = self._metadata_extension(common)
        extra_kwargs["attention_masks"] = common
        return inputs, labels, extra_kwargs

    def _prepare_multimodal_embeddings(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor,
        image_grid: torch.Tensor,
        image_spans: torch.Tensor | None,
        image_feature_indices: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.vision_encoder is None:
            raise ValueError("image inputs were provided but no vision encoder is configured")
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.flatten(0, 1)
        if image_grid.ndim == 3:
            image_grid = image_grid.flatten(0, 1)
        if image_spans is None or image_spans.numel() == 0:
            image_spans = None
        elif image_spans.ndim == 3:
            image_spans = image_spans.flatten(0, 1)
        hidden = self.tok_embeddings(tokens)
        visual = self.vision_encoder(pixel_values, image_grid)
        if self.image_marker_embeddings is not None and token_types is not None:
            hidden = self.image_marker_embeddings(hidden, token_types)
        if image_feature_indices is not None:
            counts = (
                (image_grid[:, 0].to(torch.long) + self.vision_encoder.aligner.downsample_ratio - 1)
                // self.vision_encoder.aligner.downsample_ratio
            ) * (
                (image_grid[:, 1].to(torch.long) + self.vision_encoder.aligner.downsample_ratio - 1)
                // self.vision_encoder.aligner.downsample_ratio
            )
            flat_visual = (
                torch.cat(
                    [features[: int(count)] for features, count in zip(visual, counts.tolist(), strict=True)],
                    dim=0,
                )
                if counts.numel()
                else visual.new_empty((0, visual.shape[-1]))
            )
            return scatter_image_features(hidden, flat_visual, image_feature_indices)
        if image_spans is None:
            return hidden + visual.sum() * 0
        return scatter_image_embeddings(hidden, visual, image_spans)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid: torch.Tensor | None = None,
        image_spans: torch.Tensor | None = None,
        image_feature_indices: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ):
        """V4.1 forward with Single-Pass mHC for vision and text-only batches."""
        self.attention_context.reset()

        if pixel_values is None:
            image_mask = None
            embeds = input_embeds
        else:
            if image_grid is None or (image_spans is None and image_feature_indices is None):
                raise ValueError("pixel_values requires image_grid and image_spans or image_feature_indices")
            image_mask = token_types.ge(0) if token_types is not None else None
            embeds = self._prepare_multimodal_embeddings(
                tokens,
                pixel_values=pixel_values,
                image_grid=image_grid,
                image_spans=image_spans,
                image_feature_indices=image_feature_indices,
                token_types=token_types,
            )

        tok_embeddings = self.tok_embeddings
        input_ids = tokens.detach().long()
        hidden = embeds if embeds is not None else (tok_embeddings(tokens) if tok_embeddings is not None else tokens)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        pre_mix = None
        last_layer = None
        for layer_key, layer in self.layers.items():
            layer_id = int(layer_key)
            if pre_mix is None:
                pre_mix = _make_identity_pre_mix(hidden, self.hc_mult)
            from . import attention as _v41_attention

            if _v41_attention._SHARED_INPUTS:
                shared = self.attention_context.resolve_shared(self._layer_shared_spec[layer_id])
                hidden, pre_mix, published = layer(
                    hidden,
                    input_ids,
                    attention_masks,
                    positions,
                    pre_mix=pre_mix,
                    image_mask=image_mask,
                    shared=shared,
                )
                if published is not None:
                    # Shared-inputs dataflow (compile path): the block returns
                    # the published tuple; the eager path publishes via the
                    # context side channel inside the block instead.
                    self.attention_context.absorb(layer_id, published)
            else:
                hidden, pre_mix = layer(
                    hidden,
                    input_ids,
                    attention_masks,
                    positions,
                    pre_mix=pre_mix,
                    image_mask=image_mask,
                )
            last_layer = layer

        if last_layer is None:
            raise RuntimeError("V4.1 model has no transformer layers")
        main_hidden = last_layer.collapse_pre_mix(hidden, pre_mix)  # pyrefly: ignore [not-callable]
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        output = main_hidden if self._skip_lm_head or self.lm_head is None else self.lm_head(main_hidden)
        return output.float()
