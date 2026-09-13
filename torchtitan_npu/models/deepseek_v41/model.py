"""V4.1 model layer: the multimodal forward and the ViT on top of the V4 base.

The V4 base decoder stays decoder-only; everything V4.1 adds -- the image
tower, the marker embeddings, the image scatter and the context-parallel
splitting of image metadata -- lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh  # noqa: TC002
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.models.common.attention import AttentionMasksType  # noqa: TC002

from torchtitan_npu.models.deepseek_v4.golden import golden_enabled
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu.models.deepseek_v4.mtp import _make_identity_pre_mix

# Annotation-only names, kept importable at runtime on purpose: the trainer
# resolves ``Model.Config`` fields by name, so moving them behind TYPE_CHECKING
# would break introspection for no runtime gain.
from .vision import DeepSeekV4VisionEncoder, ImageMarkerEmbeddings  # noqa: TC001
from .vision_data import scatter_image_features


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
        if sample_idx < 0 or sample_idx >= hidden.shape[0] or start < 0 or length < 0:
            raise ValueError(f"invalid image span {(sample_idx, start, length)}")
        if start + length > hidden.shape[1] or length > visual.shape[1]:
            raise ValueError(
                f"image span {(sample_idx, start, length)} exceeds hidden/visual shapes "
                f"{tuple(hidden.shape)}/{tuple(visual.shape)}"
            )
        output[sample_idx, start : start + length] = visual[image_idx, :length].to(output.dtype)
    return output


class V41Model(DeepSeekV4Model):
    """DeepSeek-V4.1 backbone with the vision tower wired in."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        vision_encoder: DeepSeekV4VisionEncoder.Config | None = None
        image_marker_embeddings: ImageMarkerEmbeddings.Config | None = None

        def update_from_config(self, *, config, **kwargs):
            if config.parallelism.context_parallel_degree != 1:
                raise NotImplementedError(
                    f"DeepSeek V4.1 currently supports CP=1 only; got CP={config.parallelism.context_parallel_degree}"
                )
            super().update_from_config(config=config, **kwargs)

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = config.vision_encoder.build() if config.vision_encoder is not None else None
        self.image_marker_embeddings = (
            config.image_marker_embeddings.build() if config.image_marker_embeddings is not None else None
        )
        # V4.1-specific CSA2 plan/context construction lives here (not in the
        # V4 base) so deepseek_v41 never becomes an upstream dependency of
        # deepseek_v4. The V4 base only exposes the generic seam.
        if config.kv_source_layers is not None:
            from .attention import (
                V41AttentionContext,
                build_v41_compression_spec,
            )

            self._v41_plan = build_v41_compression_spec(
                layer_ids=tuple(range(config.n_layers)),
                ratios=self.compress_ratios[: config.n_layers],
                kv_source_layers=config.kv_source_layers,
                index_source_layers=config.index_source_layers or (),
                candidate_source_layer=(20 if config.candidate_source_layer is None else config.candidate_source_layer),
                candidate_topk_blocks=config.candidate_topk_blocks,
                candidate_block_size=config.candidate_block_size,
            )
            self._v41_context = V41AttentionContext.empty()
            for layer in self.layers.values():
                layer._v41_plan = self._v41_plan
                layer._v41_context = self._v41_context

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
        # The caller allows image_spans to be absent when image_feature_indices is
        # given, so the empty check must come before the shape handling.
        if image_spans is None or image_spans.numel() == 0:
            image_spans = None
        elif image_spans.ndim == 3:
            image_spans = image_spans.flatten(0, 1)
        safe_tokens = tokens.clamp_max(self.vocab_size - 1)
        hidden = self.tok_embeddings(safe_tokens)
        # Under CP, a rank can own no image-token slots while the visual
        # parameters remain replicated. Run the same ViT graph on every rank so
        # replicated vision weights receive identical gradients; scattering is
        # a no-op on ranks without local image slots.
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

    @staticmethod
    def _shard_image_spans_for_cp(
        image_spans: torch.Tensor,
        *,
        cp_mesh: DeviceMesh,
        seq_len: int,
        load_balancer_type: str | None,
    ) -> torch.Tensor:
        """Convert contiguous CP shard image spans to rank-local coordinates."""
        if image_spans.ndim == 3:
            image_spans = image_spans.flatten(0, 1)
        if image_spans.numel() == 0:
            return image_spans.reshape(0, 3)
        positions = cp_shard(cp_mesh, (torch.arange(seq_len),), None, load_balancer_type, 1)[0][0]
        if positions.numel() == 0:
            return image_spans.new_empty((0, 3))
        first, last = int(positions[0]), int(positions[-1])
        if (positions[1:] - positions[:-1]).abs().max().item() != 1:
            raise ValueError("image spans currently support contiguous or headtail CP sharding")
        rows = []
        for row in image_spans.tolist():
            image_id, start, length = (int(value) for value in row)
            lo, hi = start, start + length
            if hi <= first or lo > last:
                continue
            if not (lo >= first and hi - 1 <= last):
                raise ValueError("an image span is not contiguous after context-parallel sharding")
            rows.append((image_id, lo - first, length))
        if not rows:
            return image_spans.new_empty((0, 3))
        return image_spans.new_tensor(rows)

    def shard_extra_kwargs_for_cp(
        self,
        extra_kwargs: dict,
        *,
        cp_mesh: DeviceMesh,
        global_seq_len: int,
        load_balancer_type: str | None,
    ) -> None:
        """Shard the image spans and token-type tensors along the CP axis."""
        if "image_spans" in extra_kwargs:
            extra_kwargs["image_spans"] = self._shard_image_spans_for_cp(
                extra_kwargs["image_spans"],
                cp_mesh=cp_mesh,
                seq_len=global_seq_len,
                load_balancer_type=load_balancer_type,
            )
        metadata_tensors = tuple(
            extra_kwargs[name] for name in ("token_types", "image_feature_indices") if name in extra_kwargs
        )
        if metadata_tensors:
            local_metadata, _ = cp_shard(
                cp_mesh,
                metadata_tensors,
                None,
                load_balancer_type,
                1,
            )
            offset = 0
            for name in ("token_types", "image_feature_indices"):
                if name in extra_kwargs:
                    extra_kwargs[name] = local_metadata[offset]
                    offset += 1

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        mtp_batch=None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid: torch.Tensor | None = None,
        image_spans: torch.Tensor | None = None,
        image_feature_indices: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ):
        """V4.1 forward with Single-Pass mHC for vision and text-only batches.

        V4.1 runs all layers via ``forward_with_pre_mix`` — the pre_mix flows
        between sub-layers — and collapses at the decoder exit with
        ``collapse_pre_mix`` rather than the V4 classic ``hc_head``.
        """
        if self._v41_context is not None:
            self._v41_context.reset()

        if pixel_values is None:
            # A text-only batch must not inherit the image mask left on the
            # layers by a previous multimodal batch in the same process.
            for layer in self.layers.values():
                layer._v41_image_mask = None
            embeds = input_embeds
        else:
            if image_grid is None or (image_spans is None and image_feature_indices is None):
                raise ValueError("pixel_values requires image_grid and image_spans or image_feature_indices")
            image_mask = token_types.ge(0) if token_types is not None else None
            for layer in self.layers.values():
                layer._v41_image_mask = image_mask
            embeds = self._prepare_multimodal_embeddings(
                tokens,
                pixel_values=pixel_values,
                image_grid=image_grid,
                image_spans=image_spans,
                image_feature_indices=image_feature_indices,
                token_types=token_types,
            )

        # Single-Pass mHC main stack (V4.1): the pre_mix flows between
        # sub-layers and collapse_pre_mix replaces the V4 decoder hc_head.
        tok_embeddings = self.tok_embeddings
        input_ids = tokens.detach().long()
        hidden = embeds if embeds is not None else (tok_embeddings(tokens) if tok_embeddings is not None else tokens)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        pre_mix = None
        last_layer = None
        for layer in self.layers.values():
            if pre_mix is None:
                pre_mix = _make_identity_pre_mix(hidden, self.hc_mult)
            hidden, pre_mix = layer.forward_with_pre_mix(
                hidden,
                input_ids,
                attention_masks,
                positions,
                pre_mix=pre_mix,
            )
            last_layer = layer

        if last_layer is None:
            raise RuntimeError("V4.1 model has no transformer layers")
        main_hidden = last_layer.collapse_pre_mix(hidden, pre_mix)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        output = main_hidden if self._skip_lm_head or self.lm_head is None else self.lm_head(main_hidden)
        return output.float() if golden_enabled() else output
