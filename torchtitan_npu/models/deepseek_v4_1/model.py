# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 text backbone.

Shape legend for this file:
    B = batch, L = sequence length, D = model dimension, hc = ``hc_mult`` residual
    branches.

A plain :class:`torchtitan.models.common.decoder.Decoder` — no MTP depths, no
context-parallel sharding (both rejected in ``update_from_config``).

The block carries the residual stream as ``hc`` parallel branches and threads the
cross-layer attention state (compressed KV, index keys, selected entries, student
logits, candidate pool) explicitly: an attention source layer returns the tensors it
produced, every other layer returns what it was handed.  That keeps the reuse structure
visible in the forward signature instead of in mutable module state, at the cost of a
longer tuple.

Single-Pass mHC means each sublayer collapses its input with the mixing coefficients
predicted by the *previous* sublayer, so the block returns the coefficients its
successor needs.  The stack has no learned output head: the model collapses the
branches with the last block's coefficients and feeds the result straight to the output
norm.

Packed documents are described by :class:`DeepSeekV41Metadata`: the only varlen
metadata is a document id per token, which both Attention Gym's operator (window
branch) and the indexer (entry-axis isolation) derive their masks from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchtitan.models.common.decoder import Decoder, TransformerBlock

from .mhc import HcPost, HcPre
from .vision import DeepSeekV41VisionEncoder, ImageMarkerEmbeddings  # noqa: TC001
from .vision_data import scatter_image_features

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE

    from .attention import Attention


@dataclass(frozen=True, kw_only=True, slots=True)
class DeepSeekV41Metadata:
    """Per-forward varlen metadata, built by :meth:`V41Model.get_attention_masks`.

    ``doc_ids_BL`` is the only field: a non-decreasing document index per token, built
    from the positions resetting to 0 at every packed segment start.  The same tensor
    serves both consumers, which is why they agree by construction:
    ``selected_attention`` applies ``doc_ids[q] == doc_ids[k]`` to its sliding-window
    branch, and the indexer applies the same equality one axis over, entry ``j``
    covering tokens ``[j * compress_ratio, (j + 1) * compress_ratio)``.  No
    ``cu_seqlens``-style ragged form is carried because the operators consume the
    per-token view directly.
    """

    doc_ids_BL: torch.Tensor  # noqa: N815


class DeepSeekV41TransformerBlock(TransformerBlock):
    """Transformer block with HC mixing around attention and the MoE feed-forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # Redeclared with the V4.1 types so config builders can reach the fields the
        # sharding and MTP helpers need.
        attention: Attention.Config  # pyrefly: ignore [bad-override]
        moe: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.moe_enabled = True
        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        self.moe = cfg.moe.build()
        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: DeepSeekV41Metadata | None,
        positions: torch.Tensor | None = None,
        *,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(x, pre_mix, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        ``pre_mix`` is the attention-input coefficient the next block must consume, and
        the shared attention tensors are this block's contribution to the chain.
        """
        residual = x
        x, attn_pre, post, comb = self.hc_attn_pre(x, pre_mix)
        x, cmp_k, idx_k, topk_indices, topk_scores, candidates = self.attention(
            self.attention_norm(x),
            positions,
            attention_masks,
            cmp_k=cmp_k,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, ffn_pre, post, comb = self.hc_ffn_pre(x, attn_pre)
        x = self.moe(self.ffn_norm(x), input_ids=input_ids, image_mask=image_mask)
        x = self.hc_post(x, residual, post, comb)
        return x, ffn_pre, cmp_k, idx_k, topk_indices, topk_scores, candidates

    def collapse_pre_mix(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        return self.hc_attn_pre.collapse(x, pre_mix)


def _vision_encoder_anchor(hidden: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
    """Keep the vision tower in the autograd graph when nothing is scattered.

    A batch can carry a tower whose features have no slot to go to (no spans and no
    feature indices, or empty spans).  DDP/FSDP require every parameter to take part in
    the backward pass, so the hidden states keep a zero-valued contribution from the
    tower instead of dropping it -- and this makes that intent explicit rather than a
    bare multiply-by-zero.
    """
    return hidden + visual.sum() * 0


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
        kv_source_layers: tuple[int, ...] = ()
        index_source_layers: tuple[int, ...] = ()
        candidate_source_layer: int = 20
        candidate_topk_blocks: int = 2048
        candidate_block_size: int = 8
        vision_encoder: DeepSeekV41VisionEncoder.Config | None = None
        image_marker_embeddings: ImageMarkerEmbeddings.Config | None = None

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
            compile_config = getattr(config, "compile", None)
            if compile_config is not None and getattr(compile_config, "enable", False):
                raise NotImplementedError(
                    "DeepSeek V4.1 does not support torch.compile yet; "
                    "CSA2 cross-layer state is currently an eager-only runtime contract"
                )
            Decoder.Config.update_from_config(self, config=config, **kwargs)

            if len(self.compress_ratios) != self.n_layers:
                raise ValueError(
                    f"compress_ratios must match n_layers ({self.n_layers}), got {len(self.compress_ratios)}"
                )

            if hasattr(config, "training"):
                from torchtitan.models.common.rope import RoPE

                seq_len = config.training.seq_len
                for _, rope_cfg, _, _ in self.traverse(RoPE.Config):
                    setattr(rope_cfg, "max_seq_len", seq_len)  # noqa: B010

            from .sharding import set_deepseek_v4_1_sharding_config

            set_deepseek_v4_1_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            from typing import cast

            from torchtitan.models.utils import get_moe_model_nparams_and_flops

            deepseek_v4_1_model = cast("V41Model", model)
            first_attention = self.layers[0].attention
            head_dims = 2 * first_attention.head_dim
            nparams, num_flops_per_token = get_moe_model_nparams_and_flops(
                self,
                deepseek_v4_1_model,
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
                        compressed_seq_len = min(compressed_seq_len, attention.indexer.index_topk)
                    num_flops_per_token += 6 * attention.n_heads * (2 * attention.head_dim) * compressed_seq_len
            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config
        self.hc_mult = cfg.hc_mult
        self.compress_ratios = tuple(cfg.compress_ratios)

        self.vision_encoder = config.vision_encoder.build() if config.vision_encoder is not None else None
        self.image_marker_embeddings = (
            config.image_marker_embeddings.build() if config.image_marker_embeddings is not None else None
        )

    def apply_activation_checkpointing_extensions(self, policy) -> None:
        """Apply the shared AC policy to the V4.1 vision blocks.

        ``_wrap_block`` is private because the pinned torchtitan exposes no public hook
        for wrapping a block that is not part of a ``Decoder`` layer list.
        """
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

    def get_attention_masks(  # pyrefly: ignore [bad-override]
        self, positions: torch.Tensor
    ) -> DeepSeekV41Metadata:
        """Build the only varlen metadata: a document id per token.

        ``selected_attention`` consumes it for the window branch, and the indexer derives
        its entry-axis ``doc_ids[:, ::compress_ratio]`` rule from it, so no ragged
        ``cu_seqlens`` form is needed.
        """
        if positions is None:
            raise ValueError("DeepSeek V4.1 requires positions to build its attention metadata")
        return DeepSeekV41Metadata(doc_ids_BL=torch.cumsum((positions == 0).to(torch.int32), dim=-1) - 1)

    def build_attention_masks(self, inputs, labels, extra_kwargs, *, cp_mesh=None, load_balancer_type=None):
        """Build the model-owned per-batch varlen metadata.

        The pinned trainer calls this hook for models that own their metadata instead of
        the Flex/Varlen ``get_attention_masks`` dispatch, so the model attaches its
        metadata here.
        """
        del load_balancer_type
        if cp_mesh is not None:
            raise NotImplementedError("DeepSeek V4.1 currently supports CP=1 only")
        extra_kwargs["attention_masks"] = self.get_attention_masks(extra_kwargs.get("positions"))
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
            return _vision_encoder_anchor(hidden, visual)
        return scatter_image_embeddings(hidden, visual, image_spans)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: DeepSeekV41Metadata | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid: torch.Tensor | None = None,
        image_spans: torch.Tensor | None = None,
        image_feature_indices: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ):
        """V4.1 forward with Single-Pass mHC for vision and text-only batches.

        The cross-layer attention state is threaded through the stack as ordinary local
        variables: an attention source layer returns the tensors it produced, and every
        other layer returns what it was handed.
        """
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

        pre_mix = HcPre.identity_pre_mix(hidden, self.hc_mult)
        cmp_k: torch.Tensor | None = None
        idx_k: torch.Tensor | None = None
        topk_indices: torch.Tensor | None = None
        topk_scores: torch.Tensor | None = None
        candidates: torch.Tensor | None = None
        last_layer = None
        for layer in self.layers.values():
            (
                hidden,
                pre_mix,
                cmp_k,
                idx_k,
                topk_indices,
                topk_scores,
                candidates,
            ) = layer(
                hidden,
                input_ids,
                attention_masks,
                positions,
                pre_mix=pre_mix,
                image_mask=image_mask,
                cmp_k=cmp_k,
                idx_k=idx_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
                candidates=candidates,
            )
            last_layer = layer

        if last_layer is None:
            raise RuntimeError("V4.1 model has no transformer layers")
        main_hidden = last_layer.collapse_pre_mix(hidden, pre_mix)  # pyrefly: ignore [not-callable]
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        output = main_hidden if self._skip_lm_head or self.lm_head is None else self.lm_head(main_hidden)
        return output.float()
