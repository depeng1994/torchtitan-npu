# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import get_moe_model_nparams_and_flops

from torchtitan_npu.models.common.metadata_extension import MetadataExtension

from .metadata import (
    CompressedVarlenMetadata,
    build_compressed_varlen_metadata,
    ensure_index_dense_masks,
)
from .mhc import HcPost, HcPre
from .mtp import (
    DeepSeekV4MTPDecoder,
    MTPBatch,
    prepare_mtp_batch,
)
from .token_dispatcher import build_cp_plan

if TYPE_CHECKING:
    from .compressor import Indexer
    from .mtp import DeepSeekV4MTPTransformerBlock


class DeepSeekV4TransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # DeepSeek-V4 has no non-MoE layer; ``moe`` is required (overrides the
        # inherited ``MoE.Config | None = None``).
        moe: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config
        layer_id: int = -1

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.moe_enabled = True
        self.layer_id = config.layer_id
        self._v41_plan = None
        self._v41_context = None

        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        self.moe = cfg.moe.build()

        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()

    def forward_with_pre_mix(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        pre_mix: torch.Tensor,
    ):
        residual = x
        x, post, comb, attn_pre = self.hc_attn_pre.forward_with_pre_mix(x, pre_mix)
        attention_kwargs = {}
        if self._v41_plan is not None:
            attention_kwargs = {
                "v41_layer_id": self.layer_id,
                "v41_plan": self._v41_plan,
                "v41_context": self._v41_context,
            }
        x = self.attention(
            self.attention_norm(x),
            attention_masks,
            positions,
            **attention_kwargs,
        )
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb, ffn_pre = self.hc_ffn_pre.forward_with_pre_mix(x, attn_pre)
        x = self.moe(self.ffn_norm(x), input_ids=input_ids, image_mask=getattr(self, "_v41_image_mask", None))
        x = self.hc_post(x, residual, post, comb)
        return x, ffn_pre

    def collapse_pre_mix(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the final mHC streams before the decoder norm and head."""
        return self.hc_attn_pre.collapse(x, pre_mix)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """Classic DeepSeek-V4 mHC forward.

        Each sub-block (attention / FFN) collapses the multi-stream state with
        its own ``pre`` mix inside ``HcPre``, and the decoder applies the
        standalone ``hc_head`` collapse after all layers.  V4.1's single-pass
        mHC (mix carried between sub-layers) goes through
        ``forward_with_pre_mix`` instead.
        """
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(self.attention_norm(x), attention_masks, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        x = self.moe(self.ffn_norm(x), input_ids=input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class DeepSeekV4Model(DeepSeekV4MTPDecoder):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4MTPDecoder.Config):
        vocab_size: int
        compress_ratios: tuple[int, ...]
        n_layers: int
        window_size: int
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        metadata_extension: MetadataExtension.Config = field(default_factory=MetadataExtension.Config)
        kv_source_layers: tuple[int, ...] | None = None
        index_source_layers: tuple[int, ...] | None = None
        candidate_source_layer: int | None = None
        candidate_topk_blocks: int = 2048
        candidate_block_size: int = 8

        def update_from_config(self, *, config, **kwargs):
            if hasattr(config, "training"):
                seq_len = config.training.seq_len
                for _, rope_cfg, _, _ in self.traverse(RoPE.Config):
                    setattr(rope_cfg, "max_seq_len", seq_len)  # noqa: B010

            DeepSeekV4MTPDecoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for i in range(self.n_layers):
                    layer_cfg = self.layers[i]
                    n_heads = layer_cfg.attention.n_heads
                    if n_heads % tp != 0:
                        raise ValueError(f"n_heads ({n_heads}) must be divisible by tp ({tp})")
                    n_groups = layer_cfg.attention.n_groups
                    if n_groups % tp != 0:
                        raise ValueError(f"n_groups ({n_groups}) must be divisible by tp ({tp})")

            # Context parallel is supported on the AscendC fused path only: the
            # model's build_attention_masks derives the per-rank dispatch
            # plan when the trainer passes the CP mesh; the model-dir
            # reference tier and the golden stay no-CP-only and raise there.
            from .sharding import set_deepseek_v4_sharding_config

            set_deepseek_v4_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4441
        # Temporary backport of the upstream DeepSeek-V4 sparse FLOPs
        # estimation, adapted to the pinned TorchTitan dependency.
        # Remove this implementation after the TorchTitan dependency includes
        # the PR.

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            deepseek_v4_model = cast("DeepSeekV4Model", model)
            first_attention = self.layers[0].attention
            head_dims = 2 * first_attention.head_dim
            nparams, num_flops_per_token = get_moe_model_nparams_and_flops(
                self,
                deepseek_v4_model,
                first_attention.n_heads,
                head_dims,
                seq_len,
            )

            num_flops_per_token -= 6 * len(self.layers) * first_attention.n_heads * head_dims * seq_len

            for layers in (self.layers, self.mtp_layers):
                for layer in layers:
                    attention = layer.attention
                    inner_attention = attention.inner_attention
                    num_flops_per_token += (
                        6 * attention.n_heads * (2 * attention.head_dim) * min(seq_len, inner_attention.window_size)
                    )

                    if attention.compress_ratio > 1:
                        compressed_seq_len = seq_len // attention.compress_ratio
                        if attention.compress_ratio == 4:
                            indexer = cast("Indexer.Config", attention.indexer)
                            num_flops_per_token += (
                                6 * indexer.num_index_heads * indexer.index_head_dim * compressed_seq_len
                            )
                            compressed_seq_len = min(compressed_seq_len, inner_attention.index_topk)
                        num_flops_per_token += 6 * attention.n_heads * (2 * attention.head_dim) * compressed_seq_len

            model_mtp_layers = deepseek_v4_model.mtp_layers
            if model_mtp_layers is not None:
                lm_head = cast("nn.Module", deepseek_v4_model.lm_head)
                num_flops_per_token += 6 * len(model_mtp_layers) * sum(param.numel() for param in lm_head.parameters())
                num_flops_per_token += (
                    6
                    * (self.hc_mult - 1)
                    * sum(
                        param.numel()
                        for mtp_layer in model_mtp_layers
                        for param in cast("DeepSeekV4MTPTransformerBlock", mtp_layer).h_proj.parameters()
                    )
                )

            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config

        self.compress_ratios = tuple(cfg.compress_ratios) + tuple(
            layer.attention.compress_ratio for layer in cfg.mtp_layers
        )
        self.window_size = cfg.window_size
        self.block_size = cfg.block_size
        self.vocab_size = cfg.vocab_size

        self._metadata_extension = cfg.metadata_extension.build()
        self._v41_plan = None
        self._v41_context = None
        if cfg.kv_source_layers is not None:
            from torchtitan_npu.models.deepseek_v4_1.attention import (
                V41AttentionContext,
                build_v41_compression_spec,
            )

            self._v41_plan = build_v41_compression_spec(
                layer_ids=tuple(range(cfg.n_layers)),
                ratios=self.compress_ratios[: cfg.n_layers],
                kv_source_layers=cfg.kv_source_layers,
                index_source_layers=cfg.index_source_layers or (),
                candidate_source_layer=(
                    20 if cfg.candidate_source_layer is None else cfg.candidate_source_layer
                ),
                candidate_topk_blocks=cfg.candidate_topk_blocks,
                candidate_block_size=cfg.candidate_block_size,
            )
            self._v41_context = V41AttentionContext.empty()
            for layer in self.layers.values():
                layer._v41_plan = self._v41_plan
                layer._v41_context = self._v41_context

    def shard_extra_kwargs_for_cp(
        self,
        extra_kwargs: dict,
        *,
        cp_mesh: DeviceMesh,
        global_seq_len: int,
        load_balancer_type: str | None,
    ) -> None:
        """Shard batch metadata a subclass carries through the CP split.

        The base decoder owns no such metadata, so this is a no-op; the V4.1
        multimodality subclass shards its image spans and token-type tensors.
        """

    def build_attention_masks(
        self,
        inputs,
        labels,
        extra_kwargs,
        *,
        cp_mesh: DeviceMesh | None = None,
        load_balancer_type: str | None = None,
    ):
        """The model-owned per-batch metadata construction (the single
        overridable mask-handling seam).

        One entry for both modes: the common contract
        (``build_compressed_varlen_metadata``) is always built; under CP
        ``_build_cp_metadata`` shards the inputs and derives the rank-local
        plan from the global context in-frame (no plan-time communication);
        the ``metadata_extension`` (e.g. the reference tier or the AscendC
        kernel metadata) runs last.
        """
        positions = extra_kwargs.get("positions")
        global_seq_len = None
        mtp_batch = None
        if cp_mesh is not None and self.mtp_layers is not None:
            mtp_batch = prepare_mtp_batch(
                inputs,
                labels,
                positions,
                len(self.mtp_layers),
            )
        masks = self.get_attention_masks(positions=positions)
        if not isinstance(masks, VarlenMetadata):
            raise TypeError(
                "DeepSeek-V4 compression requires a varlen stream (the "
                "inner attention is varlen-typed), got "
                f"{type(masks)}."
            )
        common = build_compressed_varlen_metadata(masks, self.compress_ratios)
        global_seq_len = common.seq_len
        if cp_mesh is not None:
            inputs, labels, positions, common, mtp_batch = self._build_cp_metadata(
                inputs,
                labels,
                positions,
                common,
                cp_mesh,
                load_balancer_type,
                mtp_batch,
            )
            extra_kwargs["positions"] = positions
            self.shard_extra_kwargs_for_cp(
                extra_kwargs,
                cp_mesh=cp_mesh,
                global_seq_len=global_seq_len,
                load_balancer_type=load_balancer_type,
            )
        if mtp_batch is not None:
            extra_kwargs["mtp_batch"] = mtp_batch
        common = ensure_index_dense_masks(common)
        if self._metadata_extension is not None:
            common = self._metadata_extension(common)
        extra_kwargs["attention_masks"] = common
        return inputs, labels, extra_kwargs

    def _build_cp_metadata(
        self,
        inputs,
        labels,
        positions,
        common,
        cp_mesh,
        load_balancer_type,
        mtp_batch,
    ):
        """The context-parallel metadata: shard the tensors via the generic
        path and derive the rank-local plan from the global context (the
        common metadata's varlen + the load-balancer permutation).

        Returns ``(inputs, labels, positions, metadata, mtp_batch)``."""
        seq_len = common.seq_len
        cp_size = cp_mesh.size(0)
        if seq_len % cp_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by cp_size ({cp_size}).")
        shard_len = seq_len // cp_size
        lb = _HeadTailLoadBalancer(seq_len, cp_size, cp_mesh.device_type) if load_balancer_type == "headtail" else None
        tensors = (inputs, labels, positions)
        if mtp_batch is not None:
            tensors += tuple(mtp_batch)
        tensors, _ = cp_shard(
            cp_mesh,
            tensors,
            None,
            load_balancer_type,
            1,
        )
        inputs, labels, positions = tensors[:3]
        if mtp_batch is not None:
            mtp_batch = MTPBatch(*tensors[3:])
        rank = cp_mesh.get_local_rank()
        cp_meta, plans, window = build_cp_plan(
            common.varlen,
            lb,
            rank=rank,
            cp_size=cp_size,
            shard_len=shard_len,
            window_size=self.window_size,
            ratios=sorted(set(self.compress_ratios)),
        )
        return (
            inputs,
            labels,
            positions,
            CompressedVarlenMetadata(varlen=cp_meta, plans=plans, window=window),
            mtp_batch,
        )


class GraphTrainerDeepSeekV4Model(DeepSeekV4Model):
    """DeepSeek V4 model variant for the GraphTrainer compilation path.

    Wraps ``init_states`` with ``disable_active_parametrization`` so that
    lazy-init parametrizations (e.g., RoPE freq buffers) are materialized
    before the FX tracer records the graph.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        pass

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            disable_active_parametrization,
        )

        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)
