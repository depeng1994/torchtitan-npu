# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

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

from .metadata import CompressedVarlenMetadata, build_compressed_varlen_metadata
from .mhc import HcPost, HcPre
from .mtp import DeepSeekV4MTPDecoder, MTPBatch, prepare_mtp_batch
from .token_dispatcher import build_cp_plan


class DeepSeekV4TransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # DeepSeek-V4 has no non-MoE layer; ``moe`` is required (overrides the
        # inherited ``MoE.Config | None = None``).
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
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
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

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_deepseek_v4_nparams_and_flops(self, model, seq_len)

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config

        self.compress_ratios = tuple(cfg.compress_ratios) + tuple(
            layer.attention.compress_ratio for layer in cfg.mtp_layers
        )
        self.window_size = cfg.window_size
        self.block_size = cfg.block_size

        self._metadata_extension = cfg.metadata_extension.build()

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
        if mtp_batch is not None:
            extra_kwargs["mtp_batch"] = mtp_batch
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


def _deepseek_v4_attention_flops_per_token(attention, seq_len: int) -> int:
    """Return DSV4 sparse-attention operator FLOPs per token.

    This follows TorchTitan's MFU convention: forward plus backward matmul
    contractions are counted, while causal sparsity and activation-checkpoint
    recomputation are not. DSV4's structural sparsity is counted because the
    fused kernels never execute the omitted attention contractions.
    """
    inner_attention = attention.inner_attention
    attended_tokens = min(inner_attention.window_size, seq_len)
    ratio = attention.compress_ratio
    num_compressed_tokens = seq_len // ratio if ratio > 1 else 0

    if ratio == 4:
        attended_tokens += min(inner_attention.index_topk, num_compressed_tokens)
    elif ratio > 1:
        attended_tokens += num_compressed_tokens

    # QK and PV have the same head dimension in DSV4. The factor of 6 is the
    # same convention as TorchTitan's generic attention formula: one forward
    # contraction plus two backward contractions, with multiply-add counted as
    # two FLOPs.
    return 6 * attention.n_heads * (2 * attention.head_dim) * attended_tokens


def _deepseek_v4_indexer_flops_per_token(attention, seq_len: int) -> int:
    """Return LightningIndexer score-matmul FLOPs per token for CSA layers."""
    if attention.compress_ratio != 4 or attention.indexer is None:
        return 0

    indexer = attention.indexer
    num_compressed_tokens = seq_len // attention.compress_ratio
    # LightningIndexer has one QK score contraction (no value contraction).
    # Its AscendC training path computes the forward score and SLIG backward
    # gradients for Q/K, hence the same 6x multiply-add coefficient.
    return 6 * indexer.num_index_heads * indexer.index_head_dim * num_compressed_tokens


def get_deepseek_v4_nparams_and_flops(
    model_config: DeepSeekV4Model.Config,
    model: nn.Module,
    seq_len: int,
) -> tuple[int, int]:
    """Estimate DSV4 model FLOPs using TorchTitan's standard MFU convention."""
    first_attention = model_config.layers[0].attention
    head_dims = 2 * first_attention.head_dim

    # Reuse TorchTitan's MoE parameter accounting so routed experts contribute
    # only top-k active parameters while the router/shared experts/dense
    # parameters retain the upstream 6N convention.
    nparams, num_flops_per_token = get_moe_model_nparams_and_flops(
        model_config,
        model,
        first_attention.n_heads,
        head_dims,
        seq_len,
    )

    # The generic helper assumes full-sequence attention for each main layer.
    # Replace that term with DSV4's SWA/CSA/HCA structural sparsity, then add
    # the MTP attention layers (the generic helper does not include them).
    num_main_attention_layers = sum(
        1 for layer in model_config.layers if layer.attention is not None
    )
    num_flops_per_token -= (
        6
        * num_main_attention_layers
        * first_attention.n_heads
        * head_dims
        * seq_len
    )

    mtp_layer_configs = model_config.mtp_layers or []
    for layers in (model_config.layers, mtp_layer_configs):
        for layer in layers:
            attention = layer.attention
            num_flops_per_token += _deepseek_v4_attention_flops_per_token(
                attention, seq_len
            )
            num_flops_per_token += _deepseek_v4_indexer_flops_per_token(
                attention, seq_len
            )

    # The base parameter term counts one lm_head use. MTP applies that same
    # output projection once more for every prediction depth.
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Module):
        num_flops_per_token += 6 * len(mtp_layer_configs) * sum(
            param.numel() for param in lm_head.parameters()
        )

    # The generic 6N parameter term counts h_proj once per token. DSV4 MTP
    # applies h_proj to every mHC stream: [B, S, hc_mult, H]. Account for the
    # remaining hc_mult - 1 matrix applications here.
    model_mtp_layers = getattr(model, "mtp_layers", None)
    if model_mtp_layers is not None:
        for mtp_layer in model_mtp_layers:
            h_proj = getattr(mtp_layer, "h_proj", None)
            if isinstance(h_proj, nn.Module):
                num_flops_per_token += 6 * (model_config.hc_mult - 1) * sum(
                    param.numel() for param in h_proj.parameters()
                )

    return nparams, int(num_flops_per_token)


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
