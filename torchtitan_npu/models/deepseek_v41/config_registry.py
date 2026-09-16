# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 trainer config recipes (standalone; no V4 imports).

Operator selection follows the DSV4 pattern: the recipes build the
reference implementation; the launcher's default override set enables
the FUSION50-accepted fused stack, and USE_GOLDEN=1 selects the pure
reference path.
"""

import os
from typing import Any

from torch.distributed.tensor import Shard
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import MuonOptimizerProfile, OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.profiler import CANNProfiler
from torchtitan_npu.extensions.trainer import TrainerEx

from .cc12m_loader import DeepSeekV41Cc12mDataLoader
from .config import (
    DeepSeekV41CropConfig,
    DeepSeekV41DebugConfig,
    DeepSeekV41FullLayerConfig,
)
from .data import SyntheticTokenizer
from .model_registry import model_registry
from .vision_loader import DeepSeekV41SyntheticVisionDataLoader

# The Golden fixture image; override it via a config entry when training on real data.
DEFAULT_VISION_IMAGE_PATHS = ("tests/assets/dsv4_vit_test.jpeg",)


def _v41_muon_profile(model_spec: Any) -> MuonOptimizerProfile:
    """Build the V4.1-owned parameter and FlexShard policy for Muon.

    Same sharding policy as the DSV4 profile where the parameter spaces
    overlap (per-head ``wq_b``/``wo_a`` BlockShard, expert-sharded routed
    GEMMs, Owned for the rest).  V4.1 differences: no MTP depths, no
    ``hc_head``, no compressor APE; the indexer parameters receive no
    gradient in this recipe (no indexer loss) so they stay on the AdamW
    fallback group together with the vision tower and every 1-D parameter.
    """
    model_config = model_spec.model
    # FSDP folds the (dp_shard, cp) storage mesh into the single
    # ``dp_shard_cp`` axis when context parallelism is enabled.  Keep both
    # axis names to cover CP and non-CP runs.
    dense_dp_axes = (
        MeshAxisName.DP_SHARD.value,
        f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}",
    )
    owned = ComputeLayout(
        shardings_by_mesh_axis={axis: Owned() for axis in dense_dp_axes},
    )
    owned_attention_projections = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_projections = (*owned_attention_projections, "wq_b", "wo_a")
    expert_projections = ("w1", "w2", "w3")
    routed_expert_projections = ("w1_EFD", "w2_EDF", "w3_EFD")
    compressor_projections = ("wkv", "wgate")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    expert_sharding = ComputeLayout(
        shardings_by_mesh_axis={
            **{axis: Shard(0) for axis in dense_dp_axes},
            MeshAxisName.EFSDP.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        }
    )

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    bucket_configs: list[BucketConfig] = []
    for layer_id, layer_config in enumerate(model_config.layers):
        attention = layer_config.attention
        attention_shardings = {
            f"layers.{layer_id}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in owned_attention_projections.items()
        }
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head and
        # ``wo_a`` one [o_lora_rank, per_group_in] matrix per group, both
        # flattened on dim 0; BlockShard computes Muon per matrix.
        attention_shardings[f"layers.{layer_id}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={axis: BlockShard(dim=0, block_size=attention.head_dim) for axis in dense_dp_axes},
        )
        attention_shardings[f"layers.{layer_id}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_size=attention.wo_a.out_features) for axis in dense_dp_axes
            },
        )
        if getattr(attention, "compressor", None) is not None:
            # The ratio-1 source compressor is wkv-only (no gate).
            for projection in compressor_projections:
                if getattr(attention.compressor, projection, None) is not None:
                    attention_shardings[f"layers.{layer_id}.attention.compressor.{projection}.weight"] = owned
        attention_shardings[f"layers.{layer_id}.hc_attn_pre.hc_fn"] = owned
        dense_shardings = {
            f"layers.{layer_id}.moe.shared_experts.{projection}.weight": owned for projection in expert_projections
        }
        dense_shardings[f"layers.{layer_id}.moe.router.gate.weight"] = owned
        dense_shardings[f"layers.{layer_id}.hc_ffn_pre.hc_fn"] = owned
        routed_shardings = {
            f"layers.{layer_id}.moe.routed_experts.inner_experts.{projection}": expert_sharding
            for projection in routed_expert_projections
        }
        # Split each layer into three buckets: the routed-expert GEMMs are
        # by far the largest and get a bucket of their own so the
        # Newton-Schulz all-gather peak stays bounded on the 40-layer
        # FSDP8 shape (a single per-layer bucket OOMs at ~89% HBM).
        compute_sharding_by_fqn.update(attention_shardings)
        compute_sharding_by_fqn.update(dense_shardings)
        compute_sharding_by_fqn.update(routed_shardings)
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.attn", patterns=tuple(attention_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.dense", patterns=tuple(dense_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.routed", patterns=tuple(routed_shardings)))

    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.compressor\.(?:{'|'.join(compressor_projections)})\.weight|"
        rf"moe\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"moe\.routed_experts\.inner_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"moe\.router\.gate\.weight|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn"
        r")$"
    )
    return MuonOptimizerProfile(
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": tuple(bucket_configs),
            }
        },
    )


def _v41_optimizer_config(model_spec: Any, *, lr: float) -> OptimizerConfig:
    """Build the native-default V4.1 optimizer schema with a Muon profile.

    ``name`` stays ``native`` (pure AdamW) unless the user explicitly
    selects ``--optimizer.name Muon``; the profile only arms that CLI
    selection and does not alter the default recipe.
    """
    native = default_adamw(lr=lr, eps=1e-6)
    return OptimizerConfig(
        lr=lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_enable_nesterov=True,
        muon_ns_steps=10,
        muon_adjust_lr_fn="match_rms_adamw",
        param_groups=native.param_groups,
        implementation=native.implementation,
        optimizer_factory_kwargs_by_name=native.optimizer_factory_kwargs_by_name,
        _muon_profile=_v41_muon_profile(model_spec),
    )


def _build_v41_trainer_config(
    flavor: str,
    crop: DeepSeekV41CropConfig,
    *,
    dataloader: ParallelAwareDataloader.Config | None = None,
) -> TrainerEx.Config:
    # Operator selection follows the DSV4 pattern: the recipe builds the
    # reference implementation and the launcher's default override set
    # enables the FUSION50-accepted fused stack (rms_norm/rope/moe/mhc
    # post); USE_GOLDEN=1 keeps the pure reference path.  The historical
    # USE_GOLDEN gate was removed when those four adapters passed the
    # fixed-checkpoint 50-step acceptance.
    model_spec = model_registry(flavor)
    if model_spec.model.dim != crop.hidden_size:  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not match hidden_size")
    if len(model_spec.model.layers) != crop.num_hidden_layers:  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not match the configured layer range")
    if model_spec.model.vision_encoder.num_layers != crop.vision_layers:  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not match vision_layers")
    if (
        tuple(layer.attention.compress_ratio for layer in model_spec.model.layers)  # pyrefly: ignore [not-iterable]
        != crop.compress_ratios  # pyrefly: ignore [not-iterable]
    ):  # pyrefly: ignore [not-iterable]
        raise ValueError("registered V4.1 model does not match compression ratios")
    return TrainerEx.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=CANNProfiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        tokenizer=SyntheticTokenizer.Config(vocab_size=129280),
        dataloader=(
            DeepSeekV41SyntheticVisionDataLoader.Config(
                vocab_size=129280,
                patch_count=64,
                image_span_start=8,
                image_paths=DEFAULT_VISION_IMAGE_PATHS,
                # Deterministic synthetic token stream unless a tokenizer is
                # explicitly provided (the golden test suite points this at the
                # committed tests/assets/deepseek_v3 mini tokenizer).
                tokenizer_path=os.environ.get("DSV41_TOKENIZER_PATH", os.environ.get("DSV4_TOKENIZER_PATH")),
                text=os.environ.get("DSV41_VISION_TEXT", "Describe the image."),
            )
            if dataloader is None
            else dataloader
        ),
        optimizer=_v41_optimizer_config(model_spec, lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=crop.sequence_length,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=crop.fsdp_shard_degree,
            expert_parallel_degree=8,
            tensor_parallel_degree=1,
            context_parallel_degree=crop.context_parallel_degree,
            pipeline_parallel_degree=1,
            fsdp_reshard_after_forward="always",
            context_parallel_load_balancer="headtail",
        ),
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(enable=False, interval=10),  # pyrefly: ignore [bad-argument-type]
    )


def deepseek_v41_flash_30layers_16experts_vision() -> TrainerEx.Config:
    """Thirty continuous decoder layers for fast single-node validation."""
    return _build_v41_trainer_config(
        "deepseek_v41_flash_30layers_16experts_vision",
        DeepSeekV41CropConfig(
            hidden_size=5120,
            vision_layers=32,
        ),
    )


def deepseek_v41_flash_40layers_16experts_vision() -> TrainerEx.Config:
    """Full 40-layer decoder with the single-node 16-expert resource crop."""
    return _build_v41_trainer_config(
        "deepseek_v41_flash_40layers_16experts_vision",
        DeepSeekV41FullLayerConfig(
            hidden_size=5120,
            vision_layers=32,
        ),
    )


def _cc12m_dataloader_config() -> DeepSeekV41Cc12mDataLoader.Config:
    """The single real-data entry (CC12M captions; paths overridable)."""
    return DeepSeekV41Cc12mDataLoader.Config(
        manifest_path=os.environ.get(
            "CC12M_MANIFEST_PATH",
            "/data/p00465316/fused/datasets/cc12m/subset_8k/manifest.jsonl",
        ),
        data_dir=os.environ.get(
            "CC12M_DATA_DIR",
            "/data/p00465316/fused/datasets/cc12m/subset_8k",
        ),
        tokenizer_path=os.environ.get(
            "CC12M_TOKENIZER_PATH",
            "/data/p00465316/fused/dsv41_tokenizer",
        ),
    )


def deepseek_v41_flash_40layers_16experts_cc12m() -> TrainerEx.Config:
    """40-layer CC12M recipe — the single real-data training entry.

    3 steps verified on the current host (rc=0, 73.5% HBM) with the
    launcher-pinned ``TTNPU_DSA_ATTN_CHUNK=128``; the historical
    40-layer OOM records used the default chunk 256, whose backward
    workspace does not fit.  Do not run this shape without the pinned
    chunk.
    """
    return _build_v41_trainer_config(
        "deepseek_v41_flash_40layers_16experts_vision",
        DeepSeekV41FullLayerConfig(
            hidden_size=5120,
            vision_layers=32,
        ),
        dataloader=_cc12m_dataloader_config(),
    )


def deepseek_v41_debugmodel() -> TrainerEx.Config:
    """Reduced-width full-structure V4.1 shape for the golden trajectory tests.

    The real 40-layer compression/source structure and vision depth with
    debug widths, so the deterministic golden loss guard exercises every
    V4.1 code path quickly.
    """
    return _build_v41_trainer_config(
        "deepseek_v41_debugmodel",
        DeepSeekV41DebugConfig(
            hidden_size=512,
            vision_layers=32,
        ),
    )
