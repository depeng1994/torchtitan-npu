# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 trainer config recipes (standalone; no V4 imports).

The golden/reference path is the only supported operator selection: the
config fails fast when ``USE_GOLDEN`` is not set, because the AscendC
fused kernels do not yet accept the V4.1 ratio-1 shared-KV contract.
"""

import os
from dataclasses import dataclass

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, ParamGroupConfig, default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.components.optimizer import HostSparseOptimizersContainer
from torchtitan_npu.extensions.profiler import CANNProfiler
from torchtitan_npu.extensions.trainer import TrainerEx

from .config import (
    DeepSeekV41CropConfig,
    DeepSeekV41DebugConfig,
    DeepSeekV41FullLayerConfig,
)
from .data import SyntheticTokenizer
from .model import V41Model
from .model_registry import model_registry
from .vision_loader import DeepSeekV41SyntheticVisionDataLoader

# The Golden fixture image; override it via a config entry when training on real data.
DEFAULT_VISION_IMAGE_PATHS = ("tests/assets/dsv4_vit_test.jpeg",)


@dataclass(kw_only=True, slots=True)
class DeepSeekV41TrainerConfig(TrainerEx.Config):
    """Expose the model switch while ModelSpec is suppressed from the CLI."""

    engram_enabled: bool = True

    def __post_init__(self) -> None:
        assert self.model_spec is not None and isinstance(self.model_spec.model, V41Model.Config)
        self.model_spec.model.engram_enabled = self.engram_enabled
        TrainerEx.Config.__post_init__(self)


def _golden_enabled() -> bool:
    return (
        os.getenv(
            "USE_GOLDEN",
            os.getenv("TORCHTITAN_NPU_VISION_GOLDEN", os.getenv("TORCHTITAN_NPU_GOLDEN_TRAINING", "0")),
        )
        == "1"
    )


def _v41_optimizer_config(model_spec, *, lr: float) -> OptimizerConfig:
    """Build the native-default V4.1 optimizer schema.

    V4.1 has no Muon profile (no MTP depths, no hc_head, no APE, no
    indexer compressors); the native AdamW recipe is the resolved
    contract of the frozen baseline.
    """
    native = default_adamw(lr=lr, eps=1e-6)
    has_engram = any(getattr(layer, "engram", None) is not None for layer in model_spec.model.layers)
    optimizer_type = HostSparseOptimizersContainer.Config if has_engram else OptimizerConfig
    groups = native.param_groups
    if has_engram:
        groups = [
            ParamGroupConfig(
                pattern=r".*\.engram\.table\.weight$",
                optimizer_name="SparseAdam",
                optimizer_kwargs={"lr": 5 * lr, "betas": (0.9, 0.95), "eps": 1e-6},
            ),
            *groups,
        ]
    return optimizer_type(
        lr=lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_enable_nesterov=True,
        muon_ns_steps=10,
        muon_adjust_lr_fn="match_rms_adamw",
        param_groups=groups,
        implementation=native.implementation,
        optimizer_factory_kwargs_by_name=native.optimizer_factory_kwargs_by_name,
    )


def _build_v41_trainer_config(flavor: str, crop: DeepSeekV41CropConfig) -> TrainerEx.Config:
    # The V4.1 layer-20+ ratio-1 layers produce a real shared/global KV, which
    # the AscendC sparse-attention path still rejects ("ratio-1 asc must not
    # receive compressed KV").  Until that kernel path is adapted, V4.1 is
    # golden/reference-only: fail fast instead of silently mis-selecting the
    # AscendC kernels.
    if not _golden_enabled():
        raise NotImplementedError(
            "DeepSeek V4.1 currently supports the golden/reference path only "
            "(USE_GOLDEN=1). The AscendC fused kernels do not yet accept the "
            "V4.1 ratio-1 shared-KV contract."
        )

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
    return DeepSeekV41TrainerConfig(
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
        dataloader=DeepSeekV41SyntheticVisionDataLoader.Config(
            vocab_size=129280,
            patch_count=64,
            image_span_start=8,
            image_paths=DEFAULT_VISION_IMAGE_PATHS,
            # Deterministic synthetic token stream unless a tokenizer is
            # explicitly provided (the golden test suite points this at the
            # committed tests/assets/deepseek_v3 mini tokenizer).
            tokenizer_path=os.environ.get("DSV41_TOKENIZER_PATH", os.environ.get("DSV4_TOKENIZER_PATH")),
            text=os.environ.get("DSV41_VISION_TEXT", "Describe the image."),
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


def deepseek_v41_debugmodel() -> TrainerEx.Config:
    """Reduced-width full-structure V4.1 model with Host Engram."""
    return _build_v41_trainer_config(
        "deepseek_v41_debugmodel",
        DeepSeekV41DebugConfig(
            hidden_size=512,
            vision_layers=32,
        ),
    )
