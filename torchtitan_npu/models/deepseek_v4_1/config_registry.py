# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 trainer config recipes (standalone; no V4 imports).

The eager/reference operator path is what V4.1 runs: the AscendC fused
sparse-attention kernels do not yet accept the V4.1 ratio-1 shared-KV
contract, so the recipe selects the reference operators unconditionally.
"""

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.profiler import CANNProfiler
from torchtitan_npu.extensions.trainer import TrainerEx

from . import model_registry
from .data import SyntheticTokenizer
from .vision_loader import DeepSeekV41SyntheticVisionDataLoader

# The Golden fixture image; override it via a config entry when training on real data.
DEFAULT_VISION_IMAGE_PATHS = ("tests/assets/dsv4_vit_test.jpeg",)


def _document_alignment(model_spec) -> int:
    """Largest compression ratio the model pools with, ``1`` when it has none.

    Every row the vision loader emits is one document, so the row length has to
    be a multiple of the pooling ratio: the row edge is the document edge the
    compressed-attention pool must not straddle.
    """
    ratios = [ratio for ratio in model_spec.model.compress_ratios if ratio > 1]
    return max(ratios) if ratios else 1


def _v41_optimizer_config(model_spec, *, lr: float) -> OptimizerConfig:
    """Build the native-default V4.1 optimizer schema.

    V4.1 has no Muon profile (no MTP depths, no hc_head, no APE, no
    indexer compressors); the native AdamW recipe is the resolved
    contract of the frozen baseline.
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
    )


def _v41_trainer_config(
    flavor: str,
    *,
    seq_len: int = 512,
    fsdp_shard_degree: int = 8,
    context_parallel_degree: int = 1,
    expert_parallel_degree: int = 8,
    local_batch_size: int = 1,
    steps: int = 10,
) -> TrainerEx.Config:
    """Trainer recipe shared by the registered V4.1 single-node flavors.

    The flavors differ only in their model widths and layer count, which the
    model registry owns; the trainer shape below is the frozen
    V4.1 single-node resource crop (FSDP 8 / EP 8, eager/reference operators).
    """
    # The V4.1 layer-20+ ratio-1 layers produce a real shared/global KV, which
    # the AscendC sparse-attention path still rejects ("ratio-1 asc must not
    # receive compressed KV").  V4.1 therefore runs the eager/reference
    # operators unconditionally; selecting the AscendC kernels here would
    # silently mis-handle the ratio-1 container.
    model_spec = model_registry(flavor)
    # The registry validates the crop-level topology; this only pins that the
    # config the trainer is handed describes every layer it will build.
    if model_spec.model.n_layers != len(model_spec.model.layers):  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not describe every configured layer")

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
        dataloader=DeepSeekV41SyntheticVisionDataLoader.Config(
            vocab_size=129280,
            document_alignment=_document_alignment(model_spec),
            patch_count=64,
            image_span_start=8,
            image_paths=DEFAULT_VISION_IMAGE_PATHS,
            # Deterministic synthetic token stream unless a tokenizer is
            # explicitly provided (the integration case points this at the
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
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=fsdp_shard_degree,
            expert_parallel_degree=expert_parallel_degree,
            tensor_parallel_degree=1,
            context_parallel_degree=context_parallel_degree,
            pipeline_parallel_degree=1,
            fsdp_reshard_after_forward="always",
            context_parallel_load_balancer="headtail",
        ),
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(enable=False, interval=10),  # pyrefly: ignore [bad-argument-type]
    )


def deepseek_v4_1_flash_30layers_16experts_vision() -> TrainerEx.Config:
    """Thirty continuous decoder layers for fast single-node validation."""
    return _v41_trainer_config("deepseek_v4_1_flash_30layers_16experts_vision")


def deepseek_v4_1_flash_40layers_16experts_vision() -> TrainerEx.Config:
    """Full 40-layer decoder with the single-node 16-expert resource crop."""
    return _v41_trainer_config("deepseek_v4_1_flash_40layers_16experts_vision")


def deepseek_v4_1_debugmodel() -> TrainerEx.Config:
    """Reduced-width full-structure V4.1 shape for the CPU and integration tests.

    The real 40-layer compression/source structure and vision depth with
    debug widths, so the integration run exercises every
    V4.1 code path quickly.
    """
    return _v41_trainer_config("deepseek_v4_1_debugmodel")
