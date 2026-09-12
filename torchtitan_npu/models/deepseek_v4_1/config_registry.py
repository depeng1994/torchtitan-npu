import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import TrainingConfig
from torchtitan_npu.extensions.profiler import CANNProfiler
from torchtitan_npu.extensions.trainer import TrainerEx

from .config import DeepSeekV41CropConfig, DeepSeekV41FullLayerConfig
from .model_registry import model_registry

# Imported lazily: `deepseek_v4.config_registry` and this package must not
# form an import cycle through the V4 test list.
from .vision_loader import DeepSeekV4SyntheticVisionDataLoader

# The Golden fixture image; override it via a config entry when training on real data.
DEFAULT_VISION_IMAGE_PATHS = ("tests/assets/dsv4_vit_test.jpeg",)


def _build_v41_trainer_config(flavor: str, crop: DeepSeekV41CropConfig) -> TrainerEx.Config:
    from torchtitan_npu.models.deepseek_v4.config_registry import _dsv4_optimizer_config

    # The golden path replaces F.cross_entropy globally and is the only loss the
    # frozen baseline was produced with, so it is not conditional.
    from torchtitan_npu.models.deepseek_v4.data import SyntheticTokenizer
    model_spec = model_registry(
        flavor,
    )
    if model_spec.model.dim != crop.hidden_size:
        raise ValueError("registered V4.1 model does not match hidden_size")
    if len(model_spec.model.layers) != crop.num_hidden_layers:
        raise ValueError("registered V4.1 model does not match the configured layer range")
    if model_spec.model.vision_encoder.num_layers != crop.vision_layers:
        raise ValueError("registered V4.1 model does not match vision_layers")
    if tuple(layer.attention.compress_ratio for layer in model_spec.model.layers) != crop.compress_ratios:
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
        dataloader=DeepSeekV4SyntheticVisionDataLoader.Config(
            vocab_size=129280,
            patch_count=64,
            image_span_start=8,
            image_paths=DEFAULT_VISION_IMAGE_PATHS,
            tokenizer_path=os.environ.get(
                "DSV4_TOKENIZER_PATH", "/data/tokenizer/dsv4_tokenizer"
            ),
            text=os.environ.get(
                "DSV4_VISION_TEXT", "Describe the image."
            ),
        ),
        optimizer=_dsv4_optimizer_config(model_spec, lr=1e-5),
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
        checkpoint=CheckpointManager.Config(enable=False, interval=10),
    )


def deepseek_v4_1_flash_30layers_16experts_vision() -> TrainerEx.Config:
    """Thirty continuous decoder layers for fast single-node validation."""
    return _build_v41_trainer_config(
        "deepseek_v4_1_flash_30layers_16experts_vision",
        DeepSeekV41CropConfig(
            hidden_size=5120,
            vision_layers=32,
        ),
    )


def deepseek_v4_1_flash_40layers_16experts_vision() -> TrainerEx.Config:
    """Full 40-layer decoder with the single-node 16-expert resource crop."""
    return _build_v41_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_vision",
        DeepSeekV41FullLayerConfig(
            hidden_size=5120,
            vision_layers=32,
        ),
    )
