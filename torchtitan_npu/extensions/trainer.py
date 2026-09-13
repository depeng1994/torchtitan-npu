# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from torchtitan_npu.compile import setup_patterns
from torchtitan_npu.config import manager as config_manager
from torchtitan_npu.config.configs import (
    CompileConfig,
    ExtensionConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.config.converters import TrainerConfigConverter
from torchtitan_npu.distributed.utils import set_allow_hf32
from torchtitan_npu.extensions.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.components.sdc import SDC

from .profiler import CANNProfiler

_DECOMPOSED_ROPE_OVERRIDE = "torchtitan_npu.override.common.rope.decomposed"
"""Canonical RoPE override required by the Inductor pre-AOT patterns.

Upstream override resolution is order-independent and *conflicts* when two
overrides claim the same node, so the Inductor path must select exactly one
ComplexRoPE canonicalization: any ``asc_complex`` entry is replaced by
``decomposed`` (never co-imported).
"""


class TrainerEx(Trainer):
    """Base trainer for NPU-specific training features."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: CompileConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=CompileConfig,
        )
        extension: ExtensionConfig = field(default_factory=ExtensionConfig)
        optimizer: OptimizerConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=OptimizerConfig,
        )
        checkpoint: CheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=CheckpointManager.Config,
        )
        profiler: CANNProfiler.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=CANNProfiler.Config,
        )
        training: TrainingConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=TrainingConfig,
        )
        sdc: SDC.Config = field(default_factory=SDC.Config)

        def __post_init__(self) -> None:
            # ``slots=True`` dataclasses are recreated by the decorator, so a
            # zero-argument ``super()`` can retain the pre-decoration class cell.
            Trainer.Config.__post_init__(self)
            self.optimizer.materialize()
            if self.optimizer.name == "Muon" and (
                self.parallelism.tensor_parallel_degree > 1 or self.parallelism.pipeline_parallel_degree > 1
            ):
                raise ValueError(
                    "DeepSeek-V4 DistMuon requires "
                    "tensor_parallel_degree=1 and pipeline_parallel_degree=1; "
                    "TP _StridedShard and PP stage-local parameter groups are not admitted yet"
                )

    @staticmethod
    def _ensure_decomposed_rope(config: Config) -> None:
        """Canonicalize ComplexRoPE to decomposed for the Inductor path.

        The pre-AOT patterns match the decomposed (interleaved) RoPE graph, so
        ``--compile.backend=inductor`` must canonicalize ComplexRoPE to
        ``DecomposedComplexRoPE`` regardless of the user's eager override
        recipe.  Upstream override resolution is order-independent and raises
        on conflicting claims of the same node, so ``asc_complex`` must be
        *replaced* rather than co-imported.
        """
        imports = list(config.override.imports)

        def target(entry: object) -> str:
            return entry[0] if isinstance(entry, tuple) else str(entry)

        has_decomposed = any(target(e) == _DECOMPOSED_ROPE_OVERRIDE for e in imports)
        if has_decomposed:
            # Replace any competing ComplexRoPE canonicalization so the
            # Inductor path is never ambiguous about which override claims
            # ComplexRoPE.Config.
            imports = [e for e in imports if target(e) != "torchtitan_npu.override.common.rope.asc_complex"]
            config.override.imports = imports
            return

        # Drop the eager fused path and take decomposed for Inductor.
        imports = [e for e in imports if target(e) != "torchtitan_npu.override.common.rope.asc_complex"]
        imports.append(_DECOMPOSED_ROPE_OVERRIDE)
        config.override.imports = imports
        logger.info(
            "Inductor compile enabled: canonicalizing ComplexRoPE to %s for pre-AOT patterns",
            _DECOMPOSED_ROPE_OVERRIDE,
        )

    def __init__(self, config: Config):
        compile_extension = config.compile.extension
        if config.compile.enable and "model" in config.compile.components and config.compile.backend == "inductor":
            self._ensure_decomposed_rope(config)
            setup_patterns(
                enable_patterns=compile_extension.enable_patterns,
                pattern_blacklist=compile_extension.pattern_blacklist,
            )

        quantization_config = config.extension.quantization
        if quantization_config.enable_quantized_training:
            from interfaces.torchao_converter import apply_quantization_converter

            logger.info(
                "Applying TorchAO-NPU quantization recipe=%s before Trainer initialization",
                quantization_config.recipe,
            )
            model_compile_enabled = config.compile.enable and "model" in config.compile.components
            config.model_spec = apply_quantization_converter(
                config.model_spec,
                quantization_config,
                model_compile_enabled=model_compile_enabled,
            )

        set_allow_hf32(config.training.extension.allow_hf32)
        super().__init__(config)
        self._sdc = config.sdc.build(
            trainer_config=config,
            model_parts=self.model_parts,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

    def forward_backward_step(self, *args: Any, **kwargs: Any) -> Any:
        result = super().forward_backward_step(*args, **kwargs)
        # Advancing SDC state after a failed or partial step would corrupt its
        # accumulation window, so post-processing is intentionally success-only.
        self._sdc.finalize_sdc_step()
        return result


_trainer_config_converter = TrainerConfigConverter(
    target_type=TrainerEx.Config,
    component_types={
        "compile": CompileConfig,
        "optimizer": OptimizerConfig,
        "checkpoint": CheckpointManager.Config,
        "profiler": CANNProfiler.Config,
        "training": TrainingConfig,
    },
)
config_manager.register_config_converter(
    Trainer.Config,
    _trainer_config_converter,
)

# Also register for the original Trainer.Config in case the EMATrainer
# monkeypatch (patches/torchtitan/trainer.py) created a class-identity split.
# EMATrainer.__bases__[0] is the original Trainer captured before the patch.
_orig_trainer_config = Trainer.__bases__[0].Config
if _orig_trainer_config is not Trainer.Config:
    config_manager.register_config_converter(
        _orig_trainer_config,
        _trainer_config_converter,
    )
