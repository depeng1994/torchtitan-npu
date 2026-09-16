# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU extensions for TorchTitan's graph trainer."""

from dataclasses import dataclass, field

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_npu.config import manager as config_manager
from torchtitan_npu.config.configs import OptimizerConfig, TrainingConfig
from torchtitan_npu.config.converters import TrainerConfigConverter
from torchtitan_npu.extensions.components.checkpoint import CheckpointManager

from .profiler import CANNProfiler
from .trainer import TrainerEx


class GraphTrainerEx(TrainerEx, GraphTrainer):
    """GraphTrainer with the NPU behavior provided by :class:`TrainerEx`."""

    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config, GraphTrainer.Config):
        compile: GraphTrainerCompileConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=GraphTrainerCompileConfig,
        )

    def __init__(self, config: Config) -> None:
        config.sdc.prepare_graph(config.compile)
        super().__init__(config)


config_manager.register_config_converter(
    GraphTrainer.Config,
    TrainerConfigConverter(
        target_type=GraphTrainerEx.Config,
        component_types={
            "optimizer": OptimizerConfig,
            "checkpoint": CheckpointManager.Config,
            "profiler": CANNProfiler.Config,
            "training": TrainingConfig,
        },
    ),
)
