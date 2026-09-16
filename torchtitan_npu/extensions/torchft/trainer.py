# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit NPU TorchFT trainer configuration and normal training lifecycle."""

from dataclasses import dataclass, field

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.trainer import Trainer

from torchtitan_npu.experiments.torchft.ft_manager import FTManagerEx
from torchtitan_npu.extensions.trainer import TrainerEx


class FaultTolerantTrainerEx(TrainerEx, FaultTolerantTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):  # pyrefly: ignore [bad-override]
        optimizer: TorchFTOptimizersContainer.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=TorchFTOptimizersContainer.Config
        )
        fault_tolerance: FTManagerEx.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=FTManagerEx.Config
        )
        checkpoint: TorchFTCheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=lambda: TorchFTCheckpointManager.Config(enable=True, enable_ft_dataloader_checkpoints=True)
        )

        def __post_init__(self) -> None:
            # TorchFT's optimizer config does not have the NPU Muon materialize hook.
            Trainer.Config.__post_init__(self)

    def __init__(self, config: Config) -> None:
        if not config.checkpoint.enable or not config.checkpoint.enable_ft_dataloader_checkpoints:
            raise ValueError("NPU TorchFT requires checkpoint and per-replica dataloader checkpointing")
        super().__init__(config)

    def init_distributed(self):
        dist_utils.set_spmd_backend(self.config.parallelism.spmd_backend)
        return super().init_distributed()
