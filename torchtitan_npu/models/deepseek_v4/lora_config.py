# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.trainer import Trainer

from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel, deepseek_v4_flash
from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter
from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager


def _lora_config(config: Trainer.Config, flavor: str) -> Trainer.Config:
    config.model_spec = model_registry(flavor, converters=[DeepSeekV4LoRAConverter.Config()])
    config.checkpoint = DeepSeekV4PEFTCheckpointManager.Config(enable=True, interval=10, save_training_state=True)
    return config


def deepseek_v4_lora() -> Trainer.Config:
    return _lora_config(deepseek_v4_flash(), "deepseek_v4_flash")


def deepseek_v4_lora_debugmodel() -> Trainer.Config:
    return _lora_config(deepseek_v4_debugmodel(), "debugmodel")
