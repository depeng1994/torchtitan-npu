# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import subprocess
import sys
from unittest.mock import Mock

import pytest

from torchtitan_npu.experiments.torchft import ft_manager as manager_module
from torchtitan_npu.extensions.torchft import trainer as trainer_module
from torchtitan_npu.patches.torchft import accelerator


def test_manager_uses_synchronous_quorum_with_legacy_training_hooks(monkeypatch):
    manager = Mock()
    monkeypatch.setattr(manager_module, "Manager", manager)
    monkeypatch.setattr(manager_module, "ProcessGroupHCCLEx", Mock())
    monkeypatch.setattr(manager_module.torchft.process_group, "ManagedProcessGroup", Mock())

    ft = manager_module.FTManagerEx(manager_module.FTManagerEx.Config())

    assert ft.use_async_quorum  # v0.3.0 hook/loss-sync compatibility only
    assert manager.call_args.kwargs["use_async_quorum"] is False


@pytest.mark.parametrize("disabled", ["enable", "enable_ft_dataloader_checkpoints"])
def test_trainer_requires_checkpoint_before_initializing_resources(monkeypatch, disabled):
    config = trainer_module.FaultTolerantTrainerEx.Config()
    assert config.checkpoint.enable
    assert config.checkpoint.enable_ft_dataloader_checkpoints
    setattr(config.checkpoint, disabled, False)
    initialize = Mock()
    monkeypatch.setattr(trainer_module.TrainerEx, "__init__", initialize)

    with pytest.raises(ValueError, match="requires checkpoint"):
        trainer_module.FaultTolerantTrainerEx(config)

    initialize.assert_not_called()


def test_trainer_initializes_ft_once_and_builds_npu_sdc(monkeypatch):
    from torchtitan_npu.extensions import trainer as npu_trainer_module

    config = trainer_module.FaultTolerantTrainerEx.Config()
    initialize_ft = Mock()
    initialize_sdc = Mock(return_value=object())

    def initialize_once(self, config):
        initialize_ft(config)
        self.model_parts = []
        self.gradient_accumulation_steps = 1

    monkeypatch.setattr(trainer_module.FaultTolerantTrainer, "__init__", initialize_once)
    monkeypatch.setattr(npu_trainer_module, "set_allow_hf32", Mock())
    monkeypatch.setattr(type(config.sdc), "build", initialize_sdc)

    trainer = trainer_module.FaultTolerantTrainerEx(config)

    initialize_ft.assert_called_once_with(config)
    npu_trainer_module.set_allow_hf32.assert_called_once_with(config.training.extension.allow_hf32)
    initialize_sdc.assert_called_once_with(trainer_config=config, model_parts=[], gradient_accumulation_steps=1)
    assert trainer._sdc is initialize_sdc.return_value
    assert isinstance(config.profiler, npu_trainer_module.CANNProfiler.Config)


def test_apply_rebinds_manager_synchronize(monkeypatch):
    import torchft.manager as manager
    import torchft.process_group as process_group

    def stale_synchronize():
        raise AssertionError("the pre-patch TorchFT synchronization was called")

    monkeypatch.setattr(manager, "synchronize", stale_synchronize)
    monkeypatch.setattr(process_group, "synchronize", stale_synchronize)
    monkeypatch.setattr(accelerator, "_INSTALLED", False)
    accelerator.apply()

    assert manager.synchronize is accelerator.synchronize
    assert process_group.synchronize is accelerator.synchronize


def test_plain_import_does_not_require_torchft_and_opt_in_explains_installation():
    program = """
import importlib.abc
import sys

class BlockTorchFT(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torchft' or fullname.startswith('torchft.'):
            raise ModuleNotFoundError('TorchFT is unavailable', name=fullname)

sys.meta_path.insert(0, BlockTorchFT())
import torchtitan_npu
assert not any(name == 'torchft' or name.startswith('torchft.') for name in sys.modules)
try:
    import torchtitan_npu.experiments.torchft
except ModuleNotFoundError as error:
    assert error.name == 'torchft'
    assert "pip install -e '.[torchft]'" in str(error)
else:
    raise AssertionError('Selecting TorchFT without its dependency must fail')
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=90)
