# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA training, selective freezing and checkpoint round trip on real NPUs."""

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.deepseek_v4.lora_config import deepseek_v4_lora_debugmodel
from torchtitan_npu.extensions.trainer import TrainerEx


def _cpu_tensor(tensor):
    return (tensor.to_local() if isinstance(tensor, DTensor) else tensor).detach().cpu().clone()


class LoRACheckpointTrainer(TrainerEx):
    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):
        pass

    def __init__(self, config):
        config.checkpoint.initial_load_path = str(Path(config.dump_folder) / "base")
        super().__init__(config)
        if not (Path(self.checkpointer.initial_load_path) / ".metadata").exists():
            # A native base checkpoint contains no adapters.
            base = {
                key: value for key, value in self.checkpointer.states["model"].state_dict().items()
                if "lora_" not in key
            }
            for key, tensor in base.items():
                if key.endswith("expert_bias_E"):
                    tensor.fill_(0.125)
            dcp.save(base, checkpoint_id=self.checkpointer.initial_load_path)

    def train_step(self, data_iterator):
        model = self.model_parts[0]
        frozen = {name: _cpu_tensor(p) for name, p in model.named_parameters() if not p.requires_grad}
        biases = {name: _cpu_tensor(b) for name, b in model.named_buffers() if name.endswith("expert_bias_E")}
        adapter = model.layers["0"].attention.wq_a.lora_b.weight
        before = _cpu_tensor(adapter)
        adapter_a = model.layers["0"].attention.wq_a.lora_a.weight
        before_a = _cpu_tensor(adapter_a)
        assert biases
        super().train_step(data_iterator)
        for name, expected in frozen.items():
            parameter = model.get_parameter(name)
            assert parameter.grad is None, name
            torch.testing.assert_close(_cpu_tensor(parameter), expected, rtol=0, atol=0)
        for name, expected in biases.items():
            torch.testing.assert_close(_cpu_tensor(model.get_buffer(name)), expected, rtol=0, atol=0)
        assert not torch.equal(_cpu_tensor(adapter), before)
        if adapter_a.requires_grad and torch.count_nonzero(before):
            assert not torch.equal(_cpu_tensor(adapter_a), before_a)

    def train(self):
        super().train()
        state = {
            name: (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu()
            for name, value in self.model_parts[0].state_dict().items() if "lora_" in name
        }
        expected = self.checkpointer.sd_adapter.to_peft(state)
        checkpoint = Path(self.checkpointer._create_checkpoint_id(self.step))
        exported = load_file(checkpoint / "adapter_model.safetensors")
        assert exported.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(exported[name], value.to(self.checkpointer.export_dtype), rtol=0, atol=0)


def deepseek_v4_lora_checkpoint():
    config = deepseek_v4_lora_debugmodel()
    return LoRACheckpointTrainer.Config(
        **{field.name: getattr(config, field.name) for field in fields(config) if field.init},
    )
