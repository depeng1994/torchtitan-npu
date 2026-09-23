# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA training, selective freezing and PEFT export on real NPUs."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from torchtitan.tools.logging import logger

from torchtitan_npu.extensions.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel
from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter


def _cpu_tensor(tensor):
    return (tensor.to_local() if isinstance(tensor, DTensor) else tensor).detach().cpu().clone()


class LoRATrainer(TrainerEx):
    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):
        pass

    def _optimizer_snapshot(self):
        return {
            name: _cpu_tensor(value)
            for name, value in self.optimizers.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

    def _optimizer_snapshot_path(self, step):
        return Path(self.config.dump_folder) / f"optimizer_state_step{step}_rank{dist.get_rank()}.pt"

    def train_step(self, data_iterator):
        checkpoint = self.config.checkpoint
        if checkpoint.save_training_state and (checkpoint.load_step or 0) > 0 and self.step == checkpoint.load_step + 1:
            expected = torch.load(
                self._optimizer_snapshot_path(checkpoint.load_step),
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(self._optimizer_snapshot(), expected, rtol=0, atol=0)
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

        if checkpoint.save_training_state and checkpoint.interval > 0 and self.step % checkpoint.interval == 0:
            state = self._optimizer_snapshot()
            fields = ("momentum_buffer", "exp_avg") if self.config.optimizer.name == "Muon" else ("exp_avg",)
            for field in fields:
                values = [value for name, value in state.items() if field in name]
                assert values and any(torch.count_nonzero(value) for value in values), field
            torch.save(state, self._optimizer_snapshot_path(self.step))

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
        with (checkpoint / "adapter_config.json").open(encoding="utf-8") as source:
            metadata = json.load(source)
        options = self.config.model_spec.model.lora
        assert metadata["r"] == options.rank
        assert metadata["lora_alpha"] == options.alpha
        target_parameters = set(metadata["target_parameters"])
        exported_targets = set()
        for name in exported:
            target = name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
            # PEFT nests the gate/up parameter adapter inside the down adapter.
            if target.endswith(".mlp.experts.base_layer"):
                target = target.removesuffix(".base_layer") + ".gate_up_proj"
            elif target.endswith(".mlp.experts"):
                target += ".down_proj"
            else:
                target += ".weight"
            exported_targets.add(target)
        assert target_parameters == exported_targets
        logger.info("PEFT export verified: %d adapter tensors, rank=%d, alpha=%s",
                    len(exported), metadata["r"], metadata["lora_alpha"])


def deepseek_v4_lora_training():
    config = deepseek_v4_debugmodel(converters=[DeepSeekV4LoRAConverter.Config()])
    config.checkpoint.enable = True
    config.checkpoint.interval = 10000
    config.checkpoint.periodic_save_adapter_only = False
    return LoRATrainer.Config(
        **{field.name: getattr(config, field.name) for field in fields(config) if field.init},
    )
