# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.components.checkpoint import AsyncMode, ModelWrapper

from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager

pytestmark = pytest.mark.smoke


def test_adapter_dcp_restores_npu_weights_and_routing_bias(npu_device, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(3, 3, dtype=torch.bfloat16), requires_grad=False)
            self.lora_a = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
            self.lora_b = torch.nn.Parameter(torch.ones(3, 2, dtype=torch.bfloat16))
            self.register_buffer("expert_bias_E", torch.zeros(3))

    class Checkpoint(DeepSeekV4PEFTCheckpointManager):
        ema_optimizer = None

    def manager(model):
        result = object.__new__(Checkpoint)
        result.stager = None
        result.verify_hash_manifest = False
        result.periodic_save_adapter_only = True
        result.save_training_state = False
        result.initial_load_path = str(tmp_path / "base")
        result.initial_load_in_hf = False
        result.initial_load_in_hf_quantized = False
        result.states = {"model": ModelWrapper(model)}
        return result

    original = Model().to(device=npu_device)
    source = manager(original)
    source.dcp_save(
        {key: value for key, value in original.state_dict().items() if "lora_" not in key},
        source.initial_load_path, AsyncMode.DISABLED,
    )
    with torch.no_grad():
        original.get_parameter("lora_a").fill_(2)
        original.get_parameter("lora_b").fill_(3)
        original.get_buffer("expert_bias_E").copy_(torch.tensor([-0.1, 0.2, -0.1]))
    checkpoint = str(tmp_path / "step-2")

    source.dcp_save(source._flattened_model_states_sd(), checkpoint, AsyncMode.DISABLED)
    restored = Model().to(device=npu_device)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    target = manager(restored)
    target.dcp_load(restored.state_dict(), checkpoint)

    saved_keys = dcp.FileSystemReader(checkpoint).read_metadata().state_dict_metadata.keys()
    assert set(saved_keys) == {"lora_a", "lora_b", "expert_bias_E"}
    for name, tensor in original.state_dict().items():
        actual = restored.state_dict()[name]
        assert actual.device == npu_device
        assert actual.dtype == tensor.dtype
        torch.testing.assert_close(actual.cpu(), tensor.cpu(), rtol=0, atol=0)
