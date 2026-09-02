# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_debugmodel_uses_the_documented_per_layer_compression_ratios():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_spec = model_registry("debugmodel")

    assert model_spec.model.compress_ratios == (1, 1, 4, 128)
    assert len(model_spec.model.layers) == len(model_spec.model.compress_ratios)


def test_flash_ascendc_mfu_flops_meta_model():
    """The 4K Flash recipe's MFU estimator matches the full meta model.

    This mirrors TorchTitan Trainer startup: instantiate the complete model on
    the meta device, then ask the model config for parameter count and FLOPs per
    token.  The FLOPs oracle is the BF16 AscendC training path (forward +
    backward, no activation-checkpoint recompute): top-k active MoE experts,
    structural SWA/CSA/HCA sparsity, LightningIndexer + SLIG, one MTP depth,
    and the repeated MTP lm_head/h_proj applications.
    """
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry(
        "deepseek_v4_flash",
        num_mtp_layers=1,
    ).model

    with torch.device("meta"):
        model = model_config.build()

    nparams, flops_per_token = model_config.get_nparams_and_flops(
        model,
        seq_len=4096,
    )

    assert nparams == 290_942_278_866
    assert flops_per_token == 92_762_352_876
    assert flops_per_token * 4096 == 379_954_597_380_096
