# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.flex_shard import Owned
from torchtitan.models.common.linear import Linear

from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4.config_registry import (
    _dsv4_muon_profile,
    _dsv4_optimizer_config,
)
from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter, grouped_lora_class, linear_lora_class
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

_LORA_CONVERTER = DeepSeekV4LoRAConverter.Config(rank=4, alpha=8.0, rank_experts=4)


def _converted_debug_model() -> torch.nn.Module:
    spec = model_registry("debugmodel", num_mtp_layers=1, converters=[_LORA_CONVERTER])
    with torch.device("meta"):
        model = spec.model.build()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name)
    return model


@pytest.mark.parametrize("optimizer_name", ["AdamW", "Muon"])
def test_smoke_recipe_optimizer_selection(optimizer_name):
    from tests.integration_tests.lora_config import deepseek_v4_lora_training

    config = deepseek_v4_lora_training()
    config.optimizer.name = optimizer_name
    config.optimizer.materialize()
    expected = {"DistMuon", "AdamW"} if optimizer_name == "Muon" else {"AdamW"}
    assert {group.optimizer_name for group in config.optimizer.param_groups} == expected


def test_muon_group_is_non_empty_and_every_member_has_a_compute_layout():
    base_spec = model_registry("debugmodel", num_mtp_layers=1)
    profile = _dsv4_muon_profile(base_spec)
    compute_sharding_by_fqn = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]

    model = _converted_debug_model()
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen = [name for name, param in model.named_parameters() if not param.requires_grad]

    assert any(name.endswith(".weight") and "lora_" not in name for name in frozen)
    assert any("lora_" in name for name in trainable)

    config = _dsv4_optimizer_config(base_spec, lr=1e-4)
    config.name = "Muon"
    config.materialize()
    groups, _ = OptimizersContainer._build_param_groups(model, config.param_groups, {})
    assignment = {
        name: owner for owner, param_groups in groups.items() for group in param_groups for name in group["param_names"]
    }
    assert set(assignment) == set(trainable)
    assert not set(assignment).intersection(frozen)
    muon_fqns = [fqn for fqn, group in assignment.items() if group == "DistMuon"]
    adamw_fqns = [fqn for fqn, group in assignment.items() if group == "AdamW"]

    assert muon_fqns, "DistMuon param group is empty for the LoRA-converted model"
    assert all("lora_" in fqn for fqn in muon_fqns)
    missing = [fqn for fqn in muon_fqns if fqn not in compute_sharding_by_fqn]
    assert not missing, f"DistMuon parameters without a compute layout: {missing}"

    assert any(fqn.endswith(".w13_lora_b") for fqn in adamw_fqns)
    assert all(not fqn.endswith(".w13_lora_b") for fqn in muon_fqns)

    assert any(fqn.endswith(".lora_a.weight") for fqn in muon_fqns)
    assert any(fqn.endswith(".attention.wq_b.lora_b.weight") for fqn in muon_fqns)
    assert any(fqn.endswith(".w13_lora_a") for fqn in muon_fqns)
    assert any(fqn.endswith(".w2_lora_b") for fqn in muon_fqns)


@pytest.mark.parametrize("mesh_axis", ["dp_shard", "dp_shard_cp"], ids=["dp", "dp-cp"])
def test_dsv4_muon_policy_resolves_combined_cp_storage_mesh(tmp_path, mesh_axis):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    from torchtitan.distributed.flex_shard.dist_muon import _resolve_storage_to_compute_transition

    dist.init_process_group("gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=(mesh_axis,))
        converters = [DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)]
        spec = model_registry("debugmodel", converters=converters)
        profile = _dsv4_muon_profile(spec)
        layouts = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]
        with torch.random.fork_rng(devices=[]):
            module = spec.model.layers[0].attention.wq_a.build()
            parameter = module.lora_a.weight
        fqn = "layers.0.attention.wq_a.lora_a.weight"
        tensor = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
        transition = _resolve_storage_to_compute_transition(
            fqn,
            tensor,
            tensor.shape,
            None,
            layouts[fqn],
        )
        assert isinstance(transition.compute_sharding, Owned)
    finally:
        dist.destroy_process_group()


@pytest.fixture
def ao(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "experiments" / "torchao-npu"))
    pytest.importorskip("torchao_npu")
    module = importlib.import_module("interfaces.torchao_converter")
    monkeypatch.setattr(module, "_ensure_cann_ops_loaded", lambda: None)
    return module


def _lora_config(ao, kind):
    common = dict(rank=2, alpha=4.0, chunk_rows=2)
    if kind == "dense":
        return linear_lora_class(Linear).Config(in_features=4, out_features=8, bias=True, **common)
    if kind == "batched":
        return linear_lora_class(BatchedLinear).Config(n_heads=2, in_features=4, out_features=8, **common)
    return grouped_lora_class(ao.GroupedExperts).Config(dim=4, hidden_dim=8, num_experts=2, **common)


def _convert(ao, config, policy=None):
    return ao.NpuQuantizeConverter.Config(base_config=policy or ao._block_fp8_param_swap()).build().convert(config)


@pytest.mark.parametrize("kind", ["dense", "batched", "grouped"])
@pytest.mark.parametrize("quantize_first", [False, True])
def test_quantization_preserves_adapter_weights_and_gradients(ao, kind, quantize_first, monkeypatch):
    wrappers = importlib.import_module("torchao_npu.wrapper_tensors")
    block = importlib.import_module("torchao_npu.wrapper_tensors.block_mx_wrapper_tensor")
    calls = []

    def matmul(a, b, *_configs):
        calls.append("base")
        return torch.matmul(a, b)

    def grouped(a, b, offsets, *_configs):
        if _configs:
            calls.append("base")
        parts, start = [], 0
        for expert, end in enumerate(offsets.tolist()):
            parts.append(torch.mm(a[start:end], b[expert]))
            start = end
        return torch.cat(parts)

    monkeypatch.setattr(block, "to_block_mx_then_mm", matmul)
    monkeypatch.setattr(block, "to_block_mx_then_bmm", matmul)
    monkeypatch.setattr(block, "to_block_mx_then_grouped_mm", grouped)

    def grouped_mm(self, *, A, B_t, offs):
        if isinstance(B_t, block.BlockMXTrainingWeightWrapperTensor):
            return torch._grouped_mm(A, B_t, offs=offs)
        return grouped(A, B_t, offs)

    if quantize_first:
        base = {
            "dense": Linear.Config(in_features=4, out_features=8, bias=True),
            "batched": BatchedLinear.Config(n_heads=2, in_features=4, out_features=8),
            "grouped": ao.GroupedExperts.Config(dim=4, hidden_dim=8, num_experts=2),
        }[kind]
        config = (
            DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2, alpha=4)
            .build()
            ._make_lora_config(
                _convert(ao, base),
            )
        )
    else:
        config = _convert(ao, _lora_config(ao, kind))
    if kind == "grouped":
        monkeypatch.setattr(
            type(config)._owner,
            "_grouped_mm",
            grouped_mm,
        )
    module = deepcopy(config).build().to(torch.bfloat16)
    adapters, bases = [], []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.normal_(std=0.1)
            if "lora_" in name:
                adapters.append(parameter)
            else:
                # The model's parallelize hook freezes base weights after construction.
                parameter.requires_grad_(False)
                bases.append(parameter)
    assert adapters and all(type(p) is torch.nn.Parameter and p.requires_grad for p in adapters)
    assert all(isinstance(p, wrappers.BaseTrainingWeightWrapperTensor) == (p.ndim >= 2) for p in bases)
    x = torch.randn((5, 2, 4) if kind == "batched" else (5, 4), dtype=torch.bfloat16, requires_grad=True)
    args = (torch.tensor([3, 2]),) if kind == "grouped" else ()
    module(x, *args).float().square().sum().backward()
    assert len(calls) == (3 if kind == "grouped" else 1)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapters)
    assert all(p.grad is None for p in bases)


def test_non_lora_quantization_keeps_original_policy(ao):
    policy = ao._block_fp8_param_swap()
    config = _convert(ao, Linear.Config(in_features=4, out_features=8), policy)
    assert config._torchao_npu_config is policy


def test_parameter_filter_is_composed_without_mutating_shared_policy(ao):
    wrappers = importlib.import_module("torchao_npu.wrapper_tensors")
    policy = ao._block_fp8_param_swap()
    policy.params_filter_fn = lambda _parameter, name: name == "bias"
    original_filter = policy.params_filter_fn
    module = _convert(ao, _lora_config(ao, "dense"), policy).build()
    assert policy.params_filter_fn is original_filter
    assert all(not isinstance(p, wrappers.BaseTrainingWeightWrapperTensor) for p in module.parameters())


def test_module_replacement_policy_is_rejected_for_lora(ao):
    from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig

    with pytest.raises(ValueError, match="Float8DynamicActivationFloat8WeightConfig module replacement"):
        _convert(ao, _lora_config(ao, "dense"), Float8DynamicActivationFloat8WeightConfig())


@pytest.mark.parametrize("recipe", ["all_mxfp8", "mix", "all_block_fp8"])
@pytest.mark.parametrize("quantize_first", [False, True])
def test_registered_quantized_lora_accepts_compile_configuration(ao, recipe, quantize_first):
    from torchtitan.tools import utils

    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel
    from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter

    lora = DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)
    config = deepseek_v4_debugmodel(converters=[] if quantize_first else [lora])
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.recipe = recipe
    config.compile.enable = True
    config.extension.quantization.fsdp_prequantize = recipe != "all_mxfp8"
    config.model_spec = ao.apply_quantization_converter(
        config.model_spec,
        config.extension.quantization,
        model_compile_enabled=True,
    )
    if quantize_first:
        config.model_spec = replace(config.model_spec, model=lora.build().convert(config.model_spec.model))
    config.model_spec.model.update_from_config(config=config)
    with torch.device("meta"), utils.set_default_dtype(torch.bfloat16):
        model = config.model_spec.model.build()
    adapters = [p for name, p in model.named_parameters() if "lora_" in name]
    assert adapters and all(type(p) is torch.nn.Parameter and p.requires_grad for p in adapters)
