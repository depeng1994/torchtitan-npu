# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
import pytest
import torch
import torch.nn.functional as F
from torchtitan.models.common import Linear
from torchtitan.protocols import Module
from torchtitan_npu.models import deepseek_v4 as dsv4
from torchtitan_npu.models.deepseek_v4 import lora, parallelize as dsv4_parallelize
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear
from tests.unit_tests.models.deepseek_v4.lora_test_utils import _build_model_config


@pytest.fixture
def parallelize_kwargs(tmp_path):
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()
    return {
        "parallel_dims": None,  # Distributed placement is mocked in these CPU tests.
        "training": config.training,
        "parallelism": config.parallelism,
        "compile_config": config.compile,
        "ac_config": config.activation_checkpoint,
        "dump_folder": str(tmp_path),
    }


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def test_dsv4_special_parameters_follow_model_construction_dtype():
    from torchtitan.tools import utils

    spec = dsv4.model_registry(
        "debugmodel", num_mtp_layers=1, converters=[lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)]
    )
    with torch.device("meta"), utils.set_default_dtype(torch.bfloat16):
        model = spec.model.build()

    adapter_dtypes = {parameter.dtype for name, parameter in model.named_parameters() if "lora_" in name}
    assert adapter_dtypes == {torch.bfloat16}


def test_converter_selects_every_dsv4_projection_except_indexer_including_mtp():
    base = dsv4.model_registry("debugmodel", num_mtp_layers=1).model
    expected = {
        fqn
        for fqn, config, _, _ in base.traverse(Module.Config, recurse=True)
        if isinstance(config, (Linear.Config, BatchedLinear.Config)) and ".indexer." not in fqn
    }
    spec = dsv4.model_registry(
        "debugmodel", num_mtp_layers=1, converters=[lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)]
    )
    config = spec.model
    assert isinstance(config, dsv4.DeepSeekV4Model.Config)
    actual = {
        fqn
        for fqn, projection, _, _ in config.traverse(Module.Config, recurse=True)
        if isinstance(projection, (Linear.Config, BatchedLinear.Config)) and hasattr(projection, "chunk_rows")
    }

    assert actual == expected
    assert hasattr(config.layers[0].moe.routed_experts.inner_experts, "chunk_rows")
    assert hasattr(config.mtp_layers[0].moe.routed_experts.inner_experts, "chunk_rows")


def test_converter_preserves_non_adapter_config_types():
    base = dsv4.model_registry("debugmodel", num_mtp_layers=1).model
    original_types = {name: type(config) for name, config, _, _ in base.traverse(Module.Config, recurse=True)}
    converted = lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2).build().convert(base)

    for name, config, _, _ in converted.traverse(Module.Config, recurse=True):
        if name and not hasattr(config, "chunk_rows"):
            assert type(config) is original_types[name], name


def test_converter_accepts_unquantized_linear_despite_class_name():
    class UnquantizedLinear(Linear):
        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            pass

    from torchtitan.config import derive

    config = _build_model_config()
    config.layers[0].attention.wq_a = derive(config.layers[0].attention.wq_a, UnquantizedLinear.Config)
    converted = lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2, target_modules=["attention.wq_a"]).build().convert(config)
    projection = converted.layers[0].attention.wq_a.build()
    projection.init_states()
    inputs = torch.ones(2, 32)
    torch.testing.assert_close(projection(inputs), F.linear(inputs, projection.weight))


def test_converter_preserves_upstream_quantization_config():
    pytest.importorskip("torchao")
    from torchtitan.components.quantization.float8 import Float8Linear
    from torchtitan.config import derive

    config = _build_model_config()
    config.layers[0].attention.wq_a = derive(config.layers[0].attention.wq_a, Float8Linear.Config)
    converted = lora.DeepSeekV4LoRAConverter.Config(target_modules=["attention.wq_a"]).build().convert(config)
    projection = converted.layers[0].attention.wq_a
    assert isinstance(projection, Float8Linear.Config)
    assert issubclass(type(projection)._owner, Float8Linear)
    assert isinstance(projection, lora.LoRAOptions)


def test_strict_matching_rejects_unmatched_targets():
    converter = lora.DeepSeekV4LoRAConverter.Config(
        target_modules=["attention.does_not_exist"], adapt_routed_experts=False
    )
    with pytest.raises(RuntimeError, match="did not match"):
        dsv4.model_registry("debugmodel", converters=[converter])


def test_non_strict_matching_warns_for_unmatched_targets(caplog):
    converter = lora.DeepSeekV4LoRAConverter.Config(
        target_modules=["attention.does_not_exist"], adapt_routed_experts=False, strict=False
    )
    with caplog.at_level("WARNING"):
        dsv4.model_registry("debugmodel", converters=[converter])
    assert "did not match" in caplog.text


def test_explicit_empty_targets_with_routed_experts_disabled_insert_no_adapters():
    converter = lora.DeepSeekV4LoRAConverter.Config(target_modules=[], adapt_routed_experts=False)
    spec = dsv4.model_registry("debugmodel", converters=[converter])
    with torch.device("meta"):
        model = spec.model.build()

    assert not [name for name, _ in model.named_parameters() if "lora_" in name]


@pytest.mark.parametrize("include_mtp", [False, True])
@pytest.mark.parametrize("num_mtp_layers", [0, 1], ids=["without-mtp", "with-mtp"])
def test_default_targets_follow_mtp_configuration(include_mtp, num_mtp_layers):
    specification = dsv4.model_registry(
        "debugmodel", num_mtp_layers=num_mtp_layers,
        converters=[lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2, include_mtp=include_mtp)],
    )
    config = specification.model
    converter = config.lora
    expected_mtp_targets = {"e_proj", "h_proj"} if num_mtp_layers and include_mtp else set()
    assert set(converter.target_modules) & {"e_proj", "h_proj"} == expected_mtp_targets
    assert "attention.wo_b" in converter.target_modules


def test_converter_freezes_base_without_example_wrapper(monkeypatch, parallelize_kwargs):
    def parallelize(model, **kwargs):
        # Sharding may replace parameters and reset requires_grad.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model
    monkeypatch.setattr(dsv4_parallelize, "parallelize_deepseekv3", parallelize)
    spec = dsv4.model_registry("debugmodel", converters=[lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)])
    with torch.device("meta"):
        model = spec.model.build()
    model = spec.parallelize_fn(model, **parallelize_kwargs)
    assert spec.post_optimizer_build_fn is not None
    parameters = dict(model.named_parameters())
    assert any("lora_" in name for name in parameters)
    assert all(parameter.requires_grad == ("lora_" in name) for name, parameter in parameters.items())


@pytest.mark.parametrize("factory,flavor", [
    ("deepseek_v4_flash", "deepseek_v4_flash"),
    ("deepseek_v4_debugmodel", "debugmodel"),
])
def test_standard_recipe_selects_lora_training_components(factory, flavor):
    from torchtitan_npu.models.deepseek_v4 import config_registry
    from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager

    base = getattr(config_registry, factory)()
    config = getattr(config_registry, factory)(
        converters=[lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)],
    )
    assert config.model_spec.flavor == flavor
    assert config.model_spec.model.lora.rank == 2
    assert config.model_spec.parallelize_fn is base.model_spec.parallelize_fn
    assert config.model_spec.post_optimizer_build_fn is base.model_spec.post_optimizer_build_fn
    assert isinstance(config.checkpoint, DeepSeekV4PEFTCheckpointManager.Config)
    assert not isinstance(base.checkpoint, DeepSeekV4PEFTCheckpointManager.Config)
    assert not hasattr(base.model_spec.model, "lora")
    assert config.training == base.training


def test_registry_model_forward_backward_freezes_base_and_differentiates_adapters(monkeypatch, parallelize_kwargs):
    # CPU UT bypasses distributed placement; EP/FSDP runs in the NPU integration case.
    monkeypatch.setattr(dsv4_parallelize, "parallelize_deepseekv3", lambda model, **kwargs: model)
    spec = dsv4.model_registry(
        "debugmodel", converters=[lora.DeepSeekV4LoRAConverter.Config(rank=8, rank_experts=8)],
    )
    with torch.device("cpu"):
        model = spec.model.build()
    model.init_states()
    model = spec.parallelize_fn(model, **parallelize_kwargs)
    inputs = torch.arange(128).reshape(1, 128)
    inputs, _, kwargs = model.build_attention_masks(
        inputs, inputs, {"positions": torch.arange(128).reshape(1, 128)},
    )
    output = model(inputs, **kwargs)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    adapters = {name: value for name, value in model.named_parameters() if "lora_" in name}
    assert adapters
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in adapters.values())
    assert all(value.grad is None for name, value in model.named_parameters() if "lora_" not in name)


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("selective_ac", [False, True])
def test_shared_parallelization_and_optimizer_hook_preserve_training_mode(monkeypatch, use_lora, selective_ac, parallelize_kwargs):
    from unittest.mock import Mock
    from torchtitan.distributed.activation_checkpoint import SelectiveAC

    original_ac = SelectiveAC.Config(preserve_rng_state=False) if selective_ac else parallelize_kwargs["ac_config"]
    parallelize_kwargs = {**parallelize_kwargs, "ac_config": original_ac}
    def parallelize(model, **kwargs):
        ac = kwargs["ac_config"]
        if use_lora and selective_ac:
            assert type(ac) is lora.LoRASelectiveAC.Config
            assert ac.preserve_rng_state is False
            assert ac.build().get_save_ops() == original_ac.build().get_save_ops() - {torch.ops.aten.bmm.default}
        else:
            assert ac is original_ac
        return model
    monkeypatch.setattr(dsv4_parallelize, "parallelize_deepseekv3", parallelize)
    register_hook = Mock()
    monkeypatch.setattr(dsv4, "register_moe_load_balancing_hook", register_hook)
    converters = [lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)] if use_lora else []
    spec = dsv4.model_registry("debugmodel", converters=converters)
    with torch.device("meta"):
        model = spec.model.build()
    model = spec.parallelize_fn(model, **parallelize_kwargs)
    spec.post_optimizer_build_fn(None, [model], None)
    assert register_hook.call_count == (0 if use_lora else 1)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (not use_lora or "lora_" in name)
