# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import spmd_types as spmd
import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import distributed as dist, multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.models.common import Linear, decoder_sharding, moe
from torchtitan.protocols import Module, sharding

from torchtitan_npu.models import deepseek_v4 as dsv4
from torchtitan_npu.models.deepseek_v4 import lora, peft, state_dict_adapter
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def _assert_gradients_match(actual, expected, variables, *, rtol=1e-5, atol=1e-6):
    upstream = torch.linspace(-0.7, 0.9, actual.numel()).reshape_as(actual)
    actual_grads = torch.autograd.grad(actual, variables, upstream)
    expected_grads = torch.autograd.grad(expected, variables, upstream)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)


def _grouped_mm(*, A, B_t, offs):
    outputs = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        outputs.append(A[start:end] @ B_t[expert])
        start = end
    return torch.cat(outputs) if outputs else A.new_empty((0, B_t.shape[-1]))


_GroupedLoRAExperts = lora._get_grouped_lora_cls(moe.GroupedExperts)


class _CpuGroupedLoRAExperts(_GroupedLoRAExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(_GroupedLoRAExperts.Config):
        pass

    def _grouped_mm(self, *, A, B_t, offs):
        return _grouped_mm(A=A, B_t=B_t, offs=offs)


def _build_cpu_grouped_lora_module() -> _CpuGroupedLoRAExperts:
    config = _CpuGroupedLoRAExperts.Config(
        dim=4, hidden_dim=6, num_experts=2, rank=2, alpha=4.0, chunk_rows=2,
        param_init=dict.fromkeys(
            ("w1_EFD", "w2_EDF", "w3_EFD", "w13_lora_a", "w13_lora_b", "w2_lora_a", "w2_lora_b"), torch.nn.init.zeros_
        ),
    )
    module = config.build()
    for parameter in (module.w1_EFD, module.w2_EDF, module.w3_EFD):
        parameter.requires_grad_(False)
    module.init_states()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
    return module


def _per_expert_lora_reference(module, inputs, counts, scores, limit):
    expected_parts = []
    start = 0
    scale = 2.0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        x = inputs[start:end].bfloat16()
        w13_hidden = x @ module.w13_lora_a[expert].bfloat16().T
        gate_delta = w13_hidden @ module.w13_lora_b[expert, :, 0].bfloat16().T
        up_delta = w13_hidden @ module.w13_lora_b[expert, :, 1].bfloat16().T
        gate = x @ module.w1_EFD[expert].bfloat16().T
        gate = gate + scale * gate_delta
        up = x @ module.w3_EFD[expert].bfloat16().T
        up = up + scale * up_delta
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        hidden = F.silu(gate) * up
        hidden = (hidden.float() * scores[start:end, None]).to(hidden.dtype)
        output = hidden @ module.w2_EDF[expert].bfloat16().T
        output = output + scale * (
            (hidden @ module.w2_lora_a[expert].bfloat16().T) @ module.w2_lora_b[expert].bfloat16().T
        )
        expected_parts.append(output.float())
        start = end
    return torch.cat(expected_parts)


def test_replicated_base_keeps_both_lora_factors_replicated():
    replicated = decoder_sharding.dense_param_placement(tp=spmd.R)
    base = sharding.ShardingConfig(state_shardings={"weight": replicated})

    lora_a, lora_b = lora._dsv4_lora_adapter_sharding(base)

    assert lora_a is not None and lora_a.state_shardings["weight"] == replicated
    assert lora_b is not None and lora_b.state_shardings["weight"] == replicated


def test_dsv4_special_parameters_follow_model_construction_dtype():
    from torchtitan.tools import utils

    spec = dsv4.model_registry(
        "debugmodel", num_mtp_layers=1, converters=[lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)]
    )
    with torch.device("meta"), utils.set_default_dtype(torch.bfloat16):
        model = spec.model.build()

    adapter_dtypes = {parameter.dtype for name, parameter in model.named_parameters() if "lora_" in name}
    assert adapter_dtypes == {torch.bfloat16}


def test_chunked_dense_lora_matches_independent_formula_and_backpropagates():
    cls = lora._get_linear_lora_cls(Linear)
    assert cls is lora._get_linear_lora_cls(Linear)
    module = cls.Config(in_features=5, out_features=7, bias=False, rank=3, alpha=6.0, chunk_rows=2).build()
    module.weight.requires_grad_(False)
    module.init_states()
    with torch.no_grad():
        module.lora_b.weight.normal_(std=0.1)

    inputs = torch.randn(2, 3, 5, requires_grad=True)
    actual = module(inputs)
    expected = F.linear(inputs, module.weight) + 2.0 * F.linear(
        F.linear(inputs, module.lora_a.weight), module.lora_b.weight
    )

    torch.testing.assert_close(actual, expected)
    _assert_gradients_match(actual, expected, (inputs, module.lora_a.weight, module.lora_b.weight))
    assert not module.weight.requires_grad


def test_batched_lora_matches_shared_a_per_head_b_and_backpropagates():
    cls = lora._get_linear_lora_cls(BatchedLinear)
    assert cls is lora._get_linear_lora_cls(BatchedLinear)
    module = cls.Config(
        n_heads=3, in_features=5, out_features=4, rank=2, alpha=4.0, chunk_rows=2,
        param_init={"weight": torch.nn.init.zeros_},
    ).build()
    module.weight.requires_grad_(False)
    module.init_states()
    with torch.no_grad():
        module.weight.normal_(std=0.1)
        module.lora_b.weight.normal_(std=0.1)

    inputs = torch.randn(2, 3, 3, 5, requires_grad=True)
    actual = module(inputs)
    base_weight = module.weight.view(3, 4, 5)
    adapter_b = module.lora_b.weight.view(3, 4, 2)
    expected = torch.einsum("...hd,hod->...ho", inputs, base_weight)
    hidden = F.linear(inputs, module.lora_a.weight)
    expected = expected + 2.0 * torch.einsum("...hr,hor->...ho", hidden, adapter_b)

    torch.testing.assert_close(actual, expected)
    _assert_gradients_match(actual, expected, (inputs, module.lora_a.weight, module.lora_b.weight))
    assert not module.weight.requires_grad


@pytest.mark.parametrize("limit", [0.0, 0.05], ids=["unclamped", "clamped"])
def test_grouped_lora_output_and_gradients_match_per_expert_reference(limit):
    assert _GroupedLoRAExperts is lora._get_grouped_lora_cls(moe.GroupedExperts)
    module = _build_cpu_grouped_lora_module()
    module.swiglu_limit = limit
    counts = torch.tensor([2, 3], dtype=torch.int64)
    inputs = torch.randn(5, 4, requires_grad=True)
    scores = torch.linspace(0.2, 1.4, 5, requires_grad=True)

    actual = module(inputs, counts, routed_scores_R=scores)
    expected = _per_expert_lora_reference(module, inputs, counts, scores, limit)
    variables = (inputs, scores, module.w13_lora_a, module.w13_lora_b, module.w2_lora_a, module.w2_lora_b)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    _assert_gradients_match(actual, expected, variables, rtol=2e-2, atol=2e-3)
    assert all(not parameter.requires_grad for name, parameter in module.named_parameters() if "lora_" not in name)


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


def test_converter_rejects_upstream_quantization_config():
    pytest.importorskip("torchao")
    from torchtitan.components.quantization.float8 import Float8Linear
    from torchtitan.config import derive

    config = _build_model_config()
    config.layers[0].attention.wq_a = derive(config.layers[0].attention.wq_a, Float8Linear.Config)
    with pytest.raises(NotImplementedError, match="unquantized base"):
        lora.DeepSeekV4LoRAConverter.Config().build().convert(config)


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


@pytest.mark.parametrize("freeze_lora_a", [False, True], ids=["ab-training", "b-only-training"])
def test_dense_lora_zero_init_preserves_output_and_updates_only_adapter(freeze_lora_a):
    converter = lora.DeepSeekV4LoRAConverter.Config(
        target_modules=["attention.wo_b"], adapt_routed_experts=False, strict=False
    )
    model_config = dsv4.model_registry("debugmodel", converters=[converter]).model
    configs = model_config.traverse(Linear.Config, recurse=True)
    target_config = next(config for fqn, config, _, _ in configs if fqn == "layers.0.attention.wo_b")
    target = target_config.build()
    target.weight.requires_grad_(False)
    target.init_states()
    inputs = torch.linspace(-1.0, 1.0, steps=2 * target.in_features).reshape(2, target.in_features)

    expected = torch.nn.functional.linear(inputs, target.weight)
    actual = target(inputs)

    assert torch.count_nonzero(target.lora_a.weight) > 0
    assert torch.count_nonzero(target.lora_b.weight) == 0
    torch.testing.assert_close(actual, expected)

    target.lora_a.weight.requires_grad_(not freeze_lora_a)
    adapter_a_before = target.lora_a.weight.detach().clone()
    trainable = [parameter for parameter in target.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.1, foreach=False)
    base_before = target.weight.detach().clone()
    adapter_before = target.lora_b.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        target(inputs).square().mean().backward()

        assert all(parameter.grad is not None for parameter in trainable)
        assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
        assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in trainable)
        if freeze_lora_a:
            assert target.lora_a.weight.grad is None

        optimizer.step()

        assert torch.equal(target.weight, base_before)
        assert not torch.equal(target.lora_b.weight, adapter_before)
        if freeze_lora_a:
            assert torch.equal(target.lora_a.weight, adapter_a_before)
    if not freeze_lora_a:
        assert not torch.equal(target.lora_a.weight, adapter_a_before)


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


def _make_checkpoint_fixture(tmp_path, save_training_state, checkpoint_format, wrapper_scope, trainable_base):
    class RoutingState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias_E", torch.zeros(3))
            self.register_buffer("tokens_per_expert_E", torch.zeros(3), persistent=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(3, 3), requires_grad=False)
            if trainable_base:
                self.base.requires_grad_(True)
                self.frozen_extra = torch.nn.Parameter(torch.full((2,), 7.0), requires_grad=False)
            self.lora_a = torch.nn.Parameter(torch.ones(2, 3))
            self.lora_b = torch.nn.Parameter(torch.ones(3, 2))
            self.moe = checkpoint_wrapper(RoutingState()) if wrapper_scope == "submodule" else RoutingState()

    def build_model():
        model = Model()
        return checkpoint_wrapper(model) if wrapper_scope == "model" else model

    class StepState:
        def __init__(self, step):
            self.step = step

        def state_dict(self):
            return {"step": torch.tensor(self.step)}

        def load_state_dict(self, state):
            self.step = int(state["step"])

    def manager(model, step):
        result = object.__new__(peft.DeepSeekV4PEFTCheckpointManager)
        result.ema_optimizer = None
        result.stager = None
        result.verify_hash_manifest = False
        result.periodic_save_adapter_only = True
        result.save_training_state = save_training_state
        result.initial_load_path = str(tmp_path / "base")
        result.initial_load_in_hf = False
        result.initial_load_in_hf_quantized = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], foreach=False)
        if step or save_training_state or checkpoint_format == "full_model":
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, float(step))
            optimizer.step()
            optimizer.zero_grad()
        result.states = {"model": ModelWrapper(model), "optimizer": optimizer, "train_state": StepState(step)}
        return result

    return build_model, manager


def _check_periodic_dcp_round_trip(
    tmp_path, save_training_state, checkpoint_format, wrapper_scope, trainable_base=False
):
    build_model, manager = _make_checkpoint_fixture(
        tmp_path, save_training_state, checkpoint_format, wrapper_scope, trainable_base
    )
    original = build_model()
    source = manager(original, 9)
    dcp.save(
        {key: value for key, value in original.state_dict().items() if "lora_" not in key},
        checkpoint_id=source.initial_load_path,
    )
    with torch.no_grad():
        if trainable_base:
            original.get_parameter("base").fill_(5)
        original.get_parameter("lora_a").fill_(2)
        original.get_parameter("lora_b").fill_(3)
        original.state_dict()["moe.expert_bias_E"].copy_(torch.tensor([-0.1, 0.2, -0.1]))
    selected = source._flattened_model_states_sd()
    expected_keys = {"lora_a", "lora_b", "moe.expert_bias_E"}
    if trainable_base:
        expected_keys.add("base")
    if save_training_state:
        expected_keys |= {"optimizer", "train_state"}
    assert set(selected) == expected_keys
    if checkpoint_format == "legacy_adapter":
        selected.pop("moe.expert_bias_E")
    elif checkpoint_format == "full_model":
        selected.update(original.state_dict())
        selected.update({key: value for key, value in source.states.items() if key != "model"})
    checkpoint = str(tmp_path / "step-9")
    dcp.save(selected, checkpoint_id=checkpoint)
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    assert ("base" in metadata.state_dict_metadata) == (trainable_base or checkpoint_format == "full_model")
    assert "moe.tokens_per_expert_E" not in metadata.state_dict_metadata

    restored = build_model()
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    target = manager(restored, 0)
    initial_adapters = {key: value.clone() for key, value in restored.state_dict().items() if "lora_" in key}
    target.dcp_load(restored.state_dict(), target.initial_load_path)
    for key, value in initial_adapters.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    target.dcp_load(target._flattened_model_states_sd(target.states), checkpoint)

    for key, tensor in original.state_dict().items():
        expected = (
            torch.zeros_like(tensor) if key == "moe.expert_bias_E" and checkpoint_format == "legacy_adapter" else tensor
        )
        assert torch.equal(expected, restored.state_dict()[key]), key
    restores_training_state = save_training_state or checkpoint_format == "full_model"
    assert target.states["train_state"].step == (9 if restores_training_state else 0)
    restored_optimizer = target.states["optimizer"].state_dict()["state"]
    if restores_training_state:
        for parameter_id, state in source.states["optimizer"].state_dict()["state"].items():
            for key, tensor in state.items():
                assert torch.equal(tensor, restored_optimizer[parameter_id][key])
        for model, checkpoint_manager in ((original, source), (restored, target)):
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, 0.25)
            checkpoint_manager.states["optimizer"].step()
        for expected, actual in zip(original.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert not restored_optimizer


@pytest.mark.parametrize("save_training_state", [False, True], ids=["weights-restart", "training-resume"])
@pytest.mark.parametrize("checkpoint_format", ["adapter_buffers", "legacy_adapter", "full_model"])
@pytest.mark.parametrize("wrapper_scope", ["plain", "model", "submodule"])
def test_periodic_dcp_round_trip_keeps_base_out_and_restores_selected_state(
    tmp_path, save_training_state, checkpoint_format, wrapper_scope
):
    _check_periodic_dcp_round_trip(tmp_path, save_training_state, checkpoint_format, wrapper_scope)


@pytest.mark.parametrize("wrapper_scope", ["plain", "model"], ids=["plain", "checkpoint-wrapped"])
def test_periodic_dcp_restores_reactivated_base_weights_and_optimizer(tmp_path, wrapper_scope):
    _check_periodic_dcp_round_trip(tmp_path, True, "adapter_buffers", wrapper_scope, trainable_base=True)


N_HASH_LAYERS = 3
N_LAYERS = 4
NUM_EXPERTS = 4


def _build_model_config(num_experts=NUM_EXPERTS, num_layers=N_LAYERS):
    return dsv4._make_v4_config(
        dim=32, n_layers=num_layers, vocab_size=512, n_heads=4, head_dim=8, rope_head_dim=4, q_lora_rank=16,
        o_lora_rank=8, n_groups=2, compress_ratios=(1,) * num_layers, window_size=32, norm_eps=1e-6, index_n_heads=2,
        index_head_dim=8, index_topk=2, moe_inter_dim=16, num_experts=num_experts, num_shared_experts=1, top_k=1,
        n_hash_layers=min(N_HASH_LAYERS, num_layers), route_norm=False, route_scale=1.0, load_balance_coeff=1e-3, hc_mult=2,
        sinkhorn_iters=2, hc_eps=1e-6, max_seq_len=32, compress_rope_theta=10000.0, original_seq_len=32,
        num_mtp_layers=1,
    )


@pytest.fixture(scope="module")
def adapter():
    model_config = _build_model_config()
    return state_dict_adapter.DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)


def test_to_hf_omits_local_lora_adapter_parameters(adapter):
    local_state_dict = {
        "layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 4),
        "layers.0.moe.routed_experts.inner_experts.w13_lora_b": torch.ones(2, 4, 2, 2),
    }

    assert adapter.to_hf(local_state_dict) == {}


def test_to_peft_maps_dense_keys_and_rejects_mtp(adapter):
    local_state_dict = {
        "lm_head.lora_a.weight": torch.ones(2, 32), "lm_head.lora_b.weight": torch.ones(512, 2),
        "layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 32),
        "layers.0.attention.wq_a.lora_b.weight": torch.ones(16, 2),
    }

    peft_state_dict = adapter.to_peft(local_state_dict)

    assert set(peft_state_dict) == {
        "base_model.model.lm_head.lora_A.weight", "base_model.model.lm_head.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight", "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight",
    }

    with pytest.raises(NotImplementedError, match="MTP"):
        adapter.to_peft({"mtp_layers.0.e_proj.lora_a.weight": torch.ones(2, 32)})


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_peft_mapping_shapes_and_safetensors_round_trip(adapter, tmp_path, rank, dtype):
    shapes = {
        "w13_lora_a": (NUM_EXPERTS, rank, 32), "w13_lora_b": (NUM_EXPERTS, 16, 2, rank),
        "w2_lora_a": (NUM_EXPERTS, rank, 16), "w2_lora_b": (NUM_EXPERTS, 32, rank),
    }
    tensors = {name: torch.arange(math.prod(shape), dtype=dtype).reshape(shape) for name, shape in shapes.items()}
    exported = adapter.to_peft({f"layers.0.moe.routed_experts.inner_experts.{name}": t for name, t in tensors.items()})
    prefix = "base_model.model.model.layers.0.mlp.experts"
    assert set(exported) == {
        f"{prefix}{layer}.lora_{factor}.weight" for layer in ("", ".base_layer") for factor in ("A", "B")
    }
    for layer, stem, width in ((".base_layer", "w13", 32), ("", "w2", 16)):
        a = exported[f"{prefix}{layer}.lora_A.weight"].reshape(NUM_EXPERTS, rank, width)
        b = exported[f"{prefix}{layer}.lora_B.weight"].reshape(32, rank, NUM_EXPERTS)
        for expert in range(NUM_EXPERTS):
            torch.testing.assert_close(a[expert], tensors[f"{stem}_lora_a"][expert], rtol=0, atol=0)
            expected_b = tensors[f"{stem}_lora_b"][expert]
            if stem == "w13":
                expected_b = torch.cat((expected_b[:, 0], expected_b[:, 1]))
            torch.testing.assert_close(b[:, :, expert], expected_b, rtol=0, atol=0)
    writer = dcp.HuggingFaceStorageWriter(path=str(tmp_path), save_distributed=True, enable_consolidation=True)
    dcp.save(exported, storage_writer=writer)
    files = list(tmp_path.glob("model-*.safetensors"))
    assert len(files) == 1
    loaded = load_file(str(files[0]))
    assert loaded.keys() == exported.keys()
    for name, value in exported.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("base_path", [None, "/data/base"], ids=["assets-fallback", "explicit-base"])
def test_peft_config_metadata_targets_and_base_path(base_path):
    adapter = state_dict_adapter.DeepSeekV4StateDictAdapter(_build_model_config(), hf_assets_path="/data/assets")

    config = adapter.peft_adapter_config(base_model_name_or_path=base_path)

    assert config["base_model_name_or_path"] == (base_path or "/data/assets")
    assert (config["r"], config["lora_alpha"]) == (64, 128.0)
    assert config["peft_type"] == "LORA"
    assert config["task_type"] == "CAUSAL_LM"
    assert config["target_modules"] == []
    assert {"self_attn.q_a_proj.weight", "self_attn.o_a_proj.weight", "mlp.shared_experts.gate_proj.weight",
            "lm_head.weight"} <= set(config["target_parameters"])
    indexer_modules = (
        "model.layers.0.attn.indexer.wq_b", "model.layers.0.attn.indexer.weights_proj",
        "model.layers.0.attn.indexer.compressor.wkv", "model.layers.0.attn.indexer.compressor.wgate",
    )
    for module_name in indexer_modules:
        matched = [
            target for target in config["target_parameters"] if module_name == target or module_name.endswith("." + target)
        ]
        assert not matched, f"{module_name} would receive an adapter via {matched}"


def _adapter_with_converter(converter_config):
    model_config = converter_config.build().convert(_build_model_config())
    return state_dict_adapter.DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)


@pytest.mark.parametrize("adapt_experts", [False, True], ids=["dense", "dense-and-experts"])
def test_peft_config_targets_match_converter(adapt_experts):
    converter = lora.DeepSeekV4LoRAConverter.Config(
        rank=4, alpha=8.0, rank_experts=4, target_modules=["attention.wo_b"], adapt_routed_experts=adapt_experts,
        include_mtp=False,
    )

    config = _adapter_with_converter(converter).peft_adapter_config()
    assert (config["r"], config["lora_alpha"]) == (4, 8.0)

    assert config["target_parameters"] == [f"model.layers.{i}.self_attn.o_b_proj.weight" for i in range(N_LAYERS)] + (
        [f"model.layers.{i}.mlp.experts.{name}" for i in range(N_LAYERS) for name in ("gate_up_proj", "down_proj")]
        if adapt_experts else []
    )


def test_peft_adapter_config_rejects_mismatched_expert_rank():
    converter_config = lora.DeepSeekV4LoRAConverter.Config(
        rank=64, rank_experts=32, target_modules=["attention.wo_b"], adapt_routed_experts=True, include_mtp=False
    )
    adapter = _adapter_with_converter(converter_config)

    with pytest.raises(ValueError, match="rank_experts"):
        adapter.peft_adapter_config()


def _distributed_worker(rank, root, action, argument):
    dist.init_process_group(
        "gloo", init_method=(Path(root) / "store").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        message = None
        try:
            action(rank, root, argument)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
        Path(root, f"rank{rank}.json").write_text(json.dumps(message))
    finally:
        dist.destroy_process_group()


def _run_distributed(tmp_path, action, argument):
    mp.spawn(_distributed_worker, args=(str(tmp_path), action, argument), nprocs=2, join=True)
    return [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]


def _check_export_guard(rank, root, invalid_rank):
    model = torch.nn.Module()
    model.register_parameter("base", torch.nn.Parameter(torch.ones(2), requires_grad=rank == invalid_rank))
    model.register_parameter("lora_a", torch.nn.Parameter(torch.ones(2)))
    manager = object.__new__(peft.DeepSeekV4PEFTCheckpointManager)
    manager.states = {"model": SimpleNamespace(model=[model])}
    manager._ensure_peft_exportable()


@pytest.mark.parametrize("invalid_rank", [None, 1], ids=["adapter-only", "remote-trainable-base"])
def test_peft_export_guard_agrees_across_ranks(tmp_path, invalid_rank):
    messages = _run_distributed(tmp_path, _check_export_guard, invalid_rank)
    if invalid_rank is None:
        assert messages == [None, None]
    else:
        assert all(message and message.startswith("ValueError:") for message in messages)
        assert all(message and "trainable non-LoRA parameters: base" in message for message in messages)
        assert messages[0] == messages[1]


class _MappingStubAdapter(state_dict_adapter.DeepSeekV4StateDictAdapter):
    def __init__(self, fail_finalization):
        self.fail_finalization = fail_finalization

    def to_peft(self, state_dict):
        return state_dict

    def peft_adapter_config(self, **kwargs):
        return {"peft_type": "LORA", "fail_finalization": self.fail_finalization}


class _FinalSaveCheckpoint(peft.DeepSeekV4PEFTCheckpointManager):
    @staticmethod
    def _finalize_peft_directory(checkpoint_id, adapter_config):
        if adapter_config["fail_finalization"]:
            raise OSError("injected finalization failure")
        peft.DeepSeekV4PEFTCheckpointManager._finalize_peft_directory(checkpoint_id, adapter_config)

    def _create_checkpoint_id(self, curr_step):
        return self.checkpoint_dir


def _save_peft(rank, root, fail_finalization):
    model = torch.nn.Module()
    model.register_parameter("lora_a", torch.nn.Parameter(torch.arange(6).reshape(2, 3).float()))
    manager = object.__new__(_FinalSaveCheckpoint)
    manager.states = {"model": ModelWrapper(model)}
    manager.last_save_in_peft = True
    manager.sd_adapter = _MappingStubAdapter(fail_finalization)
    manager.initial_load_path = ""
    manager.initial_load_in_hf = False
    manager.export_dtype = torch.float32
    manager.checkpoint_dir = str(Path(root) / "checkpoint")
    manager._save_last_step(1)


@pytest.mark.parametrize("fail_finalization", [False, True], ids=["adapter-file", "rank-zero-io-failure"])
def test_final_peft_save_finishes_consistently_across_ranks(tmp_path, fail_finalization):
    messages = _run_distributed(tmp_path, _save_peft, fail_finalization)
    if fail_finalization:
        assert all(
            message and "PEFT checkpoint finalization failed: OSError: injected" in message for message in messages
        )
        assert messages[0] == messages[1]
    else:
        assert messages == [None, None]
        saved = load_file(tmp_path / "checkpoint" / "adapter_model.safetensors")
        torch.testing.assert_close(saved["lora_a"], torch.arange(6).reshape(2, 3).float(), rtol=0, atol=0)


def test_peft_export_loads_in_transformers_and_matches_merged_logits(tmp_path):
    import copy

    peft_lib = pytest.importorskip("peft", minversion="0.20.0")
    transformers = pytest.importorskip("transformers", minversion="5.17.0")
    from safetensors.torch import save_file

    config = transformers.DeepseekV4Config(
        vocab_size=64, hidden_size=32, moe_intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=4, head_dim=8, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        n_routed_experts=2, num_experts_per_tok=1, hc_mult=2, hc_sinkhorn_iters=2,
        layer_types=["sliding_attention"], mlp_layer_types=["moe"],
        partial_rotary_factor=0.5, max_position_embeddings=64,
    )
    base = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    expected = copy.deepcopy(base)
    model_config = lora.DeepSeekV4LoRAConverter.Config(
        rank=2, rank_experts=2, alpha=4.0, include_mtp=False,
        target_modules=["attention.wq_a", "attention.wo_a"],
    ).build().convert(_build_model_config(num_experts=2, num_layers=1))
    adapter = state_dict_adapter.DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)
    shapes = {
        "attention.wq_a.lora_a.weight": (2, 32), "attention.wq_a.lora_b.weight": (16, 2),
        "attention.wo_a.lora_a.weight": (2, 16), "attention.wo_a.lora_b.weight": (16, 2),
        "moe.routed_experts.inner_experts.w13_lora_a": (2, 2, 32),
        "moe.routed_experts.inner_experts.w13_lora_b": (2, 16, 2, 2),
        "moe.routed_experts.inner_experts.w2_lora_a": (2, 2, 16),
        "moe.routed_experts.inner_experts.w2_lora_b": (2, 32, 2),
    }
    tensors = {key: torch.randn(shape) * 0.1 for key, shape in shapes.items()}
    exported = adapter.to_peft({f"layers.0.{key}": value for key, value in tensors.items()})
    save_file({key: value.clone().contiguous() for key, value in exported.items()},
              str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter.peft_adapter_config()))

    block = expected.model.layers[0]
    with torch.no_grad():
        for local, hf in (("wq_a", "q_a_proj"), ("wo_a", "o_a_proj")):
            a = tensors[f"attention.{local}.lora_a.weight"]
            b = tensors[f"attention.{local}.lora_b.weight"]
            getattr(block.self_attn, hf).weight.add_(b @ a, alpha=2.0)
        routed = "moe.routed_experts.inner_experts."
        for expert in range(2):
            a = tensors[routed + "w13_lora_a"][expert]
            b = tensors[routed + "w13_lora_b"][expert]
            block.mlp.experts.gate_up_proj[expert, :16].add_(b[:, 0, :] @ a, alpha=2.0)
            block.mlp.experts.gate_up_proj[expert, 16:].add_(b[:, 1, :] @ a, alpha=2.0)
            block.mlp.experts.down_proj[expert].add_(
                tensors[routed + "w2_lora_b"][expert] @ tensors[routed + "w2_lora_a"][expert], alpha=2.0,
            )
        inputs = torch.tensor([[1, 2, 3, 4]])
        base_logits = base(inputs, use_cache=False).logits
        expected_logits = expected(inputs, use_cache=False).logits
    loaded = peft_lib.PeftModel.from_pretrained(base, tmp_path).eval()
    loaded_state = peft_lib.get_peft_model_state_dict(loaded)
    assert loaded_state.keys() == exported.keys()
    for key, value in exported.items():
        torch.testing.assert_close(loaded_state[key], value, rtol=0, atol=0)
    with torch.no_grad():
        actual_logits = loaded(inputs, use_cache=False).logits
    assert not torch.equal(base_logits, expected_logits)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("target", ["wq_a", "attention.wq_a", "layers.0.attention.wq_a"])
def test_review_peft_target_alias_matches_exported_parameter(target):
    adapter = _adapter_with_converter(lora.DeepSeekV4LoRAConverter.Config(
        target_modules=[target], adapt_routed_experts=False, include_mtp=False,
    ))
    metadata = adapter.peft_adapter_config()
    layers = [0] if target.startswith("layers.") else range(N_LAYERS)
    assert metadata["target_parameters"] == [f"model.layers.{i}.self_attn.q_a_proj.weight" for i in layers]
    exported = adapter.to_peft({"layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 32)})
    assert set(exported) == {"base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"}


def test_review_peft_rejects_partially_unmapped_adapters(adapter):
    tensors = {
        "layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 32),
        "layers.0.attention.indexer.wq_b.lora_a.weight": torch.ones(2, 32),
    }
    with pytest.raises(ValueError, match="Unsupported.*attention.indexer.wq_b"):
        adapter.to_peft(tensors)


@pytest.mark.parametrize("peft_export", [False, True])
def test_review_hf_export_rejected_before_checkpoint_initialization(monkeypatch, peft_export):
    def unexpected_init(self, config, **kwargs):
        pytest.fail("Invalid export configuration reached checkpoint initialization")
    monkeypatch.setattr(peft.NPUCheckpointManager, "__init__", unexpected_init)
    config = peft.DeepSeekV4PEFTCheckpointManager.Config(
        enable=True, last_save_in_peft=peft_export, last_save_in_hf=True,
    )
    with pytest.raises(ValueError, match="HF|last_save_in_hf"):
        config.build()


@pytest.mark.parametrize("invalid", ["mtp", "rank"])
def test_review_peft_configuration_rejected_before_checkpoint_initialization(monkeypatch, invalid):
    converter = lora.DeepSeekV4LoRAConverter.Config(
        target_modules=["attention.wq_a"], include_mtp=invalid == "mtp",
        rank=2, rank_experts=3 if invalid == "rank" else 2,
    )
    adapter = _adapter_with_converter(converter)
    def unexpected_init(self, config, **kwargs):
        pytest.fail("Invalid export configuration reached checkpoint initialization")
    monkeypatch.setattr(peft.NPUCheckpointManager, "__init__", unexpected_init)
    config = peft.DeepSeekV4PEFTCheckpointManager.Config(enable=True)
    with pytest.raises((ValueError, NotImplementedError), match="MTP|rank_experts|Unsupported"):
        config.build(sd_adapter=adapter)


def test_review_native_checkpoint_allows_mtp_and_mixed_ranks(monkeypatch):
    adapter = _adapter_with_converter(lora.DeepSeekV4LoRAConverter.Config(
        rank=2, rank_experts=3, target_modules=["attention.wq_a"],
    ))
    def initialize(self, config, **kwargs):
        self.initial_load_path = "/base"
    monkeypatch.setattr(peft.NPUCheckpointManager, "__init__", initialize)
    config = peft.DeepSeekV4PEFTCheckpointManager.Config(enable=True, last_save_in_peft=False)
    config.build(sd_adapter=adapter)


def test_review_converter_freezes_base_without_example_wrapper(monkeypatch):
    def parallelize(model, **kwargs):
        # Sharding may replace parameters and reset requires_grad.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model
    monkeypatch.setattr(dsv4, "parallelize_deepseek_v4", parallelize)
    spec = dsv4.model_registry("debugmodel", converters=[lora.DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)])
    with torch.device("meta"):
        model = spec.model.build()
    model = spec.parallelize_fn(model)
    assert spec.post_optimizer_build_fn is None
    parameters = dict(model.named_parameters())
    assert any("lora_" in name for name in parameters)
    assert all(parameter.requires_grad == ("lora_" in name) for name, parameter in parameters.items())


@pytest.mark.parametrize("factory,flavor", [
    ("deepseek_v4_lora", "deepseek_v4_flash"),
    ("deepseek_v4_lora_debugmodel", "debugmodel"),
])
def test_lora_training_entrypoints(factory, flavor):
    from torchtitan_npu.models.deepseek_v4 import lora_config

    config = getattr(lora_config, factory)()
    assert config.model_spec.flavor == flavor
    assert config.model_spec.model.lora is not None
    assert config.model_spec.parallelize_fn is dsv4._parallelize_lora
