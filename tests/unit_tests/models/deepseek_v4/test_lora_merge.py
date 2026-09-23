# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.lora.merge_adapter import merge
from tests.unit_tests.models.deepseek_v4.lora_test_utils import _build_model_config
from torchtitan_npu.models.deepseek_v4.lora import (
    OFFICIAL_MODULES,
    DeepSeekV4LoRAConverter,
    build_lora_merge_plan,
    merge_lora_weight,
)
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter


@pytest.fixture
def exported_adapter(tmp_path):
    model_config = (
        DeepSeekV4LoRAConverter.Config(
            rank=2,
            rank_experts=2,
            alpha=4.0,
            include_mtp=False,
            target_modules=["attention.wq_a", "attention.wq_b"],
        )
        .build()
        .convert(_build_model_config(num_experts=2, num_layers=1))
    )
    adapter = DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)
    with torch.device("meta"):
        model = model_config.build()
    shapes = {
        name.removeprefix("layers.0."): parameter.shape
        for name, parameter in model.named_parameters()
        if name.startswith("layers.0.") and "lora_" in name
    }
    generator = torch.Generator().manual_seed(31)
    factors = {name: torch.randn(shape, generator=generator) * 0.1 for name, shape in shapes.items()}
    exported = adapter.to_peft({f"layers.0.{name}": value for name, value in factors.items()})
    directory = tmp_path / "adapter"
    _write_adapter(directory, exported, config=adapter.peft_adapter_config())

    return model, factors, adapter, directory


@pytest.fixture
def transformers_adapter(exported_adapter):
    _, factors, adapter, directory = exported_adapter
    pytest.importorskip("transformers.models.deepseek_v4")
    import transformers

    with torch.random.fork_rng(devices=[], device_type="cpu"):
        torch.manual_seed(13)
        config = transformers.DeepseekV4Config(
            vocab_size=64,
            hidden_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=8,
            q_lora_rank=16,
            o_lora_rank=8,
            o_groups=2,
            n_routed_experts=2,
            num_experts_per_tok=1,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            layer_types=["sliding_attention"],
            mlp_layer_types=["moe"],
            partial_rotary_factor=0.5,
            max_position_embeddings=64,
        )
        base = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    expected = copy.deepcopy(base)
    layer = expected.model.layers[0]
    with torch.no_grad():
        for local, hf in (("wq_a", "q_a_proj"), ("wq_b", "q_b_proj")):
            a, b = (factors[f"attention.{local}.lora_{factor}.weight"] for factor in ("a", "b"))
            getattr(layer.self_attn, hf).weight.add_(b @ a, alpha=2.0)
        prefix = "moe.routed_experts.inner_experts."
        for expert in range(2):
            a = factors[prefix + "w13_lora_a"][expert]
            b = factors[prefix + "w13_lora_b"][expert]
            layer.mlp.experts.gate_up_proj[expert, :16].add_(b[:, 0] @ a, alpha=2.0)
            layer.mlp.experts.gate_up_proj[expert, 16:].add_(b[:, 1] @ a, alpha=2.0)
            layer.mlp.experts.down_proj[expert].add_(
                factors[prefix + "w2_lora_b"][expert] @ factors[prefix + "w2_lora_a"][expert], alpha=2.0
            )
    return base, expected, adapter, directory


def _checkpoint_weights(adapter, weights, layout):
    result = adapter.to_hf(weights)
    if layout == "official":
        return result
    renamed = {}
    for name, value in result.items():
        if ".ffn.experts." in name:
            name = name.replace(".ffn.experts.", ".mlp.experts.")
        else:
            for hf, official in OFFICIAL_MODULES.items():
                name = name.replace(f".{official}.weight", f".{hf}.weight")
        renamed["model." + name] = value
    return renamed


@pytest.mark.parametrize("layout", ["official", "transformers"])
def test_current_export_merges_checkpoint_weights_and_reloads_state(exported_adapter, tmp_path, layout):
    model, factors, adapter, adapter_dir = exported_adapter
    generator = torch.Generator().manual_seed(13)
    base = {
        name: torch.randn(parameter.shape, generator=generator)
        for name, parameter in model.named_parameters()
        if name.startswith("layers.0.")
        and "lora_" not in name
        and (name.endswith(("wq_a.weight", "wq_b.weight", "w1_EFD", "w2_EDF", "w3_EFD")))
    }
    expected = {name: value.clone() for name, value in base.items()}
    for local in ("wq_a", "wq_b"):
        a, b = (factors[f"attention.{local}.lora_{factor}.weight"] for factor in ("a", "b"))
        expected[f"layers.0.attention.{local}.weight"].add_(b @ a, alpha=2.0)
    prefix = "moe.routed_experts.inner_experts."
    for weight, factor, component in (("w1_EFD", "w13", 0), ("w3_EFD", "w13", 1), ("w2_EDF", "w2", None)):
        a, b = (factors[prefix + factor + "_lora_" + suffix] for suffix in ("a", "b"))
        if component is not None:
            b = b[:, :, component]
        expected["layers.0." + prefix + weight].add_(b @ a, alpha=2.0)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    official = _checkpoint_weights(adapter, base, layout)
    save_file({name: value.clone().contiguous() for name, value in official.items()}, base_dir / "model.safetensors")
    output = tmp_path / "merged"

    report = merge(str(base_dir), str(adapter_dir), str(output))
    written = load_file(output / "model.safetensors")
    expected_official = _checkpoint_weights(adapter, expected, layout)

    assert report["merged_count"] == 8
    assert written.keys() == expected_official.keys()
    for name, value in expected_official.items():
        torch.testing.assert_close(written[name], value, rtol=1e-6, atol=1e-7)
    if layout == "official":
        restored = adapter.from_hf(written)
        for name, value in expected.items():
            torch.testing.assert_close(restored[name].reshape_as(value), value, rtol=1e-6, atol=1e-7)
    for name, value in load_file(base_dir / "model.safetensors").items():
        torch.testing.assert_close(value, official[name], rtol=0, atol=0)


@pytest.mark.parametrize("storage", ["save_pretrained", "fused"])
def test_current_export_merges_transformers_experts_and_matches_logits(transformers_adapter, tmp_path, storage):
    from transformers import AutoModelForCausalLM

    base, expected, _, adapter_dir = transformers_adapter
    base_dir = tmp_path / "base"
    if storage == "save_pretrained":
        base.save_pretrained(base_dir)
    else:
        base_dir.mkdir()
        base.config.save_pretrained(base_dir)
        save_file(base.state_dict(), base_dir / "model.safetensors")
    output = tmp_path / "merged"

    report = merge(str(base_dir), str(adapter_dir), str(output))
    loaded = AutoModelForCausalLM.from_pretrained(output, attn_implementation="eager").eval()

    assert report["merged_count"] == (8 if storage == "save_pretrained" else 4)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=1e-6, atol=1e-7)
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3, 4]])
        actual_logits = loaded(tokens, use_cache=False).logits
        expected_logits = expected(tokens, use_cache=False).logits
        assert not torch.equal(base(tokens, use_cache=False).logits, expected_logits)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)


def _write_adapter(directory, tensors, *, rank=2, alpha=4, config=None):
    directory.mkdir()
    save_file({name: value.contiguous() for name, value in tensors.items()}, directory / "adapter_model.safetensors")
    (directory / "adapter_config.json").write_text(
        json.dumps(
            config
            or {
                "peft_type": "LORA",
                "r": rank,
                "lora_alpha": alpha,
            }
        )
    )


@pytest.mark.parametrize("kind", ["empty", "orphan_a", "orphan_b", "unknown"])
def test_merge_plan_rejects_incomplete_or_unmatched_adapter(kind):
    tensors = {
        "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
        "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
    }
    if kind == "empty":
        tensors.clear()
    elif kind != "unknown":
        tensors.pop(f"base_model.model.proj.lora_{'B' if kind == 'orphan_a' else 'A'}.weight")
    with pytest.raises(ValueError, match=r"Adapter|adapter"):
        build_lora_merge_plan(tensors, {} if kind == "unknown" else {"proj.weight": "model.safetensors"})


def test_merge_tensor_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        merge_lora_weight(torch.zeros(8, 6), torch.ones(2, 6), torch.ones(4, 2), scaling=1.0)


@pytest.mark.parametrize("location", ["base", "adapter", "existing"])
def test_merge_protects_inputs_and_existing_output(tmp_path, location):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    output = tmp_path / location / "merged"
    if location == "existing":
        output.mkdir(parents=True)
    with pytest.raises(FileExistsError if location == "existing" else ValueError):
        merge(str(base), str(adapter), str(output))


def test_merge_rejects_indexed_weight_missing_from_shard(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    save_file({"other.weight": torch.zeros(4, 4)}, base / "shard.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"proj.weight": "shard.safetensors"}}))
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )
    with pytest.raises(ValueError, match="missing tensors"):
        merge(str(base), str(adapter), str(tmp_path / "output"))


@pytest.mark.parametrize("declared", [False, True])
def test_quantized_input_requires_explicit_conversion(tmp_path, declared):
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    save_file({"proj.weight": torch.ones(4, 4, dtype=torch.float8_e4m3fn)}, base / "model.safetensors")
    if declared:
        (base / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )
    with pytest.raises(ValueError, match=r"checkpoint|Decode"):
        merge(str(base), str(adapter), str(output))
    assert not output.exists()


def test_merge_preserves_untargeted_shards_and_metadata(tmp_path):
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    save_file({"proj.weight": torch.ones(4, 4)}, base / "weights.safetensors")
    save_file({"other.scale": torch.ones(1)}, base / "untouched.safetensors")
    index = {"weight_map": {"proj.weight": "weights.safetensors", "other.scale": "untouched.safetensors"}}
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    (base / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )
    merge(str(base), str(adapter), str(output))
    torch.testing.assert_close(load_file(output / "weights.safetensors")["proj.weight"], torch.full((4, 4), 5.0))
    for name in ("config.json", "model.safetensors.index.json", "untouched.safetensors"):
        assert (output / name).read_bytes() == (base / name).read_bytes()
