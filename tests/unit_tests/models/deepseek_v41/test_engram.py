# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.config.manager import ConfigManager

from torchtitan_npu.extensions.components.optimizer import (
    HostSparseOptimizersContainer,
)
from torchtitan_npu.models.deepseek_v41 import (
    EngramArgs,
    model_registry,
)
from torchtitan_npu.models.deepseek_v41.config_registry import (
    deepseek_v41_debugmodel,
)
from torchtitan_npu.models.deepseek_v41.engram_config import _make_engram_configs
from torchtitan_npu.models.deepseek_v41.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _preserve_rng(monkeypatch):
    monkeypatch.setenv("USE_GOLDEN", "1")
    with torch.random.fork_rng(devices=[]):
        yield


def _build_engram():
    config = _make_engram_configs(
        hidden_size=8,
        hc_mult=4,
        vocab_size=32,
        engram=EngramArgs(
            layer_ids=(1,),
            vocab_size_per_ngram=(11, 17),
            n_embed_per_ngram=6,
            num_heads_per_ngram=2,
            pad_id=2,
            require_token_id_map=False,
        ),
    )[1]
    engram = config.build()
    engram.init_states(buffer_device=torch.device("cpu"))
    return engram


def _numpy_hash_reference(table, input_ids):
    tokens = input_ids.numpy().astype(np.int64, copy=False)
    B, L = tokens.shape
    shifts = []
    for distance in range(max(table.ngram_orders)):
        shifted = np.full((B, L), table.pad_id, dtype=np.int64)
        if distance == 0:
            shifted = tokens
        else:
            shifted[:, distance:] = tokens[:, :-distance]
        shifts.append(shifted)

    multipliers = table.hash_multipliers.numpy()
    sizes = table.head_vocab_sizes.numpy()
    offsets = table.offsets.numpy()
    hashes = []
    table_idx = 0
    for order in table.ngram_orders:
        mixed = shifts[0] * multipliers[0]
        for distance in range(1, order):
            mixed = np.bitwise_xor(mixed, shifts[distance] * multipliers[distance])
        end = table_idx + table.num_heads
        hashes.append(mixed[..., None] % sizes[table_idx:end] + offsets[table_idx:end])
        table_idx = end
    return torch.from_numpy(np.concatenate(hashes, axis=-1))


def _official_demo_forward_reference(engram, hidden, input_ids, positions):
    """Independent, branch-by-branch port of the public Engram demo."""
    table = engram.table
    hashes = _numpy_hash_reference(table, input_ids)
    memory = F.embedding(hashes, table.weight).flatten(2)
    D, M = engram.gate.hidden_size, engram.gate.num_branches
    # the fused projection holds each branch's key first, then the shared value
    key_rows, value_rows = engram.gate.wkv.split([M * D, D], dim=0)
    values = F.linear(memory, value_rows)

    gated = []
    for branch in range(M):
        key = F.linear(memory, key_rows[branch * D : (branch + 1) * D])
        key = F.rms_norm(
            key,
            (engram.gate.hidden_size,),
            engram.gate.k_weight[branch],
            engram.gate.norm_eps,
        )
        query = F.rms_norm(
            hidden[:, :, branch],
            (engram.gate.hidden_size,),
            engram.gate.q_weight[branch],
            engram.gate.norm_eps,
        )
        logits = (key * query).sum(-1) / math.sqrt(engram.gate.hidden_size)
        logits = logits.abs().clamp_min(1e-6).sqrt() * logits.sign()
        gated.append(logits.sigmoid().unsqueeze(-1) * values)
    return hidden + torch.stack(gated, dim=2)


def test_hash_matches_official_numpy_dataflow():
    table = _build_engram().table
    assert table.num_embeddings % 256 == 0
    assert table.num_embeddings >= table.logical_num_embeddings
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int64)
    actual = table.hash(input_ids)
    expected = _numpy_hash_reference(table, input_ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Everything else the table needs follows from the flavor, so a checkpoint
    # carries the learned rows and nothing more.
    assert set(table.state_dict()) == {"weight"}


def test_hash_resets_ngram_context_at_packed_document_boundary():
    table = _build_engram().table
    packed_ids = torch.tensor([[4, 5, 9, 10]], dtype=torch.int64)
    packed_positions = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
    packed_hashes = table.hash(packed_ids, packed_positions)

    second_document = torch.tensor([[9, 10]], dtype=torch.int64)
    document_positions = torch.tensor([[0, 1]], dtype=torch.int64)
    document_hashes = table.hash(second_document, document_positions)
    torch.testing.assert_close(packed_hashes[:, 2:], document_hashes, rtol=0, atol=0)


def test_forward_backward_matches_official_demo_dataflow():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        engram = _build_engram()
        reference = copy.deepcopy(engram)
        hidden = torch.randn(2, 9, 4, 8, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_()
        input_ids = torch.randint(0, 32, (2, 9))
        positions = torch.arange(9).expand(2, -1)
        actual = engram(hidden, input_ids, positions)
        expected = _official_demo_forward_reference(reference, reference_hidden, input_ids, positions)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        upstream_grad = torch.randn_like(actual)
        actual.backward(upstream_grad)
        expected.backward(upstream_grad)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            engram.table.pending_sparse_grad().to_dense(), reference.table.weight.grad, rtol=2e-4, atol=2e-5
        )
        for name, parameter in engram.gate.named_parameters():
            torch.testing.assert_close(
                parameter.grad, dict(reference.gate.named_parameters())[name].grad, rtol=2e-4, atol=2e-5
            )


def test_forward_casts_fp32_table_memory_to_mixed_precision_activations():
    engram = _build_engram().to(dtype=torch.bfloat16)
    engram.table.weight = torch.nn.Parameter(engram.table.weight.detach().float())
    hidden = torch.randn(1, 4, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.randint(0, 32, (1, 4))
    positions = torch.arange(4).unsqueeze(0)

    output = engram(hidden, input_ids, positions)

    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert engram.table.pending_sparse_grad() is not None
    assert engram.table.pending_sparse_grad().dtype == torch.float32


def test_bfloat16_gate_keeps_normalization_and_residual_in_fp32():
    # Independent per-branch RMSNorm oracle for cann-recipes-infer 76b9cb8,
    # models/deepseek_v4_1/models/modules/engram.py (unquantized projection).
    torch.manual_seed(31)
    engram = _build_engram().to(dtype=torch.bfloat16)
    reference = copy.deepcopy(engram)
    hidden = torch.randn(2, 17, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    input_ids = torch.randint(0, 32, (2, 17))
    with torch.no_grad():
        engram.gate.q_weight.uniform_(0.25, 1.75)
        engram.gate.k_weight.uniform_(0.25, 1.75)
        reference.gate.load_state_dict(engram.gate.state_dict())

    actual = engram(hidden, input_ids)
    memory = F.embedding(_numpy_hash_reference(reference.table, input_ids), reference.table.weight).flatten(2)
    kv = F.linear(memory.to(hidden.dtype), reference.gate.wkv).float()
    keys, value = kv.split([32, 8], dim=-1)
    gated = []
    for branch, key in enumerate(keys.split(8, dim=-1)):
        query = F.rms_norm(
            reference_hidden[:, :, branch].float(), (8,), reference.gate.q_weight[branch].float(), engram.gate.norm_eps
        )
        key = F.rms_norm(key, (8,), reference.gate.k_weight[branch].float(), engram.gate.norm_eps)
        dot = (query * key).sum(-1) / math.sqrt(8)
        magnitude = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.where(dot >= 0, magnitude, -magnitude).sigmoid()
        gated.append(gate.unsqueeze(-1) * value)
    expected = (reference_hidden.float() + torch.stack(gated, dim=2)).to(hidden.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream_grad = torch.randn_like(actual)
    actual.backward(upstream_grad)
    expected.backward(upstream_grad)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.02, atol=0.015625)
    for name, parameter in engram.gate.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(reference.gate.named_parameters())[name].grad, rtol=0.02, atol=0.015625
        )
    torch.testing.assert_close(
        engram.table.pending_sparse_grad().to_dense(), reference.table.weight.grad, rtol=0.02, atol=0.015625
    )


def test_zero_dot_gate_uses_positive_clamped_sqrt():
    engram = _build_engram()
    with torch.no_grad():
        engram.table.weight.fill_(1)
        engram.gate.wkv.zero_()
        engram.gate.wkv[-8:].fill_(1 / engram.gate.wkv.shape[1])
        engram.gate.q_weight.zero_()
    hidden = torch.zeros(1, 2, 4, 8, requires_grad=True)

    output = engram(hidden, torch.tensor([[4, 5]]))

    # Zero dot maps to sigmoid(sqrt(1e-6)), rather than sigmoid(0).
    expected_gate = 1 / (1 + math.exp(-0.001))
    torch.testing.assert_close(output, torch.full_like(output, expected_gate), rtol=0, atol=1e-7)
    output.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    torch.testing.assert_close(engram.gate.q_weight.grad, torch.zeros_like(engram.gate.q_weight), rtol=0, atol=0)


def test_debug_flavor_enables_engram_and_selects_sparse_adam():
    spec = model_registry("deepseek_v41_debugmodel")
    enabled_layers = [idx for idx, layer in enumerate(spec.model.layers) if layer.engram is not None]
    assert enabled_layers == [1, 14]

    trainer_config = deepseek_v41_debugmodel()
    param_groups = trainer_config.optimizer.param_groups
    assert param_groups[0].pattern == r".*\.engram\.table\.weight$"
    assert param_groups[0].optimizer_name == "SparseAdam"
    assert param_groups[0].optimizer_kwargs["lr"] == 5e-5
    assert param_groups[1].optimizer_name == "AdamW"


def test_released_v41_geometry_builds_published_logical_tables(monkeypatch):
    """Official row totals exercise prime allocation across both complete layers."""
    full = model_registry("deepseek_v41_flash_40layers_16experts_vision")
    layers = [(i, layer.engram) for i, layer in enumerate(full.model.layers) if layer.engram is not None]
    # Source: deepseek-ai/DeepSeek-V4.1-Flash, revision 2bc89ac, text_config.
    assert [i for i, _ in layers] == [1, 14]
    for (_, config), logical_rows in zip(layers, (384006168, 384016682), strict=True):
        table = config.table
        assert sum(table.head_vocab_sizes) == logical_rows
        assert table.num_embeddings == (logical_rows + 2047) // 2048 * 2048
        assert (table.ngram_orders, table.num_heads, table.embedding_dim) == ((2, 3, 4), 8, 256)
        assert table.pad_id == 2
        assert table.require_token_id_map and table.compressed_vocab_size == 99092
        assert config.gate.memory_dim == 6144
        assert config.gate.norm_eps == 1e-20


def test_engram_optimizer_materializes_state_for_unused_parameters():
    class PartialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Parameter(torch.ones(2))
            self.unused = torch.nn.Parameter(torch.ones(2))

    model = PartialModel()
    config = HostSparseOptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 1e-3},
            )
        ],
    )
    optimizers = config.build(model_parts=[model])
    model.used.sum().backward()
    optimizers.step()
    optimizers.zero_grad()

    state = optimizers.state_dict()
    assert state["state.used.step"].item() == 1
    assert state["state.unused.step"].item() == 0
    assert torch.count_nonzero(state["state.unused.exp_avg"]) == 0
    assert torch.count_nonzero(state["state.unused.exp_avg_sq"]) == 0


def test_host_engram_declares_gate_and_metadata_sharding_with_and_without_ep():
    from torchtitan_npu.models.deepseek_v41.sharding import (
        set_engram_sharding_config,
    )

    dense = _make_engram_configs(
        hidden_size=8,
        hc_mult=4,
        vocab_size=32,
        engram=EngramArgs(
            layer_ids=(1,),
            vocab_size_per_ngram=(11, 17),
            n_embed_per_ngram=6,
            num_heads_per_ngram=2,
            require_token_id_map=False,
        ),
    )[1]
    set_engram_sharding_config(dense, enable_ep=False, enable_sp=False)
    assert dense.sharding_config is not None
    assert dense.table.sharding_config is not None
    assert set(dense.gate.sharding_config.state_shardings) == {"wkv", "q_weight", "k_weight"}

    distributed = _make_engram_configs(
        hidden_size=8,
        hc_mult=4,
        vocab_size=32,
        engram=EngramArgs(
            layer_ids=(1,),
            vocab_size_per_ngram=(11, 17),
            n_embed_per_ngram=6,
            num_heads_per_ngram=2,
            require_token_id_map=False,
        ),
    )[1]
    set_engram_sharding_config(distributed, enable_ep=True, enable_sp=False)
    assert distributed.sharding_config is not None
    assert set(distributed.sharding_config.in_src_shardings) == {
        "hidden_states_BLMD",
        "input_ids_BL",
        "positions_BL",
        "image_mask",
    }
    assert distributed.table.sharding_config is not None
    assert set(distributed.table.sharding_config.state_shardings) == {
        "weight",
        "token_id_map",
        "head_vocab_sizes",
        "offsets",
        "hash_multipliers",
        "compressed_pad_id",
    }


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("direction", ["to_hf", "from_hf"])
def test_host_checkpoint_hf_conversion_rejected(ep_size, direction):
    model_config = model_registry("deepseek_v41_debugmodel").model
    adapter = DeepSeekV41StateDictAdapter(model_config, None)
    table = _build_engram().table
    table._ep_size = ep_size
    table._ep_rank = 0
    state = table.state_dict(prefix="layers.1.engram.table.")
    assert any(key.startswith("layers.1.engram.table.weight") for key in state)
    with pytest.raises(NotImplementedError, match="native DCP"):
        getattr(adapter, direction)(state)


def test_image_tokens_reset_hash_history_and_receive_no_residual_gradient():
    engram = _build_engram()
    ids = torch.tensor([[1, 4, 5, 6, 7, 8]])
    positions = torch.arange(6).unsqueeze(0)
    mask = torch.tensor([[False, True, True, False, False, False]])
    hashes = engram.table.hash(ids, positions, image_mask=mask)
    # Text suffix after an image must hash exactly like a new document.
    expected = engram.table.hash(ids[:, 3:], torch.arange(3).unsqueeze(0))
    torch.testing.assert_close(hashes[:, 3:], expected, rtol=0, atol=0)
    hidden = torch.randn(1, 6, 4, 8, requires_grad=True)
    output = engram(hidden, ids, positions, image_mask=mask)
    torch.testing.assert_close(output[mask], hidden[mask], rtol=0, atol=0)
    output[mask].sum().backward()
    torch.testing.assert_close(hidden.grad[mask], torch.ones_like(hidden.grad[mask]))
    assert torch.count_nonzero(engram.gate.wkv.grad) == 0
    assert torch.count_nonzero(engram.table.pending_sparse_grad().values()) == 0


def test_explicit_disable_preserves_dense_optimizer_recipe():
    trainer = deepseek_v41_debugmodel()
    dense = copy.deepcopy(trainer.optimizer.param_groups[1])
    trainer = ConfigManager().parse_args([
        "--module", "torchtitan_npu.models.deepseek_v41",
        "--config", "deepseek_v41_debugmodel", "--no-engram-enabled",
    ])
    trainer.model_spec.model.update_from_config(config=trainer)
    assert all(layer.engram is None for layer in trainer.model_spec.model.layers)
    assert trainer.optimizer.param_groups == [dense]
    adapter = DeepSeekV41StateDictAdapter(trainer.model_spec.model, None)
    assert adapter.to_hf({}) == {}
