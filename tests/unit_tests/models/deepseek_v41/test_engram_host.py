# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Default Host training without the CANN Engram operator package."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchtitan.config.override import OverrideConfig, apply_overrides
from torchtitan.distributed import fsdp

from tests.unit_tests.models.deepseek_v41.engram_host_worker import table_config
from torchtitan_npu.models.deepseek_v41.config_registry import deepseek_v41_debugmodel
from torchtitan_npu.models.deepseek_v41.engram_host import HostEngramTable
from torchtitan_npu.models.deepseek_v41.parallelize import _apply_fsdp_with_ignored_params

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("recompute", [False, True], ids=["eager", "full_ac"])
def test_local_sparse_training_and_checkpoint_match_embedding(recompute):
    table = table_config().build()
    with torch.no_grad():
        table.weight.copy_(torch.arange(48).view(12, 4) / 16)
    reference = torch.nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False, sparse=True)
    optimizer = torch.optim.SparseAdam([table.weight], lr=0.05)
    ref_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05)
    for step, rows in enumerate(([0, 7, 7, 11], [1, 7, 1], [0, 7, 2])):
        ids = torch.tensor(rows)
        output = (
            checkpoint(table._distributed_lookup, ids, use_reentrant=False)
            if recompute
            else table._distributed_lookup(ids)
        )
        expected = reference(ids)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        grad = torch.arange(output.numel()).reshape_as(output).float() + 1
        output.backward(grad)
        expected.backward(grad)
        pending = table.pending_sparse_grad()
        assert table.weight.grad is None and pending.is_sparse
        torch.testing.assert_close(pending.to_dense(), reference.weight.grad.to_dense(), rtol=0, atol=0)
        table.prepare_sparse_optimizer_step()
        optimizer.step()
        ref_optimizer.step()
        torch.testing.assert_close(table.weight, reference.weight, rtol=0, atol=0)
        table.clear_sparse_gradient()
        ref_optimizer.zero_grad(set_to_none=True)
        if step == 1:
            weights, states = copy.deepcopy(table.state_dict()), copy.deepcopy(optimizer.state_dict())
            table = table_config().build()
            table.load_state_dict(weights)
            optimizer = torch.optim.SparseAdam([table.weight], lr=0.05)
            optimizer.load_state_dict(states)


def test_default_recipe_builds_host_table_and_override_preserves_geometry(monkeypatch):
    monkeypatch.setenv("USE_GOLDEN", "1")
    trainer = deepseek_v41_debugmodel()
    layer = next(layer for layer in trainer.model_spec.model.layers if layer.engram is not None)
    cfg = layer.engram.table
    table = cfg.build()
    assert isinstance(table, HostEngramTable) and table.weight.device.type == "cpu"
    assert table.weight._engram_host_offload
    apply_overrides(
        OverrideConfig(
            imports=[
                (
                    "torchtitan_npu.override.deepseek_v41.engram.host_offload",
                    {"num_max_tokens_per_rank": 512, "pin_memory": False},
                )
            ]
        ),
        trainer,
    )
    fused = layer.engram.table.build()
    assert type(fused) is not type(table)
    assert fused.weight.shape == table.weight.shape and fused.weight.dtype == table.weight.dtype
    assert fused._checkpoint_suffix() == table._checkpoint_suffix()


def test_two_rank_host_training_and_checkpoint_match_sparse_embedding(tmp_path: Path):
    log = tmp_path / "worker.log"
    with log.open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "-m",
                "tests.unit_tests.models.deepseek_v41.engram_host_worker",
            ],
            env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "1"},
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    assert result.returncode == 0, log.read_text()


@pytest.mark.parametrize("ignore_table", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_decoder_fsdp_keeps_shared_helper_unchanged(monkeypatch, ignore_table, fail):
    model = nn.Module()
    model.enable_weight_tying = False
    model.tok_embeddings = model.norm = model.lm_head = None
    model.layers = nn.ModuleDict({"0": nn.Linear(2, 2)})
    table = nn.Parameter(torch.ones(3, 2))
    model.layers["0"].register_parameter("host_table", table)
    ignored = {table} if ignore_table else set()
    calls = []
    original_helper = fsdp.apply_fsdp_to_decoder

    def shard_spy(module, **kwargs):
        assert fsdp.fully_shard is shard_spy
        calls.append((module, kwargs.get("ignored_params", set())))
        if fail:
            raise RuntimeError("shard failure")

    monkeypatch.setattr(fsdp, "fully_shard", shard_spy)
    monkeypatch.setattr(fsdp, "disable_fsdp_gradient_division", lambda model: None)
    kwargs = dict(
        dp_mesh=SimpleNamespace(mesh_dim_names=("fsdp",)),
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        ignored_params=ignored,
    )
    if fail:
        with pytest.raises(RuntimeError, match="shard failure"):
            _apply_fsdp_with_ignored_params(model, **kwargs)
    else:
        _apply_fsdp_with_ignored_params(model, **kwargs)
        assert [module for module, _ in calls] == [model.layers["0"], model]
    assert calls and all(params == ignored for _, params in calls)
    assert fsdp.fully_shard is shard_spy
    assert fsdp.apply_fsdp_to_decoder is original_helper
