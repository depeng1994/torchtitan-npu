# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract tests for CANN Engram host-offload training."""

from __future__ import annotations

import pytest
import torch
from torchtitan.components.optimizer import ParamGroupConfig

from torchtitan_npu.extensions.components.optimizer import (
    HostSparseOptimizersContainer,
)
from torchtitan_npu.ops.ascendc.engram import EngramBufferHandle
from torchtitan_npu.override.deepseek_v41.engram import ascendc as ascendc_mod
from torchtitan_npu.override.deepseek_v41.engram.ascendc import (
    HostOffloadEngramTable,
    _EngramFetchSparseOffload,
)

pytestmark = pytest.mark.cpu

# EngramFetch serves rows in 128-element units, so every table the bridge
# accepts declares a row width that is a multiple of it.
_DIM = 128


def _ramp(rows: int, *, start: int = 0) -> torch.Tensor:
    """A deterministic (rows, _DIM) tensor, the wide stand-in for arange."""
    return torch.arange(start, start + rows * _DIM, dtype=torch.float32).reshape(rows, _DIM)


class _FakeElasticBuffer:
    instances = []
    size_hint_args = []

    def __init__(
        self,
        group,
        *,
        num_cpu_bytes,
        num_max_tokens_per_rank,
        with_grad,
    ):
        self.group = group
        self.num_cpu_bytes = num_cpu_bytes
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.with_grad = with_grad
        self.storage = None
        self.write_count = 0
        self.direct_map = True
        self.last_grad_shape = None
        self.extra_unique_local_entries = ()
        type(self).instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances.clear()
        cls.size_hint_args.clear()

    @classmethod
    def get_engram_storage_size_hint(cls, num_entries, hidden, dtype):
        cls.size_hint_args.append((num_entries, hidden, dtype))
        return 2 * 1024 * 1024

    def engram_write(self, storage):
        assert storage.device.type == "cpu"
        assert storage.is_contiguous()
        self.write_count += 1
        # Direct registration keeps the caller's memory, so later in-place
        # updates are visible without another write.
        self.storage = storage if self.direct_map else storage.clone()
        if self.direct_map:
            self._engram_storage_ref = storage

    def engram_fetch(self, indices):
        assert indices.dtype == torch.int32
        assert indices.ndim == 1
        fetched = self.storage.index_select(0, indices.long())
        fetch_ctx = indices.clone()
        return lambda: (fetched, fetch_ctx)

    def engram_fetch_grad(self, grad_fetched, fetch_ctx):
        self.last_grad_shape = grad_fetched.shape
        unique, inverse = torch.unique(fetch_ctx.long(), sorted=True, return_inverse=True)
        grad_unique = torch.zeros(
            unique.numel(),
            grad_fetched.shape[1],
            dtype=torch.float32,
        )
        grad_unique.index_add_(0, inverse, grad_fetched.float())
        if self.extra_unique_local_entries:
            extra_ids = torch.tensor(self.extra_unique_local_entries, dtype=torch.int32)
            extra_grads = torch.full(
                (extra_ids.numel(), grad_fetched.shape[1]),
                12345.0,
                dtype=grad_fetched.dtype,
            )
            grad_unique = torch.cat((grad_unique.to(grad_fetched.dtype), extra_grads))
            unique = torch.cat((unique.to(torch.int32), extra_ids))
        return grad_unique.to(grad_fetched.dtype), unique.to(torch.int32)


class _FakeEPMesh:
    def size(self):
        return 2

    def get_group(self):
        return "fake-ep-group"


def _host_table_config(*, capacity=4):
    return HostOffloadEngramTable.Config(
        vocab_size=16,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(7,),
        embedding_dim=_DIM,
        num_embeddings=10,
        require_token_id_map=False,
        num_max_tokens_per_rank=capacity,
        pin_memory=False,
        param_init={"weight": torch.nn.init.ones_},
    )


def _wire_buffer(table, monkeypatch, *, param_dtype=torch.float32):
    """Run the setup ``parallelize_deepseek_v41`` performs, without a mesh."""
    table.ep_mesh = _FakeEPMesh()
    monkeypatch.setattr(table, "_elastic_buffer_type", lambda: _FakeElasticBuffer)
    monkeypatch.setattr(ascendc_mod, "_dedicated_engram_group", lambda group, layer_id: "fake-engram-group")
    table.init_elastic_buffer(param_dtype=param_dtype)


def test_host_table_builds_its_buffer_before_the_first_lookup(monkeypatch):
    _FakeElasticBuffer.reset()
    table = HostOffloadEngramTable(_host_table_config(capacity=3))
    table.weight = torch.nn.Parameter(_ramp(5))
    _wire_buffer(table, monkeypatch, param_dtype=torch.bfloat16)

    # The buffer exists before any batch: its geometry comes from the config.
    assert _FakeElasticBuffer.size_hint_args == [(5, _DIM, torch.float32)]
    assert len(_FakeElasticBuffer.instances) == 1
    assert _FakeElasticBuffer.instances[0].num_max_tokens_per_rank == 3
    assert _FakeElasticBuffer.instances[0].group == "fake-engram-group"

    row_ids = torch.tensor([0, 2, 2], dtype=torch.int64)
    output = table._distributed_lookup(row_ids)
    torch.testing.assert_close(output, table.weight.index_select(0, row_ids))

    table._distributed_lookup(row_ids).sum().backward()
    assert len(_FakeElasticBuffer.instances) == 1

    with pytest.raises(ValueError, match="exceeds its fixed per-rank capacity"):
        table._distributed_lookup(torch.tensor([0, 1, 2, 3], dtype=torch.int64))


def test_host_table_requires_its_buffer_to_exist():
    table = HostOffloadEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(_ramp(5))
    table.ep_mesh = _FakeEPMesh()

    with pytest.raises(RuntimeError, match="has not been created"):
        table._distributed_lookup(torch.tensor([0, 1], dtype=torch.int64))


def _out_of_range_case():
    """Build a bridge whose EngramFetchGrad returns IDs outside the shard."""
    buffer = _FakeElasticBuffer(
        "group",
        num_cpu_bytes=2 * 1024 * 1024,
        num_max_tokens_per_rank=4,
        with_grad=True,
    )
    buffer.engram_write(torch.zeros(5, _DIM))
    # ops-transformer PR 10952 clamps the per-core range so owner-local IDs stay
    # inside the shard. The bridge no longer filters them on device, so a pack
    # that regresses has to fail loudly rather than lose gradient rows.
    buffer.extra_unique_local_entries = (-1, 5, 7)
    return buffer, torch.tensor([0, 2, 2, 4], dtype=torch.int64)


def test_host_offload_surfaces_out_of_range_local_rows():
    table = HostOffloadEngramTable(_host_table_config(capacity=4))
    table.weight = torch.nn.Parameter(_ramp(5))
    table._mark_host_weight()
    buffer, row_ids = _out_of_range_case()

    output = _EngramFetchSparseOffload.apply(
        table.weight, row_ids, EngramBufferHandle(buffer), table._table_handle, table._grad_keepalive
    )

    with pytest.raises(IndexError, match=r"local row outside \[0, 5\)"):
        output.backward(torch.ones(4, _DIM))


def test_host_storage_registers_the_authoritative_shard_once(monkeypatch):
    table = _host_table_config().build()
    table.weight = torch.nn.Parameter(_ramp(5))
    _wire_buffer(table, monkeypatch)
    table._init_self_parameters()
    buffer = table._elastic_buffer

    assert buffer.storage.data_ptr() == table.weight.data_ptr()
    row_ids = torch.tensor([0, 2], dtype=torch.int64)
    torch.testing.assert_close(table._distributed_lookup(row_ids), table.weight[row_ids])
    with torch.no_grad():
        table.weight.add_(2.0)
    torch.testing.assert_close(table._distributed_lookup(row_ids), table.weight[row_ids])
    assert buffer.write_count == 1


def test_host_table_rejects_nonpositive_explicit_capacity():
    with pytest.raises(ValueError, match="must be positive"):
        HostOffloadEngramTable(_host_table_config(capacity=0))


def test_host_table_accepts_empty_first_request(monkeypatch):
    _FakeElasticBuffer.reset()
    table = HostOffloadEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(_ramp(5))
    _wire_buffer(table, monkeypatch)

    output = table._distributed_lookup(torch.empty(0, dtype=torch.int64))
    assert output.shape == (0, _DIM)
    output.sum().backward()
    assert table.weight.grad is None
    torch.testing.assert_close(table.pending_sparse_grad().to_dense(), torch.zeros_like(table.weight))
    assert _FakeElasticBuffer.instances[0].num_max_tokens_per_rank == 4


def test_host_offload_bridge_accumulates_sparse_cpu_gradient():
    table = HostOffloadEngramTable(_host_table_config(capacity=4))
    table.weight = torch.nn.Parameter(_ramp(5))
    table._mark_host_weight()
    row_ids = torch.tensor([0, 2, 2, 4], dtype=torch.int64)
    buffer = _FakeElasticBuffer(
        "group",
        num_cpu_bytes=2 * 1024 * 1024,
        num_max_tokens_per_rank=4,
        with_grad=True,
    )
    buffer.engram_write(table.weight.detach())

    output = _EngramFetchSparseOffload.apply(
        table.weight, row_ids, EngramBufferHandle(buffer), table._table_handle, table._grad_keepalive
    )
    torch.testing.assert_close(output, table.weight.index_select(0, row_ids))
    grad_output = _ramp(4, start=1)
    output.backward(grad_output)

    assert table.weight.grad is None
    sparse_grad = table._pending_sparse_grad
    assert sparse_grad is not None and sparse_grad.is_sparse and sparse_grad.is_coalesced()
    assert sparse_grad._indices().tolist() == [[0, 2, 4]]
    expected = torch.zeros_like(table.weight)
    expected.index_add_(0, row_ids, grad_output)
    torch.testing.assert_close(sparse_grad.to_dense(), expected)


def test_host_offload_sparse_adam_updates_only_hit_rows():
    table = HostOffloadEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(_ramp(5))
    table._ep_rank = 1
    table._ep_size = 2
    table._mark_host_weight()
    model = torch.nn.Module()
    model.table = table
    model.dense = torch.nn.Linear(3, 3, bias=False)
    config = HostSparseOptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=r"table\.weight$",
                optimizer_name="SparseAdam",
                optimizer_kwargs={
                    "lr": 0.1,
                    "betas": (0.9, 0.95),
                    "eps": 1e-6,
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.01},
            ),
        ],
    )
    optimizers = HostSparseOptimizersContainer(config, model_parts=[model])
    assert any(isinstance(optimizer, torch.optim.SparseAdam) for optimizer in optimizers)

    before = table.weight.detach().clone()
    table.accumulate_sparse_gradient(
        torch.tensor([1, 3, 3], dtype=torch.int32),
        torch.ones(3, _DIM),
    )
    optimizers.step()
    torch.testing.assert_close(table.weight[[0, 2, 4]], before[[0, 2, 4]])
    assert not torch.equal(table.weight[[1, 3]], before[[1, 3]])

    state = optimizers.state_dict()
    assert any("weight.ep_shard_00001_of_00002" in key for key in state)
    optimizers.zero_grad()
    assert table.weight.grad is None
    assert table._pending_sparse_grad is None


def test_each_engram_table_gets_its_own_communicator(monkeypatch):
    """ElasticBuffer keys its device context by HCCL communicator name.

    Two tables sharing one communicator make the second ElasticBuffer fail with
    "Create HCCL context memory failed", so every table must get its own group.
    """
    from torchtitan_npu.override.deepseek_v41.engram import ascendc as ascendc_mod

    created = []

    class _Group:
        def __init__(self, tag):
            self.tag = tag

    def fake_new_group(*, ranks, backend, use_local_synchronization):
        group = _Group(len(created))
        created.append((tuple(ranks), backend, use_local_synchronization))
        return group

    # The production code asserts the new group is a ProcessGroup; swap the type
    # so the stub satisfies it without building a real communicator.
    monkeypatch.setattr(ascendc_mod.dist, "ProcessGroup", _Group)
    monkeypatch.setattr(ascendc_mod.dist, "new_group", fake_new_group)
    monkeypatch.setattr(ascendc_mod.dist, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(ascendc_mod, "_ENGRAM_GROUPS", {})

    ep_group = object()
    first = ascendc_mod._dedicated_engram_group(ep_group, layer_id=2)
    second = ascendc_mod._dedicated_engram_group(ep_group, layer_id=15)
    again = ascendc_mod._dedicated_engram_group(ep_group, layer_id=2)

    assert first is not second
    assert again is first
    assert len(created) == 2
    assert all(entry == ((0, 1), "hccl", True) for entry in created)


def test_engram_fetch_fake_follows_the_request_device():
    """The fetched rows live where the request does, not where the table does.

    A host-offload table's weight is a CPU parameter while the kernel writes
    device memory, so deriving the fake's device from the weight puts the whole
    Engram output on CPU inside the graph.
    """
    from torchtitan_npu.ops.ascendc.engram import _engram_fetch_fake

    weight = torch.zeros(4, _DIM)
    indices = torch.zeros(3, dtype=torch.int32, device="meta")

    fetched, _handle = _engram_fetch_fake(weight, indices, EngramBufferHandle())

    assert fetched.device == indices.device
    assert fetched.shape == (3, _DIM)
    assert fetched.dtype == weight.dtype


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_host_offload_backward_produces_a_consumable_gradient(compiled):
    """The sparse backward must return a gradient, not only a side effect.

    A backward that returns None for every input computes nothing a consumer
    depends on, so a compiled backward drops the whole subgraph: the table then
    never receives a gradient and training continues with a frozen table and no
    error. The keepalive gradient is what keeps that chain live.
    """
    table = HostOffloadEngramTable(_host_table_config(capacity=4))
    table.weight = torch.nn.Parameter(_ramp(5))
    table._mark_host_weight()
    reference = torch.nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False, sparse=True)
    optimizer = torch.optim.SparseAdam([table.weight], lr=0.1)
    reference_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.1)
    buffer = _FakeElasticBuffer("group", num_cpu_bytes=2 * 1024 * 1024, num_max_tokens_per_rank=4, with_grad=True)
    buffer.engram_write(table.weight.detach())
    handle = EngramBufferHandle(buffer)
    fetch = _EngramFetchSparseOffload.apply
    if compiled:
        fetch = torch.compile(fetch, backend="aot_eager", fullgraph=True)

    for step, rows in enumerate(([0, 2, 2, 4], [1, 2, 1, 4], [0, 3, 3, 4])):
        row_ids = torch.tensor(rows, dtype=torch.int64)
        before = table.weight.detach().clone()
        output = fetch(table.weight, row_ids, handle, table._table_handle, table._grad_keepalive)
        expected = reference(row_ids)
        torch.testing.assert_close(output, expected)

        grad_output = _ramp(4, start=1) * (-1.0 if step == 1 else 1.0)
        output.backward(grad_output)
        expected.backward(grad_output)

        assert table._grad_keepalive.grad is not None
        pending = table.pending_sparse_grad()
        assert pending is not None and pending.is_sparse and pending.is_coalesced()
        expected_grad = reference.weight.grad.coalesce()
        torch.testing.assert_close(pending.indices(), expected_grad.indices())
        torch.testing.assert_close(pending.values(), expected_grad.values())
        table.prepare_sparse_optimizer_step()
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(table.weight, reference.weight)
        hit_rows = row_ids.unique()
        missed_rows = torch.tensor(sorted(set(range(5)) - set(rows)))
        assert not torch.equal(table.weight[hit_rows], before[hit_rows])
        torch.testing.assert_close(table.weight[missed_rows], before[missed_rows], rtol=0, atol=0)

        table.mark_sparse_step_complete()
        table.clear_sparse_gradient()
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        table._grad_keepalive.grad = None
