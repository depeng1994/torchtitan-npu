# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.extensions.components import optimizer as host_mod
from torchtitan_npu.models.deepseek_v41.engram_host import HostEngramTable

pytestmark = pytest.mark.cpu
_DIM = 128


def _ramp(rows):
    return torch.arange(rows * _DIM, dtype=torch.float32).reshape(rows, _DIM)


def _host_table_config():
    return HostEngramTable.Config(
        vocab_size=16,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(7,),
        embedding_dim=_DIM,
        num_embeddings=10,
        require_token_id_map=False,
        pin_memory=False,
    )


@pytest.mark.parametrize("parameter_container", [list, iter], ids=["list", "generator"])
def test_global_clip_covers_the_host_sparse_gradient(monkeypatch, parameter_container):
    """The Host sparse gradient joins the norm and takes the same coefficient."""

    max_norm = 1.0
    dense = torch.nn.Parameter(torch.zeros(2))
    dense.grad = torch.tensor([3.0, 0.0])  # dense norm 3
    indices = torch.tensor([0, 1])
    values = torch.zeros(2, _DIM)
    values[0, 0] = 4.0  # sparse norm 4 -> total 5

    table = HostEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(torch.zeros(5, _DIM))
    table._mark_host_weight()
    table.accumulate_sparse_gradient(indices, values)

    class _Mesh:
        def get_group(self):
            return "ep"

    table.ep_mesh = _Mesh()

    def fake_original(parameters, norm_max, norm_type=2.0, *args, **kwargs):
        grads = [p.grad for p in parameters if p.grad is not None]
        total = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
        coefficient = min(float(norm_max) / (float(total) + 1e-6), 1.0)
        for grad in grads:
            grad.mul_(coefficient)
        return total

    monkeypatch.setattr(host_mod, "_ORIGINAL_CLIP_GRAD_NORM", fake_original)
    monkeypatch.setattr(host_mod, "_CLIPPED_HOST_TABLES", {table})
    monkeypatch.setattr(host_mod.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    total_norm = host_mod._clip_grad_norm_with_host_sparse_tables(parameter_container([dense]), max_norm)

    assert total_norm.item() == pytest.approx(5.0, rel=1e-5)
    # Both sides are scaled by max_norm / total_norm = 0.2, not by the
    # dense-only coefficient 1/3.
    assert dense.grad[0].item() == pytest.approx(3.0 * 0.2, rel=1e-4)
    pending = table.pending_sparse_grad()
    assert pending.values().abs().max().item() == pytest.approx(4.0 * 0.2, rel=1e-4)


@pytest.mark.parametrize("layer_order", [(1, 3), (3, 1)], ids=["forward", "reverse"])
def test_global_clip_reduces_replicas_in_layer_order(monkeypatch, layer_order):
    # The registry can iterate in different orders on each rank. Observe the
    # collective boundary only; cross-rank gradient numerics have separate UTs.
    tables = []
    reduced_layers = []
    for layer_id in layer_order:
        config = _host_table_config()
        config.layer_id = layer_id
        table = HostEngramTable(config)
        monkeypatch.setattr(
            table,
            "reduce_sparse_gradient_across_replicas",
            lambda layer_id=layer_id: reduced_layers.append(layer_id),
        )
        tables.append(table)
    monkeypatch.setattr(host_mod, "_CLIPPED_HOST_TABLES", tables)
    monkeypatch.setattr(host_mod, "_ORIGINAL_CLIP_GRAD_NORM", lambda *args, **kwargs: torch.tensor(0.0))

    host_mod._clip_grad_norm_with_host_sparse_tables([], 1.0)

    assert reduced_layers == [1, 3]


def test_global_clip_is_a_noop_without_a_sparse_gradient(monkeypatch):
    dense = torch.nn.Parameter(torch.zeros(2))
    dense.grad = torch.tensor([3.0, 0.0])
    calls = []

    def fake_original(parameters, norm_max, norm_type=2.0, *args, **kwargs):
        calls.append(norm_max)
        return torch.tensor(3.0)

    monkeypatch.setattr(host_mod, "_ORIGINAL_CLIP_GRAD_NORM", fake_original)
    monkeypatch.setattr(host_mod, "_CLIPPED_HOST_TABLES", set())

    assert host_mod._clip_grad_norm_with_host_sparse_tables([dense], 1.0).item() == pytest.approx(3.0)
    assert calls == [1.0]
    assert dense.grad[0].item() == pytest.approx(3.0)


def test_host_offload_model_state_uses_rank_specific_shard_key():
    table = HostEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(_ramp(5))
    table._ep_rank = 1
    table._ep_size = 2
    table._mark_host_weight()

    state = table.state_dict()
    assert "weight" not in state
    assert list(state) == ["weight.ep_shard_00001_of_00002"]

    restored = HostEngramTable(_host_table_config())
    restored.weight = torch.nn.Parameter(torch.zeros(5, _DIM))
    restored._ep_rank = 1
    restored._ep_size = 2
    restored._mark_host_weight()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.weight, table.weight)


class _FakeMeshDims:
    def __init__(self, shard=None, replicate=None):
        self.shard = shard
        self.replicate = replicate


class _FakeReplicaMesh:
    """A mesh that records which axes a caller selected before grouping."""

    def __init__(self, size, ndim=1, sizes=None):
        self._size = size
        self.ndim = ndim
        self._sizes = sizes or {}
        self.selected = None

    def size(self):
        return self._size

    def get_group(self):
        return "replica-group"

    def __getitem__(self, axes):
        self.selected = axes
        size = 1
        for axis in axes:
            size *= self._sizes[axis]
        sub = _FakeReplicaMesh(size, ndim=len(axes), sizes=self._sizes)
        sub.selected = axes
        return sub

    def _flatten(self):
        return _FakeReplicaMesh(self._size, ndim=1, sizes=self._sizes)


def test_a_single_replica_needs_no_cross_replica_reduction():
    """One replica must not reach a collective; there is nobody to sum with."""
    table = HostEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(_ramp(5))
    table._mark_host_weight()

    table.wire_sparse_grad_replicas(edp_mesh=_FakeReplicaMesh(1))
    assert table._replica_group is None

    table.wire_sparse_grad_replicas(edp_mesh=None)
    assert table._replica_group is None

    table.accumulate_sparse_gradient(torch.tensor([0, 2], dtype=torch.int64), _ramp(2))
    before = table.pending_sparse_grad().to_dense().clone()
    # No process group is initialized in this test, so reaching a collective
    # here would raise rather than silently pass.
    table.reduce_sparse_gradient_across_replicas()
    torch.testing.assert_close(table.pending_sparse_grad().to_dense(), before)


def test_replicas_of_a_shard_are_wired_from_the_sparse_mesh():
    table = HostEngramTable(_host_table_config())
    table.wire_sparse_grad_replicas(edp_mesh=_FakeReplicaMesh(2))
    assert table._replica_group == "replica-group"
    assert table._replica_size == 2


def test_replica_group_excludes_the_expert_parallel_axis():
    """The sparse storage mesh carries EP; reducing over it would sum shards.

    Rows are partitioned along EP, so two EP ranks own disjoint ranges. Folding
    that axis into the replica group adds one shard's rows into another's.
    """
    table = HostEngramTable(_host_table_config())
    sparse_mesh = _FakeReplicaMesh(8, ndim=3, sizes={"dp_replicate": 2, "efsdp": 2, "ep": 2})

    table.wire_sparse_grad_replicas(
        edp_mesh=sparse_mesh,
        edp_mesh_dims=_FakeMeshDims(shard="efsdp", replicate="dp_replicate"),
    )

    assert sparse_mesh.selected == ("dp_replicate", "efsdp")
    assert "ep" not in sparse_mesh.selected
    assert table._replica_size == 4


def test_replica_group_is_empty_without_data_parallel_axes():
    table = HostEngramTable(_host_table_config())
    sparse_mesh = _FakeReplicaMesh(2, ndim=1, sizes={"ep": 2})

    table.wire_sparse_grad_replicas(edp_mesh=sparse_mesh, edp_mesh_dims=_FakeMeshDims())

    assert table._replica_group is None
