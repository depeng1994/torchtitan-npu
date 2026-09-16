# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One rank of the Host Engram cross-replica sparse gradient reduction."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan_npu.models.deepseek_v41.engram_host import HostEngramTable

_DIM = 128
_ROWS = 5


def _table() -> HostEngramTable:
    config = HostEngramTable.Config(
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
    table = HostEngramTable(config)
    table.weight = torch.nn.Parameter(torch.zeros(_ROWS, _DIM))
    table._mark_host_weight()
    return table


def main() -> None:
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="gloo")

    table = _table()
    ep_size = dist.get_world_size() // 2
    mesh = init_device_mesh("cpu", (2, ep_size), mesh_dim_names=("efsdp", "ep"))
    table.wire_sparse_grad_replicas(edp_mesh=mesh, edp_mesh_dims=DataParallelMeshDims(shard="efsdp"))
    ep_rank = rank % ep_size
    replica_rank = rank // ep_size
    assert dist.get_process_group_ranks(table._replica_group) == [ep_rank, ep_rank + ep_size]
    scale = ep_rank + 1


    # Deliberately unequal row counts, with row 2 touched by both replicas so
    # the reduction has something to sum rather than just concatenate.
    if replica_rank == 0:
        ids = torch.tensor([0, 2, 4], dtype=torch.int64)
        values = torch.ones(3, _DIM) * scale
    else:
        ids = torch.tensor([2], dtype=torch.int64)
        values = torch.ones(1, _DIM) * (10.0 * scale)
    table.accumulate_sparse_gradient(ids, values)

    table.reduce_sparse_gradient_across_replicas()

    reduced = table.pending_sparse_grad()
    dense = reduced.to_dense()
    expected = torch.zeros(_ROWS, _DIM)
    expected[0] = 1.0
    expected[2] = 11.0
    expected[4] = 1.0
    expected *= scale

    report = {
        "ok": bool(torch.equal(dense, expected)),
        "rows": sorted(reduced.indices()[0].tolist()),
        "row2": dense[2, 0].item(),
    }
    # A replica with no hits must still participate in the next reduction.
    table.clear_sparse_gradient()
    if replica_rank == 0:
        table.accumulate_sparse_gradient(torch.tensor([1]), torch.full((1, _DIM), float(scale)))
    table.reduce_sparse_gradient_across_replicas()
    expected.zero_()
    expected[1] = scale
    torch.testing.assert_close(table.pending_sparse_grad().to_dense(), expected, rtol=0, atol=0)
    with open(os.environ["OUT"], "w") as handle:
        json.dump(report, handle)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
