# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two EP ranks compare Torch Host training to a full sparse embedding."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from torchtitan_npu.models.deepseek_v41.engram_host import HostEngramTable


def table_config():
    return HostEngramTable.Config(
        vocab_size=16,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(11,),
        embedding_dim=4,
        num_embeddings=12,
        require_token_id_map=False,
        pin_memory=False,
    )


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        rank = dist.get_rank()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        dims = SimpleNamespace(get_optional_mesh=lambda name: mesh)
        table = table_config().build()
        table.parallelize(dims)
        full = torch.arange(48, dtype=torch.float32).view(12, 4) / 16
        with torch.no_grad():
            table.weight.copy_(full[rank * 6 : (rank + 1) * 6])
        reference = torch.nn.Embedding.from_pretrained(full.clone(), freeze=False, sparse=True)
        opt = torch.optim.SparseAdam([table.weight], lr=0.05)
        ref_opt = torch.optim.SparseAdam(reference.parameters(), lr=0.05)
        # Uneven splits, duplicate remote rows, then an empty request rank and
        # an owner with no hits; the final step checks the resumed optimizer.
        batches = [([7, 1, 7, 11], [2, 7, 8]), ([2, 2, 1], []), ([11, 0], [0, 11, 11])]
        for step, requests in enumerate(batches):
            ids = torch.tensor(requests[rank], dtype=torch.long)
            upstream = torch.arange(ids.numel() * 4, dtype=torch.float32).view(-1, 4) + 1 + rank
            out = checkpoint(table._distributed_lookup, ids, use_reentrant=False)
            torch.testing.assert_close(out, reference(ids), rtol=0, atol=0)
            out.backward(upstream)
            all_ids = torch.tensor(requests[0] + requests[1], dtype=torch.long)
            all_grads = torch.cat(
                [
                    torch.arange(len(rows) * 4, dtype=torch.float32).view(-1, 4) + 1 + source
                    for source, rows in enumerate(requests)
                ]
            )
            reference(all_ids).backward(all_grads)
            pending = table.pending_sparse_grad()
            assert pending is not None and pending.is_sparse and table.weight.grad is None
            torch.testing.assert_close(
                pending.to_dense(), reference.weight.grad.to_dense()[rank * 6 : (rank + 1) * 6], rtol=0, atol=0
            )
            table.prepare_sparse_optimizer_step()
            opt.step()
            ref_opt.step()
            torch.testing.assert_close(table.weight, reference.weight[rank * 6 : (rank + 1) * 6], rtol=0, atol=0)
            table.clear_sparse_gradient()
            ref_opt.zero_grad(set_to_none=True)
            if step == 1:
                saved_table, saved_opt = copy.deepcopy(table.state_dict()), copy.deepcopy(opt.state_dict())
                resumed = table_config().build()
                resumed.parallelize(dims)
                resumed.load_state_dict(saved_table)
                table = resumed
                opt = torch.optim.SparseAdam([table.weight], lr=0.05)
                opt.load_state_dict(saved_opt)
        print(f"rank {rank}: forward, sparse gradient, SparseAdam and resume agree", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
