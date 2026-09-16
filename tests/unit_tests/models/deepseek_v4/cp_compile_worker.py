"""CPU Inductor/SAC regression for the real uneven CP collective."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.distributed.activation_checkpoint import SelectiveAC

from torchtitan_npu.models.deepseek_v4.metadata import CompressedBlockLayout
from torchtitan_npu.models.deepseek_v4.token_dispatcher import (
    CPTokenDispatcher,
    WindowPlan,
    _build_exchange_plan,
)


class DispatchBlock(torch.nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.dispatcher = CPTokenDispatcher.Config().build()
        self.dispatcher.wire_meshes(cp_mesh=mesh)

    def forward(self, x, window, container):
        # Two gathers model the compressor's projected KV and score paths.
        kv = self.dispatcher.gather(x, window)
        score = self.dispatcher.gather(x * 2, window)
        return self.dispatcher.select(kv.square() + score, container)


def main():
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = DeviceMesh("cpu", list(range(world)))
        module = SelectiveAC.Config().build()._wrap_block(DispatchBlock(mesh))
        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return torch._inductor.compile(gm, inputs)

        compiled = torch.compile(module, backend=backend, fullgraph=True)
        # Keep the zero/one split pattern fixed while varying all other sizes.
        # Dynamo intentionally specializes dimensions of size zero and one;
        # the final batches exercise those transitions without a graph-count limit.
        steps = [*range(6), *([-2] if world > 2 else []), -1, 6]
        for step in steps:
            counts = [0 if peer == rank else 2 + step + (rank + 2 * peer) % world for peer in range(world)]
            received = [0 if peer == rank else 2 + step + (peer + 2 * rank) % world for peer in range(world)]
            send = [i for count in counts for i in range(count)]
            num_recv = sum(received)
            exchange = _build_exchange_plan((send, counts, received, list(range(num_recv))), "cpu")
            # Reverse received rows to exercise index backward, including a
            # duplicate selected row so the expected gradient is not uniform.
            indices = torch.arange(num_recv - 1, -1, -1)
            window = WindowPlan(
                exchange=exchange,
                gather_indices=indices + 32,
                cu_seqlens_ori_kv=torch.tensor([0, num_recv], dtype=torch.int32),
            )
            selected = torch.tensor([0, 0, num_recv - 1] if num_recv else [], dtype=torch.int64)
            container = CompressedBlockLayout(
                cu_seqlens_cmp_k=None,
                block_remainder=None,
                gather_indices=None,
                compressed_rows=selected,
                out_width=5 + step,
            )
            for tensor in (exchange.send_indices, exchange.recv_offsets, indices, window.gather_indices):
                torch._dynamo.maybe_mark_dynamic(tensor, 0)
            x = (torch.arange(64, dtype=torch.float32).reshape(1, 32, 2) + rank * 100).requires_grad_()
            actual = compiled(x, window, container)
            # Independent global row oracle: each source sends a prefix to us.
            sources = [peer for peer in range(world) if peer != rank]
            rows = torch.cat(
                [
                    torch.arange(2 * (2 + step + (peer + 2 * rank) % world), dtype=torch.float32).reshape(-1, 2)
                    + peer * 100
                    for peer in sources
                ]
            )
            expected = torch.zeros(1, 5 + step, 2)
            chosen = rows.flip(0)[selected]
            expected[0, : selected.numel()] = chosen.square() + 2 * chosen
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
            grad = torch.zeros_like(x)
            for destination in range(world):
                count = 2 + step + (rank + 2 * destination) % world
                if destination == rank or count == 0:
                    continue
                source_order = [
                    peer
                    for peer in range(world)
                    if peer != destination and 2 + step + (peer + 2 * destination) % world > 0
                ]
                if rank == source_order[-1]:
                    grad[0, count - 1] += 2 * (2 * x.detach()[0, count - 1] + 2)
                if rank == source_order[0]:
                    grad[0, 0] += 2 * x.detach()[0, 0] + 2
            torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
            if step == 5:
                assert len(graphs) == 1, f"rank {rank}: compiled {len(graphs)} graphs"
        print(
            f"rank {rank}: CP={world}, SAC + Inductor, exact outputs/gradients; "
            "one graph for 6 dynamic batches; zero/one transitions passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
