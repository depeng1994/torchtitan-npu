"""Small structural regressions for DeepSeek-V4 CP plan construction."""

import pytest
import torch
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4 import model as model_mod
from torchtitan_npu.models.deepseek_v4 import token_dispatcher as cp_mod

pytestmark = pytest.mark.cpu


class _CountingHeadTail(_HeadTailLoadBalancer):
    def __init__(self, seq_len: int, cp_size: int):
        super().__init__(seq_length=seq_len, world_size=cp_size, device="cpu")
        self.forward_calls = 0

    def _generate_indices(self, restore: bool = False):
        if not restore:
            self.forward_calls += 1
        return super()._generate_indices(restore=restore)


def test_cp_plan_builds_only_current_rank_common_metadata(monkeypatch):
    """Planner geometry must not require one CPVarlenMetadata per CP rank."""
    docs = (37, 41, 63, 115)
    cp_size = 4
    seq_len = sum(docs)
    cu = torch.tensor(
        [0, *torch.tensor(docs).cumsum(0).tolist()], dtype=torch.int32
    )
    varlen = VarlenMetadata(
        cu_seq_q=cu, cu_seq_k=cu, max_q=max(docs), max_k=max(docs)
    )
    lb = _CountingHeadTail(seq_len, cp_size)

    from_global = cp_mod.CPVarlenMetadata.from_global.__func__
    calls = 0

    def counted_from_global(cls, *args, **kwargs):
        nonlocal calls
        calls += 1
        return from_global(cls, *args, **kwargs)

    monkeypatch.setattr(
        cp_mod.CPVarlenMetadata,
        "from_global",
        classmethod(counted_from_global),
    )

    cp_meta, plans, window = cp_mod.build_cp_plan(
        varlen,
        lb,
        rank=0,
        cp_size=cp_size,
        shard_len=seq_len // cp_size,
        window_size=8,
        ratios=[1, 4, 128],
    )

    assert calls == 1
    assert lb.forward_calls == 1
    assert int(cp_meta.cu_seq_q[-1]) == seq_len // cp_size
    assert set(plans) == {1, 4, 128}
    assert window is not None


def test_tensorized_cp_routing_matches_reference_and_fullgraph():
    """Protect routing equality and Dynamo fullgraph capture."""
    rows = torch.tensor([0, 1, 35, 66, 99, 34, 67, 100, 100, 7, 70, 103])
    dest = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    cp_size, shard_len, rank = 4, 32, 1

    # Python oracle for the old routing semantics.
    foreign = [[] for _ in range(cp_size)]
    for pos, dst in zip(rows.tolist(), dest.tolist(), strict=True):
        if pos // shard_len != dst:
            foreign[dst].append(pos)
    send = [[] for _ in range(cp_size)]
    send_splits = [[0] * cp_size for _ in range(cp_size)]
    recv_splits, recv_offsets = [], []
    for dst, positions in enumerate(foreign):
        counts = [0] * cp_size
        for pos in positions:
            src = pos // shard_len
            send[src].append(pos % shard_len)
            send_splits[src][dst] += 1
            counts[src] += 1
        starts, total = [0] * cp_size, 0
        for src, count in enumerate(counts):
            starts[src], total = total, total + count
        seen, offsets = [0] * cp_size, []
        for pos in positions:
            src = pos // shard_len
            offsets.append(starts[src] + seen[src])
            seen[src] += 1
        recv_splits.append(counts)
        recv_offsets.append(offsets)
    expected = (send[rank], send_splits[rank], recv_splits[rank], recv_offsets[rank])

    route = cp_mod._routing_tensors(
        rows, dest, rank=rank, shard_len=shard_len, cp_size=cp_size
    )
    assert [x.tolist() for x in route] == list(expected)

    rank_rows = rows[dest == rank]
    expected_order, recv_of = [], {}
    for pos in rank_rows.tolist():
        if pos // shard_len == rank:
            expected_order.append(pos - rank * shard_len)
        else:
            if pos not in recv_of:
                recv_of[pos] = len(recv_of)
            expected_order.append(shard_len + recv_of[pos])
    order, unique = cp_mod._row_order_tensor(
        rank_rows, rank=rank, shard_len=shard_len, cp_size=cp_size
    )
    assert order.tolist() == expected_order
    assert int(unique) == len(recv_of)

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled_route = torch.compile(
            lambda r, d: cp_mod._routing_tensors(
                r, d, rank=rank, shard_len=shard_len, cp_size=cp_size
            ), backend="eager", fullgraph=True, dynamic=True,
        )
        compiled_order = torch.compile(
            lambda r: cp_mod._row_order_tensor(
                r, rank=rank, shard_len=shard_len, cp_size=cp_size
            ), backend="eager", fullgraph=True, dynamic=True,
        )
        assert [x.tolist() for x in compiled_route(rows, dest)] == list(expected)
        got_order, got_unique = compiled_order(rank_rows)
        assert got_order.tolist() == expected_order
        assert int(got_unique) == len(recv_of)


def test_build_attention_masks_cp_runs_real_metadata_wiring(monkeypatch):
    """CP producer/consumer wiring reaches the real planner and skips plain plans."""
    cu = torch.tensor([0, 8, 16], dtype=torch.int32)
    varlen = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=8, max_k=8)
    seen = {}

    class _FakeCPMesh:
        device_type = "cpu"

        def size(self, dim=None):
            assert dim in (None, 0)
            return 2

        def get_local_rank(self):
            return 0

    class _Stub:
        mtp_layers = None
        compress_ratios = (1, 4, 128)
        window_size = 4
        _lightning_indexer_metadata = None
        _metadata_extension = None
        _build_cp_metadata = model_mod.DeepSeekV4Model._build_cp_metadata

        def get_attention_masks(self, *, positions):
            seen["positions_before_shard"] = positions
            return varlen

    def fake_cp_shard(
        cp_mesh,
        tensors,
        seq_dims,
        load_balancer_type,
        seq_dim,
    ):
        seen["cp_shard_mesh"] = cp_mesh
        seen["cp_shard_load_balancer_type"] = load_balancer_type
        seen["cp_shard_seq_dims"] = seq_dims
        seen["cp_shard_seq_dim"] = seq_dim
        # CPU wiring test boundary only: production cp_shard owns the actual
        # HeadTail tensor placement.  Return rank-0-shaped tensors so the real
        # _build_cp_metadata -> build_cp_plan path remains under test.
        local = tuple(t[:, : t.shape[1] // 2].clone() for t in tensors)
        return local, None

    def fail_plain_builder(*args, **kwargs):
        raise AssertionError(
            "CP path must not build plain global compressed metadata"
        )

    monkeypatch.setattr(model_mod, "cp_shard", fake_cp_shard)
    monkeypatch.setattr(
        model_mod,
        "build_compressed_varlen_metadata",
        fail_plain_builder,
    )

    positions = torch.arange(16).view(1, -1)
    inputs = torch.zeros((1, 16), dtype=torch.long)
    labels = torch.zeros((1, 16), dtype=torch.long)
    extra_kwargs = {"positions": positions}
    cp_mesh = _FakeCPMesh()

    out_inputs, out_labels, out_kwargs = (
        model_mod.DeepSeekV4Model.build_attention_masks(
            _Stub(),
            inputs,
            labels,
            extra_kwargs,
            cp_mesh=cp_mesh,
            load_balancer_type="headtail",
        )
    )

    metadata = out_kwargs["attention_masks"]
    assert isinstance(metadata, model_mod.CompressedVarlenMetadata)
    assert int(metadata.varlen.cu_seq_q[-1]) == 8
    assert set(metadata.plans) == {1, 4, 128}
    assert metadata.window is not None

    assert out_inputs.shape == (1, 8)
    assert out_labels.shape == (1, 8)
    assert out_kwargs["positions"].shape == (1, 8)
    assert seen["positions_before_shard"] is positions
    assert seen["cp_shard_mesh"] is cp_mesh
    assert seen["cp_shard_load_balancer_type"] == "headtail"
    assert seen["cp_shard_seq_dims"] is None
    assert seen["cp_shard_seq_dim"] == 1
