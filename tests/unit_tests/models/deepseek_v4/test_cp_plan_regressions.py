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



def test_tensorized_cp_planner_helpers_fullgraph():
    """Protect device planner math, including zero-block ranges, under Dynamo."""
    docs = (37, 41, 63, 115)
    cp_size = 4
    seq_len = sum(docs)
    shard_len = seq_len // cp_size
    cu = torch.tensor(
        [0, *torch.tensor(docs).cumsum(0).tolist()], dtype=torch.int32
    )
    lb = _CountingHeadTail(seq_len, cp_size)
    rearrange = lb._generate_indices(False).reshape(-1).to(torch.int32)
    restore = torch.empty_like(rearrange)
    restore[rearrange] = torch.arange(seq_len, dtype=torch.int32)

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        def segment_fields(c, r):
            geometry = cp_mod._segment_geometry(
                c, r, cp_size=cp_size, shard_len=shard_len
            )
            return geometry.ranks, geometry.doc_starts, geometry.seqlens_k

        segment_fn = torch.compile(
            segment_fields,
            backend="eager",
            fullgraph=True,
            dynamic=True,
        )
        ranks, doc_starts, seqlens_k = segment_fn(cu, rearrange)

        rows = torch.tensor([0, 35, 66, 99, 34, 67, 100], dtype=torch.int32)
        dest = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int64)
        route_fn = torch.compile(
            lambda x, d: cp_mod._routing_tensors(
                x,
                d,
                rank=1,
                shard_len=shard_len,
                cp_size=cp_size,
            ),
            backend="eager",
            fullgraph=True,
            dynamic=True,
        )
        route = route_fn(rows, dest)
        assert len(route) == 5
        recv_src = torch.div(
            rows[(rows // shard_len != dest) & (dest == 1)],
            shard_len,
            rounding_mode="floor",
        ).to(torch.long)
        starts = torch.bincount(recv_src, minlength=cp_size).cumsum(0)
        starts = starts - torch.bincount(recv_src, minlength=cp_size)
        seen = [0] * cp_size
        expected_offsets = []
        for src_rank in recv_src.tolist():
            expected_offsets.append(int(starts[src_rank]) + seen[src_rank])
            seen[src_rank] += 1
        assert route[3].tolist() == expected_offsets

        # ratio=128 intentionally creates segments with no complete plan block.
        geometry = cp_mod._segment_geometry(
            cu, rearrange, cp_size=cp_size, shard_len=shard_len
        )
        p0 = geometry.p0
        q0 = p0 + geometry.seg_lens
        ratio = 128
        A = q0
        block_end = torch.div(q0, ratio, rounding_mode="floor") * ratio
        strip = torch.zeros_like(q0)
        container_fn = torch.compile(
            lambda rk, ds, sk, a, e, st: cp_mod._container_layout(
                rk,
                ds,
                sk,
                a,
                e,
                st,
                ratio=ratio,
                rank=0,
                cp_size=cp_size,
                seq_len=seq_len,
            ),
            backend="eager",
            fullgraph=True,
            dynamic=True,
        )
        gather, kept_per_rank = container_fn(
            geometry.ranks,
            geometry.doc_starts,
            geometry.seqlens_k,
            A,
            block_end,
            strip,
        )
        assert gather.ndim == 1
        assert kept_per_rank.shape == (cp_size,)


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
    assert metadata.seq_len_host == 16
    assert metadata.seq_len == 16
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
