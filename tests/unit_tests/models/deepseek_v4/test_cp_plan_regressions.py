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


def test_build_attention_masks_cp_skips_plain_global_compressed_builder(monkeypatch):
    """CP wiring consumes global varlen directly and must skip plain plans."""
    cu = torch.tensor([0, 8, 16], dtype=torch.int32)
    varlen = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=8, max_k=8)
    sentinel = object()
    seen = {}

    class _Stub:
        mtp_layers = None
        compress_ratios = (1, 4, 128)
        _lightning_indexer_metadata = None
        _metadata_extension = None

        def get_attention_masks(self, *, positions):
            seen["positions"] = positions
            return varlen

        def _build_cp_metadata(
            self,
            inputs,
            labels,
            positions,
            global_varlen,
            cp_mesh,
            load_balancer_type,
            mtp_batch,
        ):
            seen["global_varlen"] = global_varlen
            seen["cp_mesh"] = cp_mesh
            seen["load_balancer_type"] = load_balancer_type
            return inputs, labels, positions, sentinel, mtp_batch

    def fail_plain_builder(*args, **kwargs):
        raise AssertionError(
            "CP path must not build plain global compressed metadata"
        )

    monkeypatch.setattr(
        model_mod,
        "build_compressed_varlen_metadata",
        fail_plain_builder,
    )

    positions = torch.arange(16).view(1, -1)
    inputs = torch.zeros((1, 16), dtype=torch.long)
    labels = torch.zeros((1, 16), dtype=torch.long)
    extra_kwargs = {"positions": positions}
    cp_mesh = object()

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

    assert out_inputs is inputs
    assert out_labels is labels
    assert out_kwargs["attention_masks"] is sentinel
    assert seen["global_varlen"] is varlen
    assert seen["cp_mesh"] is cp_mesh
    assert seen["load_balancer_type"] == "headtail"
