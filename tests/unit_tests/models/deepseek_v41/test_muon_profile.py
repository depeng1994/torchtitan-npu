# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU coverage for the V4.1 Muon profile.

Independent positive/negative FQN examples from the real model verify
the muon_pattern; mesh axis sets are checked exactly; placement types
use imported classes with dim/block_size.  Config-level only; NPU
kernel execution is validated on the target hardware.
"""

import re
from collections import Counter

import pytest
from torch.distributed.tensor import Shard
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.distributed.parallel_dims import MeshAxisName

from torchtitan_npu.models.deepseek_v41.config_registry import (
    _v41_muon_profile,
    deepseek_v41_flash_40layers_16experts_cc12m,
)
from torchtitan_npu.models.deepseek_v41.model_registry import model_registry


@pytest.fixture(scope="module")
def spec():
    return model_registry("deepseek_v41_flash_40layers_16experts_vision")


@pytest.fixture(scope="module")
def profile(spec):
    return _v41_muon_profile(spec)


@pytest.fixture(scope="module")
def layouts(profile):
    return profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]


class TestDefaultRecipe:
    def test_default_is_native_adamw(self):
        cfg = deepseek_v41_flash_40layers_16experts_cc12m()
        assert cfg.optimizer.name == "native"
        assert len(cfg.optimizer.param_groups) == 1
        assert cfg.optimizer.param_groups[0].optimizer_name == "AdamW"


class TestMuonPattern:
    POSITIVE = [
        "layers.0.attention.wq_a.weight",
        "layers.0.attention.wq_b.weight",
        "layers.0.attention.wo_a.weight",
        "layers.0.attention.wkv.weight",
        "layers.0.attention.wo_b.weight",
        "layers.0.moe.shared_experts.w1.weight",
        "layers.5.moe.routed_experts.inner_experts.w1_EFD",
        "layers.10.moe.router.gate.weight",
        "layers.20.hc_attn_pre.hc_fn",
        "layers.30.hc_ffn_pre.hc_fn",
    ]

    NEGATIVE = [
        "vision_encoder.patch_embed.proj.weight",
        "vision_encoder.blocks.0.attn.wqkv.weight",
        "vision_encoder.norm.weight",
        "layers.2.attention.indexer.wq_b.weight",
        "layers.2.attention.indexer.weights_proj.weight",
        "layers.2.attention.indexer.k_norm.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.attention.q_norm.weight",
        "layers.0.attention.kv_norm.weight",
        "layers.0.attention.attn_sink",
        "layers.0.moe.router.bias_vl",
        "tok_embeddings.weight",
        "lm_head.weight",
        "norm.weight",
        "image_marker_embeddings.image_start",
    ]

    def test_pattern_matches_positive(self, profile):
        pattern = re.compile(profile.muon_pattern)
        for fqn in self.POSITIVE:
            assert pattern.search(fqn), f"expected Muon: {fqn}"

    def test_pattern_rejects_negative(self, profile):
        pattern = re.compile(profile.muon_pattern)
        for fqn in self.NEGATIVE:
            assert not pattern.search(fqn), f"expected AdamW: {fqn}"

    def test_layout_fqns_all_positive(self, profile):
        pattern = re.compile(profile.muon_pattern)
        for fqn in profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]:
            assert pattern.search(fqn), f"layout FQN: {fqn}"


class TestMuonPlacement:
    def test_wq_b_block_shard_all_dense_axes(self, layouts):
        layout = layouts["layers.0.attention.wq_b.weight"]
        dp = MeshAxisName.DP_SHARD.value
        dp_cp = f"{dp}_{MeshAxisName.CP.value}"
        axes = set(layout.shardings_by_mesh_axis)
        assert {dp, dp_cp} <= axes
        for axis in (dp, dp_cp):
            p = layout.shardings_by_mesh_axis[axis]
            assert isinstance(p, BlockShard), f"wq_b {axis}: {type(p).__name__}"
            assert p.dim == 0
            assert p.block_size == 512

    def test_wo_a_block_shard(self, layouts, spec):
        layout = layouts["layers.0.attention.wo_a.weight"]
        bs = spec.model.layers[0].attention.wo_a.out_features
        dp = MeshAxisName.DP_SHARD.value
        dp_cp = f"{dp}_{MeshAxisName.CP.value}"
        assert set(layout.shardings_by_mesh_axis) == {dp, dp_cp}
        for axis in (dp, dp_cp):
            p = layout.shardings_by_mesh_axis[axis]
            assert isinstance(p, BlockShard), f"wo_a {axis}: {type(p).__name__}"
            assert p.dim == 0
            assert p.block_size == bs

    def test_routed_shard_all_axes(self, layouts):
        layout = layouts["layers.0.moe.routed_experts.inner_experts.w1_EFD"]
        dp = MeshAxisName.DP_SHARD.value
        dp_cp = f"{dp}_{MeshAxisName.CP.value}"
        expected = {dp, dp_cp, MeshAxisName.EFSDP.value, MeshAxisName.EP.value}
        assert set(layout.shardings_by_mesh_axis) == expected
        for axis, p in layout.shardings_by_mesh_axis.items():
            assert isinstance(p, Shard), f"routed {axis}: {type(p).__name__}"
            assert p.dim == 0

    def test_owned_dense_projections(self, layouts):
        for fqn in (
            "layers.0.attention.wq_a.weight",
            "layers.0.attention.wkv.weight",
            "layers.0.moe.router.gate.weight",
            "layers.0.hc_attn_pre.hc_fn",
        ):
            layout = layouts[fqn]
            assert MeshAxisName.DP_SHARD.value in layout.shardings_by_mesh_axis
            for axis, p in layout.shardings_by_mesh_axis.items():
                assert isinstance(p, Owned), f"{fqn} {axis}: {type(p).__name__}"


class TestMuonBuckets:
    def test_bucket_fqns_match_layout(self, profile, layouts):
        buckets = profile.optimizer_factory_kwargs["DistMuon"]["bucket_configs"]
        all_fqns = []
        for b in buckets:
            assert len(b.patterns) > 0
            all_fqns.extend(b.patterns)
        assert set(all_fqns) == set(layouts.keys())

    def test_no_duplicate_bucket(self, profile):
        buckets = profile.optimizer_factory_kwargs["DistMuon"]["bucket_configs"]
        all_fqns = []
        for b in buckets:
            all_fqns.extend(b.patterns)
        dupes = [f for f, c in Counter(all_fqns).items() if c > 1]
        assert not dupes, f"duplicates: {dupes[:5]}"

    def test_bucket_count(self, profile, spec):
        buckets = profile.optimizer_factory_kwargs["DistMuon"]["bucket_configs"]
        assert len(buckets) == len(spec.model.layers) * 3
