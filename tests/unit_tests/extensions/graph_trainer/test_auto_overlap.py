"""Tests for auto-overlap pass construction and calibration rounds."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY

from torchtitan_npu.extensions.graph_trainer import auto_overlap as graph_trainer_auto_overlap
from torchtitan_npu.extensions.graph_trainer.utils import node_id


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        compile=SimpleNamespace(
            passes=[],
            precompile_artifact_dir="",
        )
    )


def _named_pass(name: str):
    def pass_fn(gm, example_inputs):
        return gm

    pass_fn.__name__ = name
    return pass_fn


def test_auto_overlap_pipeline_is_registered():
    assert PASS_PIPELINE_REGISTRY["npu_auto_overlap"] is graph_trainer_auto_overlap.construct_npu_auto_overlap_passes
    assert (
        PASS_PIPELINE_REGISTRY["mutation-functionalization+npu_auto_overlap"]
        is graph_trainer_auto_overlap.construct_mutation_functionalization_npu_auto_overlap_passes
    )


def test_deepseek_v4_auto_overlap_config_selects_npu_pipeline():
    from torchtitan_npu.models.deepseek_v4.config_registry import (
        graph_trainer_deepseek_v4_flash_43layers_16experts,
        graph_trainer_deepseek_v4_flash_43layers_16experts_auto_overlap,
    )

    manual = graph_trainer_deepseek_v4_flash_43layers_16experts()
    auto = graph_trainer_deepseek_v4_flash_43layers_16experts_auto_overlap()

    assert manual.compile.pass_pipeline == "mutation-functionalization"
    assert auto.compile.pass_pipeline == "mutation-functionalization+npu_auto_overlap"
    assert auto.compile.ep_overlap.enabled


def test_npu_auto_overlap_runs_after_chunk_shape_concretization():
    config = _config()
    passes = [
        _named_pass("before"),
        _named_pass("joint_transformer_block_bucketing_reordering_pass"),
        _named_pass("ep_overlap_schedule_pass"),
        _named_pass("concretize_ep_chunk_symbolic_shapes_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert [graph_trainer_auto_overlap._pass_name(pass_fn) for pass_fn in configured] == [
        "before",
        "joint_transformer_block_bucketing_reordering_pass",
        "concretize_ep_chunk_symbolic_shapes_pass",
        "npu_moe_auto_overlap_pass",
        "regional_inductor_pass",
    ]


def test_npu_auto_overlap_keeps_original_pipeline_without_ep_preparation(caplog):
    config = _config()
    passes = [_named_pass("before"), _named_pass("regional_inductor_pass")]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "ep_overlap_schedule_pass=0" in caplog.text


def test_npu_auto_overlap_keeps_original_pipeline_with_duplicate_manual_pass(caplog):
    config = _config()
    manual_pass = _named_pass("ep_overlap_schedule_pass")
    passes = [
        manual_pass,
        manual_pass,
        _named_pass("concretize_ep_chunk_symbolic_shapes_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "ep_overlap_schedule_pass=2" in caplog.text


def test_npu_auto_overlap_keeps_original_pipeline_without_concretize_pass(caplog):
    config = _config()
    passes = [
        _named_pass("ep_overlap_schedule_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "concretize_ep_chunk_symbolic_shapes_pass=0" in caplog.text


def test_npu_auto_overlap_keeps_precompiled_runtime_pipeline(caplog):
    config = _config()
    config.compile.precompile_artifact_dir = "/tmp/precompiled"
    passes = [_named_pass("cudagraph_pass")]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "precompiled artifact" in caplog.text


@pytest.mark.parametrize(("debug", "expect_dump"), [("0", False), ("1", True)])
def test_npu_auto_overlap_writes_back_scheduler_result(monkeypatch, tmp_path, caplog, debug, expect_dump):
    from torchtitan_npu.extensions.graph_trainer import npu_moe_auto_scheduler

    class Scheduler:
        def __init__(self, candidate_gm, **_kwargs):
            self.candidate_gm = candidate_gm

        def run(self):
            return self.candidate_gm

    monkeypatch.setattr(
        npu_moe_auto_scheduler,
        "NpuMoeAutoOverlapScheduler",
        Scheduler,
    )
    config = _config()
    config.dump_folder = str(tmp_path)
    monkeypatch.setenv("NPU_AUTO_OVERLAP_DEBUG", debug)
    passes = [
        _named_pass("before"),
        _named_pass("ep_overlap_schedule_pass"),
        _named_pass("concretize_ep_chunk_symbolic_shapes_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert [graph_trainer_auto_overlap._pass_name(pass_fn) for pass_fn in configured] == [
        "before",
        "concretize_ep_chunk_symbolic_shapes_pass",
        "npu_moe_auto_overlap_pass",
        "regional_inductor_pass",
    ]
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.output(value)
    gm = torch.fx.GraphModule({}, graph)
    monkeypatch.chdir(tmp_path)
    assert configured[2](gm, ()) is gm
    before_dump = tmp_path / "fx_graphs/auto_overlap_before_rank0.py"
    after_dump = tmp_path / "fx_graphs/auto_overlap_after_rank0.py"
    assert before_dump.is_file() is expect_dump
    assert after_dump.is_file() is expect_dump
    if expect_dump:
        assert "class GraphModule" in after_dump.read_text(encoding="utf-8")
    assert "did not provide a runtime context" in caplog.text


def test_pipeline_factory_preserves_default_preparation_and_partial_passes(monkeypatch):
    from torchtitan.experiments.graph_trainer import passes as upstream_passes

    traced_result, parallel_dims = object(), object()
    config = _config()
    config.compile.ep_overlap = SimpleNamespace(module_fqn="layers.*.moe")
    before = _named_pass("before")
    concretize = _named_pass("concretize_ep_chunk_symbolic_shapes_pass")
    terminal = partial(_named_pass("regional_inductor_pass"))
    default_passes = [
        before,
        partial(_named_pass("ep_overlap_schedule_pass")),
        concretize,
        terminal,
    ]
    factory = Mock(return_value=default_passes)
    monkeypatch.setattr(upstream_passes, "construct_default_graph_passes", factory)

    configured = graph_trainer_auto_overlap.construct_npu_auto_overlap_passes(
        traced_result,
        config,
        parallel_dims=parallel_dims,
    )

    factory.assert_called_once_with(traced_result, config, parallel_dims=parallel_dims)
    assert configured[0] is before
    assert configured[1] is concretize
    assert configured[2].func is graph_trainer_auto_overlap.npu_moe_auto_overlap_pass
    assert configured[2].keywords["config"] is config
    assert configured[2].keywords["traced_result"] is traced_result
    assert configured[2]._requires_runtime_context is True
    assert configured[3] is terminal
    assert len(default_passes) == 4  # Do not mutate the upstream pass list.


def test_mutation_functionalization_pipeline_is_composed_before_auto_overlap(
    monkeypatch,
):
    traced_result, parallel_dims = object(), object()
    config = _config()
    functionalize = _named_pass("functionalize_recompute_mutations_pass")
    manual = _named_pass("ep_overlap_schedule_pass")
    concretize = _named_pass("concretize_ep_chunk_symbolic_shapes_pass")
    base_passes = [functionalize, manual, concretize]
    factory = Mock(return_value=base_passes)
    monkeypatch.setitem(
        PASS_PIPELINE_REGISTRY,
        "mutation-functionalization",
        factory,
    )

    configured = graph_trainer_auto_overlap.construct_mutation_functionalization_npu_auto_overlap_passes(
        traced_result,
        config,
        parallel_dims=parallel_dims,
    )

    factory.assert_called_once_with(
        traced_result,
        config,
        parallel_dims=parallel_dims,
    )
    assert [graph_trainer_auto_overlap._pass_name(pass_fn) for pass_fn in configured] == [
        "functionalize_recompute_mutations_pass",
        "concretize_ep_chunk_symbolic_shapes_pass",
        "npu_moe_auto_overlap_pass",
    ]
    assert base_passes == [functionalize, manual, concretize]


def test_two_whole_graph_rounds_measure_previous_schedule_and_keep_latest_fallback(
    monkeypatch,
    tmp_path,
):
    from torchtitan_npu.extensions.graph_trainer import npu_moe_auto_scheduler as scheduler_module
    from torchtitan_npu.extensions.graph_trainer import whole_graph_benchmark as benchmark_module

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    compute = graph.call_function(torch.ops.aten.neg.default, (x,))
    comm = graph.call_function(torch.ops._c10d_functional.all_reduce.default, (compute, "sum", "test_group"))
    graph.output(comm)
    gm = torch.fx.GraphModule({}, graph)
    schedulers, profiles = [], []
    runner = Mock()

    class Scheduler:
        def __init__(self, candidate, **kwargs):
            assert candidate is gm
            self.kwargs = kwargs
            self.costs_by_node_id = {}
            schedulers.append(self)

        def run(self):
            # A real scheduler calls the cost callbacks synchronously in run().
            self.costs_by_node_id[node_id(compute)] = (
                self.kwargs["compute_cost_fn"](compute) if "compute_cost_fn" in self.kwargs else 10.0
            )
            self.costs_by_node_id[node_id(comm)] = (
                self.kwargs["collective_cost_fn"](comm) if "collective_cost_fn" in self.kwargs else 20.0
            )
            gm.meta["test_schedule_generation"] = len(schedulers)
            return gm

    def profile(candidate, run_candidate, *, profile_node_ids, calibration_round):
        assert candidate is gm
        assert run_candidate is runner
        assert candidate.meta["test_schedule_generation"] == calibration_round
        cid, aid = node_id(compute), node_id(comm)
        assert profile_node_ids == frozenset((cid, aid))
        profiles.append(calibration_round)
        # Round 2 missing collective must reuse round 1's 3ms, not initial 20ms.
        # A measured zero is valid, not a missing mapping.
        costs = {cid: 4.0, aid: 3.0} if calibration_round == 1 else {cid: 0.0}
        return benchmark_module.WholeGraphProfileCosts(costs, costs, frozenset(costs))

    monkeypatch.setattr(scheduler_module, "NpuMoeAutoOverlapScheduler", Scheduler)
    monkeypatch.setattr(benchmark_module, "profile_whole_graph_costs", profile)
    monkeypatch.setattr(
        benchmark_module,
        "make_calibration_runner",
        lambda _traced_result, _runtime_context: runner,
    )
    config = _config()
    config.dump_folder = str(tmp_path)
    monkeypatch.setenv("NPU_AUTO_OVERLAP_DEBUG", "1")
    monkeypatch.chdir(tmp_path)
    passes = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        [_named_pass("ep_overlap_schedule_pass"), _named_pass("concretize_ep_chunk_symbolic_shapes_pass")],
        config,
    )
    result = passes[1](gm, (), runtime_context={"test": "context"})
    assert result is gm
    assert profiles == [1, 2]
    assert len(schedulers) == 3
    assert gm.meta["test_schedule_generation"] == 3
    for scheduler in schedulers[1:]:
        assert scheduler.kwargs["use_profile_compute_costs"] is True
        assert scheduler.kwargs["align_across_ranks"] is False
        assert scheduler.kwargs["canonical_order_override"] is schedulers[0].kwargs["canonical_order_override"]
    assert schedulers[-1].costs_by_node_id[node_id(compute)] == 0.0
    assert schedulers[-1].costs_by_node_id[node_id(comm)] == 3.0
    assert (tmp_path / "fx_graphs/auto_overlap_after_rank0.py").is_file()
