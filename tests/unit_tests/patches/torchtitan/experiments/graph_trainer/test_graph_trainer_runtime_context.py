"""Tests for the GraphTrainer runtime-context compatibility patch."""

from types import SimpleNamespace

import torch


def test_runtime_context_is_passed_only_to_opted_in_passes():
    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        graph_trainer_runtime_context as trainer_patch,
    )

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    gm = torch.fx.GraphModule({}, graph)
    context = {"runtime": "state"}
    seen = []

    def ordinary(candidate, example_inputs):
        seen.append(("ordinary", example_inputs))
        return candidate

    def runtime_aware(candidate, example_inputs, *, runtime_context=None):
        seen.append(("runtime", example_inputs, runtime_context))
        return candidate

    runtime_aware._requires_runtime_context = True
    result = trainer_patch.patched_apply_graph_passes(
        gm,
        (),
        [ordinary, runtime_aware],
        runtime_context=context,
    )

    assert result is gm
    assert seen == [("ordinary", []), ("runtime", [], context)]


def test_runtime_context_is_built_lazily_during_tracing(monkeypatch):
    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        graph_trainer_runtime_context as trainer_patch,
    )

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    gm = torch.fx.GraphModule({}, graph)
    traced_result = SimpleNamespace(gm=gm)
    model = object()
    inputs, labels, tokens = object(), object(), object()
    extra_kwargs = {"metadata": object()}
    train_context = object()
    instance = SimpleNamespace(
        _traced_step=traced_result,
        train_context=train_context,
    )
    seen = {}

    def fake_original(self, *args):
        def runtime_aware(candidate, example_inputs, *, runtime_context=None):
            seen.update(runtime_context)
            return candidate

        runtime_aware._requires_runtime_context = True
        trainer_patch.patched_apply_graph_passes(gm, (), [runtime_aware])
        return "result"

    monkeypatch.setattr(
        trainer_patch,
        "original_make_fx_forward_backward_step",
        fake_original,
    )
    result = trainer_patch.patched_make_fx_forward_backward_step(
        instance,
        model,
        inputs,
        labels,
        tokens,
        [],
        extra_kwargs,
    )

    assert result == "result"
    assert seen == {
        "traced_result": traced_result,
        "module": model,
        "args": (inputs, labels, tokens, extra_kwargs),
        "train_context": train_context,
    }
