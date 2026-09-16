# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for GraphTrainerEx and its SDC graph integration."""

import sys
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchtitan.config import ConfigManager
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_pass
from torchtitan.experiments.graph_trainer.inductor_passes import regional_inductor_pass
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY
from torchtitan.trainer import Trainer

import torchtitan_npu.compile.sdc_checksum as checksum_module
from torchtitan_npu.compile.sdc_checksum import sdc_checksum_graph_pass
from torchtitan_npu.config.configs import OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.components.sdc import sdc as sdc_module
from torchtitan_npu.extensions.graph_trainer import GraphTrainer, GraphTrainerEx
from torchtitan_npu.extensions.profiler import CANNProfiler
from torchtitan_npu.extensions.trainer import TrainerEx


def test_config_build(monkeypatch, tmp_path):
    received = []

    def record_init(self, config):
        received.append(config)

    monkeypatch.setattr(GraphTrainerEx, "__init__", record_init)
    source = GraphTrainer.Config(hf_assets_path=str(tmp_path), dump_folder="graph-output")
    source.training.steps = 17
    source.checkpoint.interval = 23
    source.profiler.profile_freq = 31

    def test_config():
        return source

    registry = ModuleType("_graph_config_registry")
    registry.test_config = test_config
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    config = ConfigManager().parse_args(["--module", registry.__name__, "--config", "test_config"])

    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.checkpoint, CheckpointManager.Config)
    assert isinstance(config.profiler, CANNProfiler.Config)
    assert config.dump_folder == "graph-output"
    assert config.training.steps == 17
    assert config.checkpoint.interval == 23
    assert config.profiler.profile_freq == 31
    assert config.compile == source.compile
    instance = config.build()

    assert isinstance(instance, GraphTrainerEx)
    assert isinstance(instance, GraphTrainer)
    assert isinstance(instance, TrainerEx)
    assert len(received) == 1
    assert isinstance(received[0], GraphTrainerEx.Config)
    assert isinstance(received[0].compile, GraphTrainerCompileConfig)
    assert received[0].compile.mode == "aot_fx_trace"
    assert received[0].training.extension.allow_hf32 is True


@pytest.mark.parametrize("fail_step", [False, True], ids=["success", "failure"])
def test_step_finalize(monkeypatch, fail_step):
    trainer = object.__new__(GraphTrainerEx)
    trainer.config = GraphTrainerEx.Config()
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    with torch.random.fork_rng():
        trainer.model_parts = [torch.nn.Linear(2, 2)]
    events = []
    result = torch.tensor(1.0)
    failure = RuntimeError("step failed")

    def finalize():
        events.append("finalize")

    def execute_step(*args, **kwargs):
        events.append("step")
        if fail_step:
            raise failure
        return result

    def postprocess(input_dict, labels):
        return input_dict, labels, {}

    setattr(trainer, "_sdc", SimpleNamespace(finalize_sdc_step=finalize))  # noqa: B010
    monkeypatch.setattr(Trainer, "forward_backward_step", execute_step)
    monkeypatch.setattr(GraphTrainer, "_make_fx_forward_backward_step", execute_step)
    monkeypatch.setattr(trainer, "post_dataloading_process", postprocess)
    kwargs = {
        "input_dict": {"tokens": torch.ones(1, 2)},
        "labels": torch.ones(1, 2),
        "global_valid_tokens": torch.tensor(2),
    }

    if fail_step:
        with pytest.raises(RuntimeError, match="step failed") as caught:
            trainer.forward_backward_step(**kwargs)
        assert caught.value is failure
        assert events == ["step"]
    else:
        assert trainer.forward_backward_step(**kwargs) is result
        assert events == ["step", "finalize"]


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_aot_backend(monkeypatch, backend):
    config = GraphTrainer.Config(compile=GraphTrainerCompileConfig(enable=True, components=["model"], backend=backend))
    monkeypatch.delenv("NPU_ASD_CONFIG", raising=False)
    monkeypatch.delenv("NPU_ASD_ENABLE", raising=False)
    checker = Mock()
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(asd=SimpleNamespace(asd=SimpleNamespace(matmul_check=checker)))
    )
    sdc_module.SDC(
        sdc_module.SDC.Config(gradient_enabled=True),
        trainer_config=config,
        model_parts=[],
        gradient_accumulation_steps=1,
    )
    checker.set_matmul_hook_enable.assert_called_once_with(1)


@pytest.mark.parametrize("trainer_kind", ["standard", "jit"])
def test_non_aot_backend(trainer_kind):
    config = TrainerEx.Config() if trainer_kind == "standard" else GraphTrainer.Config()
    config.compile.enable = True
    config.compile.components = ["model"]
    config.compile.backend = "aot_eager"
    if trainer_kind == "jit":
        config.compile.mode = "jit"

    with pytest.raises(ValueError, match="inductor"):
        sdc_module.SDC(
            sdc_module.SDC.Config(gradient_enabled=True),
            trainer_config=config,
            model_parts=[],
            gradient_accumulation_steps=1,
        )


class _NPUTensor(torch.Tensor):
    """CPU-backed tensor subclass carrying NPU metadata for FX pass tests."""

    @property
    def device(self):
        return SimpleNamespace(type="npu")


def _metadata_tensor(*, dtype: torch.dtype, device_type: str) -> torch.Tensor:
    tensor = torch.empty(2, 2, dtype=dtype)
    return tensor.as_subclass(_NPUTensor) if device_type == "npu" else tensor


def _make_binary_graph(
    target,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_type: str = "npu",
    add_metadata: bool = True,
) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    result = graph.call_function(target, args=(left, right))
    graph.output(result)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    if add_metadata:
        value = _metadata_tensor(dtype=dtype, device_type=device_type)
        for node in (left, right, result):
            node.meta["val"] = value
    return graph_module, result


def _checksum_nodes(graph_module: torch.fx.GraphModule) -> list[torch.fx.Node]:
    checksum_op = torch.ops.torchtitan_npu.sdc_matmul_checksum.default
    return [node for node in graph_module.graph.nodes if node.target == checksum_op]


def test_checksum_insert() -> None:
    graph_module, matmul = _make_binary_graph(torch.ops.aten.mm.default)

    result = sdc_checksum_graph_pass(graph_module, ())
    sdc_checksum_graph_pass(graph_module, ())

    checks = _checksum_nodes(graph_module)
    assert result is graph_module
    assert len(checks) == 1
    assert checks[0].args == (*matmul.args[:2], matmul)
    assert checks[0].meta["val"] is None
    graph_module.graph.lint()


@pytest.mark.parametrize(
    ("target", "dtype", "device_type", "add_metadata"),
    [
        pytest.param(torch.ops.aten.mm.default, torch.bfloat16, "cpu", True, id="cpu"),
        pytest.param(torch.ops.aten.mm.default, torch.float32, "npu", True, id="fp32"),
        pytest.param(torch.ops.aten.add.Tensor, torch.bfloat16, "npu", True, id="unsupported-op"),
        pytest.param(torch.ops.aten.mm.default, torch.bfloat16, "npu", False, id="missing-meta"),
    ],
)
def test_checksum_skip(
    target,
    dtype: torch.dtype,
    device_type: str,
    add_metadata: bool,
) -> None:
    graph_module, _ = _make_binary_graph(
        target,
        dtype=dtype,
        device_type=device_type,
        add_metadata=add_metadata,
    )

    sdc_checksum_graph_pass(graph_module, ())

    assert _checksum_nodes(graph_module) == []


def _checksum_config() -> GraphTrainerEx.Config:
    return GraphTrainerEx.Config(
        compile=GraphTrainerCompileConfig(
            enable=True,
            components=["model"],
            enable_passes=True,
            disable_passes=["cudagraph_pass"],
        ),
        sdc=sdc_module.SDC.Config(gradient_enabled=True, with_checksum=True),
    )


def _pass_name(pass_fn) -> str:
    return pass_fn.func.__name__ if isinstance(pass_fn, partial) else pass_fn.__name__


def test_checksum_pass_order(monkeypatch) -> None:
    config = _checksum_config()

    def semantic_pass(graph_module, example_inputs):
        return graph_module

    default_passes = [
        semantic_pass,
        partial(regional_inductor_pass, boxed_codegen=False),
        partial(cudagraph_pass, static_input_indices=[], tensor_input_indices=[]),
    ]
    monkeypatch.setattr(
        sdc_module,
        "construct_default_graph_passes",
        lambda *args, **kwargs: list(default_passes),
    )
    monkeypatch.setattr(TrainerEx, "__init__", lambda self, received_config: None)
    GraphTrainerEx(config)

    assert config.compile.pass_pipeline != "default"
    pipeline = PASS_PIPELINE_REGISTRY[config.compile.pass_pipeline]
    passes = pipeline(object(), config, parallel_dims=None)

    assert [_pass_name(pass_fn) for pass_fn in passes] == [
        "semantic_pass",
        "sdc_checksum_graph_pass",
        "regional_inductor_pass",
        "cudagraph_pass",
    ]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        pytest.param("enable_passes", False, "enable_passes", id="passes-disabled"),
        pytest.param("precompile_artifact_dir", "/tmp/artifact", "precompile", id="precompile"),
        pytest.param("pass_pipeline", "custom", "pass_pipeline", id="custom-pipeline"),
        pytest.param(
            "disable_passes",
            ["cudagraph_pass", "sdc_checksum_graph_pass"],
            "sdc_checksum_graph_pass",
            id="checksum-pass-disabled",
        ),
    ],
)
def test_checksum_invalid_config(
    monkeypatch,
    field: str,
    value,
    match: str,
) -> None:
    config = _checksum_config()
    setattr(config.compile, field, value)
    initialized = []
    monkeypatch.setattr(TrainerEx, "__init__", lambda self, received_config: initialized.append(received_config))

    with pytest.raises(ValueError, match=match):
        GraphTrainerEx(config)

    assert initialized == []


def test_checksum_no_post_grad(monkeypatch) -> None:
    config = _checksum_config()
    checker = Mock()
    installs = []

    def record_install() -> None:
        installs.append("post-grad")

    monkeypatch.delenv("NPU_ASD_CONFIG", raising=False)
    monkeypatch.delenv("NPU_ASD_ENABLE", raising=False)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(asd=SimpleNamespace(asd=SimpleNamespace(matmul_check=checker)))
    )
    monkeypatch.setattr(checksum_module, "install_checksum_pass", record_install)
    monkeypatch.setattr(TrainerEx, "__init__", lambda self, received_config: None)
    GraphTrainerEx(config)

    sdc = sdc_module.SDC(config.sdc, trainer_config=config, model_parts=[], gradient_accumulation_steps=1)
    sdc.finalize_sdc_step()

    assert installs == []
    checker.set_with_checksum.assert_called_once_with(True)
    checker.init_stream.assert_called_once_with()
