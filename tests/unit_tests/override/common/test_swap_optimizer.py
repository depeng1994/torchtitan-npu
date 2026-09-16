# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""CPU contracts for the explicit optimizer-state swap override policy."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.components.checkpoint import AsyncMode
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, OverrideConfig, apply_overrides

from torchtitan_npu.extensions.components.checkpoint import CheckpointManager
from torchtitan_npu.override.common import optimizer as product_swap
from torchtitan_npu.override.common.optimizer import OptimizerStateSwapContainer

pytestmark = pytest.mark.cpu


def _patch_container_global(monkeypatch, name: str, value) -> None:
    """Patch the module globals used by the imported container class.

    Other CPU tests can temporarily replace ``torchtitan_npu`` modules. Patch
    the class under test directly so fakes remain effective even when a package
    attribute points at a different module instance.
    """
    swap_adamw = getattr(OptimizerStateSwapContainer, "_swap_adamw")
    monkeypatch.setitem(swap_adamw.__globals__, name, value)


def _patch_container_swap_api(monkeypatch, swap_api) -> None:
    _patch_container_global(monkeypatch, "swap_api", swap_api)


def _install_adamw_swap(optimizer: torch.optim.AdamW) -> None:
    getattr(OptimizerStateSwapContainer, "_swap_adamw")(optimizer)


def _patch_fake_swap_runtime(
    monkeypatch,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[SimpleNamespace]]]:
    registered: dict[str, torch.Tensor] = {}
    handles: dict[str, tuple[SimpleNamespace]] = {}

    def register_tensor(tensor, name) -> None:
        registered[name] = tensor

    def execute(name, action) -> None:
        if action == "D2H":
            handles[name] = (
                SimpleNamespace(
                    swap_event=None,
                    is_completed=False,
                    tensor_cpu=registered[name].detach().view(torch.uint8).clone(),
                ),
            )
        elif action == "H2D":
            registered[name].view(torch.uint8).copy_(handles[name][0].tensor_cpu)

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(register_tensor=register_tensor, execute=execute),
    )
    _patch_container_global(monkeypatch, "SwapEngine", SimpleNamespace(_handles=handles))
    return registered, handles


def _build_checkpoint_model(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.GELU(),
        torch.nn.Linear(3, 2),
    )
    named_parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in named_parameters],
                "param_names": [name for name, _ in named_parameters],
            }
        ],
        lr=0.03,
        foreach=False,
    )
    _install_adamw_swap(optimizer)
    container = object.__new__(OptimizerStateSwapContainer)
    container.optimizers = [optimizer]
    return model, optimizer, container


def _build_single_parameter_container(parameter: torch.nn.Parameter, lr: float):
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "param_names": ["weight"]}],
        lr=lr,
        foreach=False,
    )
    _install_adamw_swap(optimizer)
    container = object.__new__(OptimizerStateSwapContainer)
    container.optimizers = [optimizer]
    return optimizer, container


def _named_optimizer_parameters(optimizer):
    for group in optimizer.param_groups:
        yield from zip(group["params"], group["param_names"], strict=True)


def _clone_flat_tensor_state(optimizer) -> dict[str, torch.Tensor]:
    result = {}
    for parameter, fqn in _named_optimizer_parameters(optimizer):
        for state_name, value in optimizer.state[parameter].items():
            if isinstance(value, torch.Tensor):
                result[f"state.{fqn}.{state_name}"] = value.detach().clone()
    return result


def _swapped_tensor_identities(optimizer) -> dict[tuple[torch.Tensor, str], torch.Tensor]:
    result = {}
    state_names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    for parameter, state in optimizer.state.items():
        for state_name, value in state.items():
            if state_name in state_names:
                result[(parameter, state_name)] = value
    return result


def _assert_checkpoint_views(optimizer, flat_state, handles) -> None:
    parameter_fqns = dict(_named_optimizer_parameters(optimizer))
    locations = getattr(optimizer, "_torchtitan_npu_checkpoint_locations")
    for (parameter, state_name), (tensor_name, byte_offset) in locations.items():
        view = flat_state[f"state.{parameter_fqns[parameter]}.{state_name}"]
        raw = handles[tensor_name][0].tensor_cpu
        assert view.data_ptr() == raw.data_ptr() + byte_offset
        assert view.global_shape == tuple(view.shape)
        assert view.global_offsets == (tuple(0 for _ in view.shape),)
        assert view.local_offsets == (tuple(0 for _ in view.shape),)
        assert view.local_sizes == (tuple(view.shape),)


def _set_model_gradients(model, value: float) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, value)


def _assert_named_tensors(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(actual[name], expected_tensor, rtol=0, atol=0)


class _Root(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )


def test_optimizer_state_swap_override_replaces_optimizer_config() -> None:
    config = _Root.Config()

    replacements = apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.optimizer.swap_optimizer",
            ]
        ),
        config,
    )

    assert len(replacements) == 1
    assert isinstance(config.optimizer, OptimizerStateSwapContainer.Config)


def test_optimizer_state_swap_rejects_pinned_memory_checkpoint() -> None:
    config = CheckpointManager.Config(
        enable=True,
        async_mode=AsyncMode.ASYNC_WITH_PINNED_MEM.value,
    )

    with pytest.raises(ValueError, match="async_with_pinned_mem.*unsupported"):
        CheckpointManager(
            config,
            optimizers=SimpleNamespace(supports_async_with_pinned_mem=False),
        )


def test_native_optimizer_allows_pinned_memory_checkpoint(monkeypatch) -> None:
    config = CheckpointManager.Config(
        enable=True,
        async_mode=AsyncMode.ASYNC_WITH_PINNED_MEM.value,
    )
    captured = {}

    def init_base(_self, *, config, **kwargs) -> None:
        captured.update(config=config, **kwargs)

    monkeypatch.setattr(CheckpointManager.__mro__[1], "__init__", init_base)
    optimizers = object()

    CheckpointManager(config, optimizers=optimizers)

    assert captured["config"] is config
    assert captured["optimizers"] is optimizers


def test_virtual_optimizer_override_replaces_optimizer_config() -> None:
    config = _Root.Config()

    replacements = apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.optimizer.virtual",
            ]
        ),
        config,
    )

    assert len(replacements) == 1
    assert type(config.optimizer).__module__ == "torchtitan_npu.override.common.optimizer"
    assert type(config.optimizer).__qualname__ == "VirtualOptimizersContainer.Config"


def test_virtual_optimizers_container_registers_state_init_hook(monkeypatch) -> None:
    hooks = []
    optimizer = SimpleNamespace(register_step_pre_hook=hooks.append)

    def initialize_container(self, *, config, model_parts):
        self.optimizers = [optimizer]

    monkeypatch.setattr(
        product_swap.OptimizersContainer,
        "__init__",
        initialize_container,
    )

    product_swap.VirtualOptimizersContainer(config=object(), model_parts=[])

    assert hooks == [product_swap._swap_state_init_hook]


def test_optimizer_state_swap_conflicts_with_virtual_optimizer_override() -> None:
    config = _Root.Config()

    with pytest.raises(ValueError, match="both claim node 'optimizer'"):
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_npu.override.common.optimizer.virtual",
                    "torchtitan_npu.override.common.optimizer.swap_optimizer",
                ]
            ),
            config,
        )


def test_optimizer_state_swap_async_dcp_round_trip(monkeypatch, tmp_path) -> None:
    registered, handles = _patch_fake_swap_runtime(monkeypatch)
    source_model, source_optimizer, source_container = _build_checkpoint_model(seed=7)
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4).div(7)
    source_model(inputs).square().sum().backward()
    source_optimizer.step()
    source_optimizer.param_groups[0]["lr"] = 0.017

    expected_parameters = {name: parameter.detach().clone() for name, parameter in source_model.named_parameters()}
    expected_state = _clone_flat_tensor_state(source_optimizer)
    source_flat_state = source_container.state_dict()
    _assert_checkpoint_views(source_optimizer, source_flat_state, handles)
    original_cpu_buffers = {name: handle[0].tensor_cpu.clone() for name, handle in handles.items()}

    checkpoint_dir = tmp_path / "dcp"
    future = dcp.async_save(
        {"model": source_model, "optimizer": source_container},
        checkpoint_id=checkpoint_dir,
        no_dist=True,
    )
    for handle in handles.values():
        handle[0].tensor_cpu.fill_(0xFF)
    future.result()

    for name, raw in original_cpu_buffers.items():
        handles[name][0].tensor_cpu.copy_(raw)
    _set_model_gradients(source_model, 0.125)
    source_optimizer.step()
    expected_next_parameters = {name: parameter.detach().clone() for name, parameter in source_model.named_parameters()}

    del source_flat_state
    registered.clear()
    handles.clear()
    del source_container, source_optimizer, source_model

    target_model, target_optimizer, target_container = _build_checkpoint_model(seed=19)
    target_container.state_dict()
    target_state_tensors = _swapped_tensor_identities(target_optimizer)
    dcp.load(
        {"model": target_model, "optimizer": target_container},
        checkpoint_id=checkpoint_dir,
        no_dist=True,
    )

    assert target_optimizer.param_groups[0]["lr"] == 0.017
    for key, tensor in target_state_tensors.items():
        assert target_optimizer.state[key[0]][key[1]] is tensor

    target_parameters = dict(target_model.named_parameters())
    _assert_named_tensors(target_parameters, expected_parameters)
    target_state = target_container.state_dict()
    _assert_named_tensors(target_state, expected_state)
    _set_model_gradients(target_model, 0.125)
    target_optimizer.step()
    _assert_named_tensors(target_parameters, expected_next_parameters)


def test_optimizer_state_swap_direct_state_dict_round_trip(monkeypatch) -> None:
    _patch_fake_swap_runtime(monkeypatch)
    source_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    source_optimizer, source_container = _build_single_parameter_container(source_parameter, lr=0.03)
    source_parameter.grad = torch.tensor([0.25, -0.5])
    source_optimizer.step()
    source_optimizer.param_groups[0]["lr"] = 0.017
    expected_state = {name: value.detach().clone() for name, value in source_optimizer.state[source_parameter].items()}
    source_state = deepcopy(source_container.state_dict())

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target_optimizer, target_container = _build_single_parameter_container(target_parameter, lr=0.5)
    target_container.state_dict()
    target_identities = {
        name: value
        for name, value in target_optimizer.state[target_parameter].items()
        if name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    }

    target_container.load_state_dict(source_state)

    assert target_optimizer.param_groups[0]["lr"] == 0.017
    for name, tensor in target_identities.items():
        assert target_optimizer.state[target_parameter][name] is tensor
    target_state = target_container.state_dict()
    for name, expected in expected_state.items():
        torch.testing.assert_close(target_state[f"state.weight.{name}"], expected, rtol=0, atol=0)

    gradient = torch.tensor([-0.125, 0.375])
    source_parameter.grad = gradient.clone()
    target_parameter.grad = gradient.clone()
    source_optimizer.step()
    target_optimizer.step()

    torch.testing.assert_close(target_parameter, source_parameter, rtol=0, atol=0)
    for name, source_value in source_optimizer.state[source_parameter].items():
        torch.testing.assert_close(
            target_optimizer.state[target_parameter][name],
            source_value,
            rtol=0,
            atol=0,
        )


def test_optimizer_state_swap_muon_uses_registered_checkpoint_view(
    monkeypatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    class FakeMuon(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__(
                [
                    {
                        "params": [parameter],
                        "param_names": ["weight"],
                        "lr": 0.2,
                    }
                ],
                defaults={},
            )

    optimizer = FakeMuon()
    momentum = torch.tensor([3.0, 4.0])
    optimizer.state[parameter]["momentum_buffer"] = momentum
    setattr(
        optimizer,
        "_torchtitan_npu_checkpoint_locations",
        {(parameter, "momentum_buffer"): ("optimizer_state.0.weight.momentum_buffer", 0)},
    )
    checkpoint_view = torch.tensor([30.0, 40.0])
    calls = []

    def get_checkpoint_view(tensor_name, tensor, *, byte_offset):
        calls.append((tensor_name, tensor, byte_offset))
        return checkpoint_view

    _patch_container_global(monkeypatch, "DistMuon", FakeMuon)
    _patch_container_global(monkeypatch, "init_optim_state", lambda optimizer: None)
    _patch_container_global(monkeypatch, "get_checkpoint_view", get_checkpoint_view)
    container = object.__new__(OptimizerStateSwapContainer)
    container.optimizers = [optimizer]

    flat_state = container.state_dict()

    assert calls == [("optimizer_state.0.weight.momentum_buffer", momentum, 0)]
    assert flat_state["state.weight.momentum_buffer"] is checkpoint_view
    assert flat_state["param_groups.weight.lr"] == 0.2


def test_optimizer_state_swap_has_no_unowned_cleanup_facade() -> None:
    source = product_swap.__file__
    assert source is not None
    contents = Path(source).read_text()

    assert "close_swap_state" not in contents
    assert "_torchtitan_npu_close_swap_state" not in contents
    assert "_torchtitan_npu_swap_error" not in contents


def test_optimizer_state_swap_adamw_pipelines_novaswap_buckets(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(4))
    parameter_b = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter_a, parameter_b], lr=1e-3, foreach=False)
    events = []
    phases = {}

    def register_tensor(tensor, name) -> None:
        events.append(("register", name, tensor))

    def execute(name, action) -> None:
        events.append(("execute", name, action))
        if action == "D2H":
            phases[name] = "D2H"

    fake_swap_api = SimpleNamespace(
        register_tensor=register_tensor,
        execute=execute,
        get_handle_phase=lambda name: phases.get(name),
        remove_tensor=lambda name: events.append(("remove", name)),
    )
    _patch_container_swap_api(monkeypatch, fake_swap_api)
    _install_adamw_swap(optimizer)

    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    optimizer.step()

    names = [
        f"adamw.{id(optimizer)}.bucket.0",
        f"adamw.{id(optimizer)}.bucket.1",
    ]
    assert [event[1] for event in events if event[0] == "register"] == names
    assert [event[2] for event in events if event[0] == "execute"] == [
        "D2H",
        "D2H",
    ]
    for parameter in (parameter_a, parameter_b):
        state = optimizer.state[parameter]
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert state["exp_avg"].untyped_storage().data_ptr() == state["exp_avg_sq"].untyped_storage().data_ptr()
    assert "step" not in optimizer.param_groups[0]

    events.clear()
    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    optimizer.step()

    assert [event[2] for event in events if event[0] == "execute"] == [
        "H2D",
        "WAIT_DEVICE",
        "H2D",
        "D2H",
        "WAIT_DEVICE",
        "D2H",
    ]


def test_optimizer_state_swap_adamw_prefetches_before_current_bucket_compute(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(4))
    parameter_b = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter_a, parameter_b], lr=1e-3, foreach=False)
    events = []
    monkeypatch.setattr(
        product_swap,
        "swap_api",
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: events.append((action, name)),
        ),
    )
    swap = product_swap._NovaSwapAdamW(optimizer)

    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    swap.step()

    events.clear()
    swap._original_step = lambda closure=None: events.append(("AdamW", None))
    swap.step()

    assert [action for action, _ in events] == [
        "H2D",
        "WAIT_DEVICE",
        "H2D",
        "AdamW",
        "D2H",
        "WAIT_DEVICE",
        "AdamW",
        "D2H",
    ]


def test_optimizer_state_swap_adamw_keeps_stock_state_and_updates(monkeypatch) -> None:
    swapped_parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([3.0, 4.0])),
    ]
    reference_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in swapped_parameters
    ]
    swapped = torch.optim.AdamW(swapped_parameters, lr=1e-3, foreach=False)
    reference = torch.optim.AdamW(reference_parameters, lr=1e-3, foreach=False)
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: None,
        ),
    )
    _install_adamw_swap(swapped)

    for _ in range(2):
        original_param_groups = swapped.param_groups
        for swapped_parameter, reference_parameter in zip(
            swapped_parameters, reference_parameters, strict=True
        ):
            grad = torch.full_like(swapped_parameter, 0.25)
            swapped_parameter.grad = grad
            reference_parameter.grad = grad.clone()
        swapped.step()
        reference.step()
        assert swapped.param_groups is original_param_groups

    for swapped_parameter, reference_parameter in zip(
        swapped_parameters, reference_parameters, strict=True
    ):
        assert torch.equal(swapped_parameter, reference_parameter)
        swapped_state = swapped.state[swapped_parameter]
        reference_state = reference.state[reference_parameter]
        assert swapped_state.keys() == reference_state.keys()
        for key in swapped_state:
            assert torch.equal(swapped_state[key], reference_state[key])


def test_optimizer_state_swap_keeps_tensor_wait_and_local_offload_at_prepare(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(2))
    parameter_b = torch.nn.Parameter(torch.ones(3))
    layout_a = SimpleNamespace(param=parameter_a, fqn="a")
    layout_b = SimpleNamespace(param=parameter_b, fqn="b")
    events = []

    class FakeMuon:
        def __init__(self) -> None:
            self.state = {
                parameter_a: {"momentum_buffer": parameter_a.detach().clone()},
                parameter_b: {"momentum_buffer": parameter_b.detach().clone()},
            }

            self._redistribution_runtime = SimpleNamespace(
                _enqueue_storage_to_compute=lambda *args, **kwargs: None,
            )

        def _momentum(self, compute_layout, grad):
            return self.state[compute_layout.param]["momentum_buffer"]

        def _prepare_local(self, compute_layout, out):
            events.append(("prepare", compute_layout))

    def execute(name, action) -> None:
        events.append(("execute", name, action))

    phases = {}

    def get_handle_phase(name):
        return phases.get(name, "D2H")

    def execute_with_phase(name, action) -> None:
        execute(name, action)
        if action == "H2D":
            phases[name] = "H2D"
        elif action == "D2H":
            phases[name] = "D2H"

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=execute_with_phase,
            get_handle_phase=get_handle_phase,
        ),
    )

    optimizer = FakeMuon()
    OptimizerStateSwapContainer._swap_muon(optimizer, model_part=0)

    optimizer._prepare_local(layout_a, None)
    optimizer._prepare_local(layout_b, None)

    assert [event[2] for event in events if event[0] == "execute"] == [
        "H2D",
        "WAIT_DEVICE",
        "D2H",
        "H2D",
        "WAIT_DEVICE",
        "D2H",
    ]
    assert [event[0] for event in events] == [
        "execute",
        "execute",
        "prepare",
        "execute",
        "execute",
        "execute",
        "prepare",
        "execute",
    ]


def test_optimizer_state_swap_prefetches_plan_before_a2a_and_defers_offload(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(2))
    parameter_b = torch.nn.Parameter(torch.ones(3))
    parameter_c = torch.nn.Parameter(torch.ones(4))
    layout_a = SimpleNamespace(param=parameter_a, fqn="a")
    layout_b = SimpleNamespace(param=parameter_b, fqn="b")
    layout_c = SimpleNamespace(param=parameter_c, fqn="c")
    events = []
    phases = {}
    streams = []

    class FakeRuntime:
        def _enqueue_storage_to_compute(self, plan, slot, context, *, prepare):
            for layout in plan.redistributed_items:
                prepare(layout, None)
            events.append(("a2a", plan))
            return "work"

    class FakeMuon:
        def __init__(self) -> None:
            self.state = {
                parameter_a: {"momentum_buffer": parameter_a.detach().clone()},
                parameter_b: {"momentum_buffer": parameter_b.detach().clone()},
                parameter_c: {"momentum_buffer": parameter_c.detach().clone()},
            }
            self._redistribution_runtime = FakeRuntime()

        def _momentum(self, compute_layout, grad):
            return self.state[compute_layout.param]["momentum_buffer"]

        def _prepare_local(self, compute_layout, out):
            events.append(("prepare", compute_layout.fqn))

    def execute(name, action) -> None:
        events.append((action, name))
        phases[name] = action

    def stream_context(stream):
        streams.append(stream)
        return nullcontext()

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=execute,
            get_handle_phase=lambda name: phases.get(name, "D2H"),
        ),
    )
    _patch_container_global(
        monkeypatch,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(stream=stream_context)),
    )

    optimizer = FakeMuon()
    OptimizerStateSwapContainer._swap_muon(optimizer, model_part=0)
    plan = SimpleNamespace(
        redistributed_items=(layout_a, layout_b),
        unredistributed_items=(layout_c,),
    )

    transfer_stream = object()
    assert optimizer._redistribution_runtime._enqueue_storage_to_compute(
        plan,
        slot=None,
        context=SimpleNamespace(transfer_stream=transfer_stream),
        prepare=optimizer._prepare_local,
    ) == "work"
    assert streams == [transfer_stream]
    assert [event[0] for event in events] == [
        "H2D",
        "H2D",
        "H2D",
        "WAIT_DEVICE",
        "prepare",
        "WAIT_DEVICE",
        "prepare",
        "a2a",
        "D2H",
        "D2H",
    ]
    assert [event[1] for event in events[:3]] == [
        product_swap.make_swap_state_name("optimizer_state", 0, "a", "momentum_buffer"),
        product_swap.make_swap_state_name("optimizer_state", 0, "b", "momentum_buffer"),
        product_swap.make_swap_state_name("optimizer_state", 0, "c", "momentum_buffer"),
    ]
