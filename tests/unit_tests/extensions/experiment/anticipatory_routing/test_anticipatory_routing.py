# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recovery contracts using upstream train_step and actual NPU routers, without distributed setup."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan_npu.extensions.trainer import Trainer, TrainerEx
from torchtitan_npu.extensions.experiment.anticipatory_routing.cache import RoutingMode
from torchtitan_npu.extensions.experiment.anticipatory_routing.config import AnticipatoryRoutingConfig

from torchtitan_npu.extensions.experiment.anticipatory_routing.router import build_routing_cache
from torchtitan_npu.extensions.experiment.anticipatory_routing.schedule import AnticipatorySchedule, Phase
from torchtitan_npu.patches.torchtitan.models.common.moe import HashRouter
from torchtitan_npu.extensions.experiment.anticipatory_routing.router import AnticipatoryHashRouter


class Scheduler(torch.optim.lr_scheduler.StepLR):
    def get_metrics(self):
        return {"lr": self.get_last_lr()[0]}


class Loader:
    def __init__(self, size=40):
        self.position = 0
        self.reads = []
        self.size = size

    def __iter__(self):
        for i in range(self.position, self.size):
            self.position = i + 1
            self.reads.append(i)
            yield [{"input": torch.tensor([[[i + 1.0, 0.0]]])}, torch.tensor([[i]])]

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        router = AnticipatoryHashRouter.__new__(AnticipatoryHashRouter)
        nn.Module.__init__(router)
        router.gate = nn.Identity()
        router.hash = False
        router.sorted_topk = False
        router.score_func = "sigmoid"
        router.num_experts, router.top_k = 2, 1
        router.num_expert_groups = None
        router.route_norm, router.route_scale = False, 1.0
        router._debug_force_load_balance = False
        self.router = router
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("counter", torch.tensor(0.0))


    def forward(self, x):
        self.counter.add_(1)
        self.router(x)
        return x * self.weight


def make_trainer(monkeypatch, *, delay=2, active=3, size=40, steps=10, device="cpu"):
    trainer = object.__new__(TrainerEx)
    trainer.device = torch.device(device)
    trainer.step, trainer.ntokens_seen = 0, 0
    trainer.model_parts = [Model().to(trainer.device)]
    trainer.dataloader = Loader(size)
    config = AnticipatoryRoutingConfig(
        enable=True, delay_steps=delay, active_steps=active
    )
    trainer.config = SimpleNamespace(
        anticipatory=config, training=SimpleNamespace(steps=steps, disable_cuda_graphs=True, max_norm=100.0), checkpoint=SimpleNamespace(interval=2, keep_latest_k=0)
    )
    trainer.gradient_accumulation_steps, trainer.num_pipeline_parallel_microbatches = 2, 1
    trainer.parallel_dims = SimpleNamespace(
        pp_enabled=False, dp_enabled=False, ep_enabled=False, dp_cp_enabled=False, get_optional_mesh=lambda name: None
    )
    trainer.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0, data_loading_times=[], should_log=lambda step: False
    )
    trainer.train_context = nullcontext
    trainer._sdc = SimpleNamespace(finalize_sdc_step=lambda: None)
    # Avoid torch_npu's automatic optimizer backend probe in CPU-only CI.
    trainer.optimizers = torch.optim.SGD(
        trainer.model_parts[0].parameters(), lr=0.001, momentum=0.9, foreach=False, fused=False
    )
    trainer.lr_schedulers = Scheduler(trainer.optimizers, step_size=1, gamma=0.9)
    saves = []

    def save(curr_step, last_step=False):
        saves.append((curr_step, last_step))
        return True

    trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None, save=save, saves=saves)

    def process(inputs, labels):
        trainer.ntokens_seen += labels.numel()
        return inputs["input"], labels, {}

    trainer.post_dataloading_process = process
    trainer.anticipatory_schedule = AnticipatorySchedule(trainer)
    cache = trainer.anticipatory_schedule.cache
    trained = []

    def forward_backward(self, *, input_dict, labels, global_valid_tokens):
        index = int(labels.item())
        trained.append((self.step, index, cache.mode))
        inputs, _, _ = self.post_dataloading_process(input_dict, labels)
        loss = self.model_parts[0](inputs).square().sum() / global_valid_tokens
        loss.backward()
        return loss

    monkeypatch.setattr(Trainer, "forward_backward_step", forward_backward)
    return trainer, trained


def run_step(trainer):
    trainer.step += 1
    trainer.train_step(trainer.batch_generator(trainer.dataloader))


def snapshot(trainer):
    return deepcopy({
        "model": trainer.model_parts[0].state_dict(),
        "optimizer": trainer.optimizers.state_dict(),
        "scheduler": trainer.lr_schedulers.state_dict(),
        "data": trainer.dataloader.state_dict(),
        "step": trainer.step,
        "tokens": trainer.ntokens_seen,
    })


def install_checkpoint(trainer, saved):
    """In-memory checkpoint boundary; exercise the real engine rollback path."""
    def load(*, step):
        assert step == saved["step"]
        trainer.model_parts[0].load_state_dict(saved["model"])
        trainer.optimizers.load_state_dict(saved["optimizer"])
        trainer.lr_schedulers.load_state_dict(saved["scheduler"])
        trainer.dataloader.load_state_dict(saved["data"])
        trainer.step = saved["step"]
        trainer.ntokens_seen = saved["tokens"]
        return True

    trainer.checkpointer.load = load
    trainer.checkpointer.maybe_wait_for_saving = lambda: None


def test_rollback_restores_training_state_and_restarts_iterator(monkeypatch):
    trainer, trained = make_trainer(monkeypatch)
    stream = trainer.batch_generator(trainer.dataloader)
    for _ in range(2):
        run_step(trainer)
    saved = snapshot(trainer)
    assert saved["optimizer"]["state"]  # Momentum must actually have been created.
    assert saved["scheduler"]["last_epoch"] == 2
    install_checkpoint(trainer, saved)

    # Establish the exact next update from the saved state.
    run_step(trainer)
    expected_next = snapshot(trainer)
    run_step(trainer)
    old_iterator = stream.iterator
    assert trainer.dataloader.position == 8
    assert trainer.lr_schedulers.last_epoch == 4
    assert not torch.equal(trainer.model_parts[0].weight, saved["model"]["weight"])

    trainer.anticipatory_schedule.engine.rollback_to(2)
    torch.testing.assert_close(snapshot(trainer), saved, rtol=0, atol=0)
    assert all(p.grad is None for p in trainer.model_parts[0].parameters())
    assert stream.iterator is None
    assert trainer.batch_generator(trainer.dataloader) is stream

    # The abandoned iterator has its own cursor: without reset it would yield 8.
    run_step(trainer)
    assert stream.iterator is not old_iterator
    assert [index for _, index, _ in trained[-2:]] == [4, 5]
    torch.testing.assert_close(snapshot(trainer), expected_next, rtol=0, atol=0)


def test_multiple_microbatches_replay_their_own_layer_routes(monkeypatch):
    trainer, _ = make_trainer(monkeypatch)
    model = trainer.model_parts[0]
    model.second_router = deepcopy(model.router)
    cache = build_routing_cache([model], index_store_dtype="auto", device=trainer.device)
    trainer.anticipatory_schedule.cache = cache
    routers = [model.router, model.second_router]
    slots, recorded = [], []
    # Each (microbatch, layer) gets a distinct ordered token-to-expert pattern.
    patterns = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    for microbatch in range(2):
        slot, expected = {}, []
        with cache.capturing(slot):
            for layer, router in enumerate(routers):
                ids = torch.tensor(patterns[2 * microbatch + layer])
                logits = torch.nn.functional.one_hot(ids, num_classes=2).float() * 5
                _, actual, _ = router(logits)
                torch.testing.assert_close(actual.reshape(-1), ids, rtol=0, atol=0)
                expected.append(actual.detach().clone())
        assert set(slot) == {router._anticipatory_key for router in routers}
        slots.append(slot)
        recorded.append(expected)

    consumed = []
    def forward_backward(self, *, input_dict, labels, global_valid_tokens):
        microbatch = int(labels.item())
        for layer, router in enumerate(routers):
            expected = recorded[microbatch][layer]
            # Live top-k deliberately disagrees at EVERY token.
            live_ids = 1 - expected.reshape(-1)
            logits = (torch.nn.functional.one_hot(live_ids, 2).float() * 5).requires_grad_()
            scores, actual, _ = router(logits)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Consume the returned IDs, rather than comparing cache to itself.
            experts = torch.tensor([2.0, 7.0])
            torch.testing.assert_close(experts[actual], experts[expected], rtol=0, atol=0)
            scores.sum().backward()
            selected_grad = logits.grad.gather(-1, expected)
            unselected_grad = logits.grad.gather(-1, 1 - expected)
            assert torch.all(selected_grad != 0)
            assert torch.all(unselected_grad == 0)
            consumed.append((microbatch, layer))
        return torch.tensor(1.0)

    monkeypatch.setattr(Trainer, "forward_backward_step", forward_backward)
    engine = trainer.anticipatory_schedule.engine
    engine.begin_step()
    engine.in_step = True
    try:
        with cache.replaying(slots):
            for microbatch in range(2):
                # The real TrainerEx wrapper must select the right slot.
                trainer.forward_backward_step(
                    input_dict={}, labels=torch.tensor(microbatch), global_valid_tokens=torch.tensor(1)
                )
    finally:
        engine.in_step = False
    assert consumed == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert engine._microbatch_index == 2
    assert cache.mode is RoutingMode.OFF
    assert cache.slot is None and cache.step_slots is None


@pytest.mark.parametrize("device", ["cpu", "npu:0"])
def test_spike_recovery_warmup_active_drain_and_normal(monkeypatch, device):
    if device != "cpu" and not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    trainer, trained = make_trainer(monkeypatch, active=2, steps=8, device=device)
    engine, schedule = trainer.anticipatory_schedule.engine, trainer.anticipatory_schedule
    saves = trainer.checkpointer.saves
    assert trainer.checkpointer.save(curr_step=0)
    observations = []

    def observe_spike(step, loss):
        observations.append(step)
        return 4 if step == 4 and observations.count(4) == 1 else None

    # Inject a detector verdict to exercise recovery without a manual-trigger config.
    monkeypatch.setattr(schedule.detector, "observe", observe_spike)
    router = trainer.model_parts[0].router
    recorded_ids, replayed_ids = [], []

    def change_live_choice(module, args):
        if schedule.cache.mode is RoutingMode.REPLAY:
            # Recorded inputs favor expert 0; live replay scores favor expert 1.
            return (args[0].flip(-1),)

    def observe_routes(module, args, output):
        ids = output[1].detach().clone()
        if schedule.cache.mode is RoutingMode.CAPTURE:
            recorded_ids.append(ids)
        elif schedule.cache.mode is RoutingMode.REPLAY:
            expected = recorded_ids[len(replayed_ids)]
            torch.testing.assert_close(ids, expected, rtol=0, atol=0)
            assert torch.all(ids == 0)
            replayed_ids.append(ids)

    gate_hook = router.gate.register_forward_pre_hook(change_live_choice)
    router_hook = router.register_forward_hook(observe_routes)
    monkeypatch.setattr(engine, "find_rollback_target", lambda onset: 2 if onset == 4 else None)
    updates = []
    original_step = trainer.optimizers.step

    def count_update(*args, **kwargs):
        updates.append(trainer.step)
        return original_step(*args, **kwargs)

    # Install after scheduler construction; preserve its optimizer-step marker.
    count_update._wrapped_by_lr_sched = True
    monkeypatch.setattr(trainer.optimizers, "step", count_update)
    for _ in range(2):
        run_step(trainer)
    saved = snapshot(trainer)
    install_checkpoint(trainer, saved)
    run_step(trainer)
    run_step(trainer)

    # WARMUP is inline: no optimizer/scheduler/token/buffer advancement.
    assert updates == [1, 2, 3, 4]
    after_warmup = snapshot(trainer)
    assert after_warmup.pop("data") == {"position": 8}
    restored = deepcopy(saved)
    restored.pop("data")
    torch.testing.assert_close(after_warmup, restored, rtol=0, atol=0)
    assert all(module.training for part in trainer.model_parts for module in part.modules())
    assert schedule.phase is Phase.ACTIVE
    assert len(schedule._queue) == 2
    assert [[int(labels.item()) for _, labels in entry.microbatches]
            for entry in schedule._queue] == [[4, 5], [6, 7]]
    assert all(entry.slots and len(entry.slots) == 2 for entry in schedule._queue)
    for entry in schedule._queue:
        for inputs, labels in entry.microbatches:
            assert inputs["input"].device.type == labels.device.type == "cpu"
        for slot in entry.slots:
            assert all(ids.device == trainer.device for ids in slot.values())
    assert engine.suppress_checkpoint_saves
    assert not trainer.checkpointer.save(curr_step=trainer.step)

    phases, lengths, positions, suppressed = [], [], [], []
    while trainer.step < 8:
        run_step(trainer)
        phases.append(schedule.phase)
        lengths.append(len(schedule._queue))
        positions.append(trainer.dataloader.position)
        suppressed.append(engine.suppress_checkpoint_saves)
        assert trainer.checkpointer.save(trainer.step) is (not engine.suppress_checkpoint_saves)

    assert phases == [Phase.ACTIVE, Phase.DRAIN, Phase.DRAIN, Phase.NORMAL, Phase.NORMAL, Phase.NORMAL]
    assert lengths == [2, 2, 1, 0, 0, 0]
    assert positions == [10, 12, 12, 12, 14, 16]
    assert suppressed == [True, True, True, False, False, False]
    assert [index for _, index, _ in trained] == list(range(8)) + list(range(4, 16))
    assert [mode for _, _, mode in trained[8:]] == [RoutingMode.REPLAY] * 8 + [RoutingMode.OFF] * 4
    assert trainer.dataloader.reads == list(range(8)) + list(range(4, 16))
    assert updates == [1, 2, 3, 4, 3, 4, 5, 6, 7, 8]
    assert trainer.lr_schedulers.last_epoch == 8
    assert trainer.ntokens_seen == 16
    assert trainer.model_parts[0].counter.item() == 16
    assert schedule._num_rollbacks == 1
    assert observations == [1, 2, 3, 4, 6, 7, 8]
    assert not schedule._queue
    assert schedule.cache.mode is RoutingMode.OFF
    assert schedule.cache.slot is None and schedule.cache.step_slots is None
    assert not engine.in_step and not engine.failed
    assert saves == [(0, False), (6, False), (7, False), (8, False)]
    engine.failed = True
    assert not trainer.checkpointer.save(8, last_step=True)
    assert len(recorded_ids) == len(replayed_ids) == 8
    gate_hook.remove()
    router_hook.remove()
    assert "train" not in TrainerEx.__dict__
