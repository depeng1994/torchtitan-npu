# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torchtitan.trainer
from torchtitan.distributed import full_dtensor
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.trainer import Trainer

from torchtitan_npu.patches.torchtitan.components.checkpoint import EMACheckpointManager
from torchtitan_npu.patches.torchtitan.components.ema import EMAOptimizersContainer

logger = logging.getLogger(__name__)

original_post_dataloading_process = Trainer.post_dataloading_process
original_pp_forward_backward_step = Trainer.pp_forward_backward_step


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels):
    """Dispatch to the model's own mask/metadata construction when present.

    Models that define ``build_attention_masks`` (e.g. DeepSeek-V4) own the
    whole per-batch metadata handling — including their own context-parallel
    sharding and plan derivation — as the single overridable seam replacing
    the removed ``mask_handler`` pattern.  Other models keep the upstream
    flow (``get_attention_masks`` + the generic ``prepare_context_parallel_input``)
    exactly as-is.  The steps after the build (the token accounting and the
    spmd-input annotation) are replicated here for the custom path.
    """
    model = self.model_parts[0]
    build = getattr(model, "build_attention_masks", None)
    if build is not None:
        inputs = input_dict["input"]
        extra_kwargs: dict[str, Any] = {k: v for k, v in input_dict.items() if k != "input"}
        cp_mesh = self.parallel_dims.get_mesh("cp") if self.parallel_dims.cp_enabled else None
        inputs, labels, extra_kwargs = build(
            inputs,
            labels,
            extra_kwargs,
            cp_mesh=cp_mesh,
            load_balancer_type=self.config.parallelism.context_parallel_load_balancer,
        )
        # Accumulate after CP sharding so labels.numel() reflects the actual
        # unique tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += labels.numel()
        if self.config.parallelism.spmd_backend == "full_dtensor":
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )
        elif self.config.parallelism.spmd_backend == "spmd_types":
            inputs, labels, extra_kwargs = annotate_input_spmd_types(
                self.parallel_dims,
                inputs,
                labels,
                extra_kwargs,
            )
        return inputs, labels, extra_kwargs
    return original_post_dataloading_process(self, input_dict, labels)


def _run_presplit_pipeline_schedule(
    schedule,
    *,
    arg_mbs,
    kwarg_mbs,
    target_mbs,
    losses,
    loss_kwargs,
    return_outputs,
):
    """Bridge the NPU Torch pipeline schedule's private microbatch API."""
    if (
        schedule._has_backward
        and getattr(schedule, "_backward_requires_autograd", True)
        and not torch.is_grad_enabled()
    ):
        raise RuntimeError("pipeline backward requires gradients")

    stages = getattr(schedule, "_stages", None)
    if stages is None:
        stages = [schedule._stage]
    for stage in stages:
        stage.has_backward = schedule._has_backward
        stage.clear_runtime_states()
    return schedule._step_microbatches(
        arg_mbs=arg_mbs,
        kwarg_mbs=kwarg_mbs,
        target_mbs=target_mbs,
        losses=losses,
        loss_kwargs=loss_kwargs,
        return_outputs=return_outputs,
    )


@functools.wraps(original_pp_forward_backward_step)
def patched_pp_forward_backward_step(self, *args, **kwargs):
    """Adapt older NPU pipeline schedules without changing Trainer semantics."""
    schedule = self.pp_schedule
    parameters = inspect.signature(schedule.step).parameters
    if {"arg_mbs", "kwarg_mbs", "target_mbs"}.issubset(parameters):
        return original_pp_forward_backward_step(self, *args, **kwargs)
    if not hasattr(schedule, "_step_microbatches"):
        raise RuntimeError("installed pipeline schedule has no pre-split API")

    original_step = schedule.step
    had_instance_step = "step" in vars(schedule)
    schedule.step = functools.partial(_run_presplit_pipeline_schedule, schedule)
    try:
        return original_pp_forward_backward_step(self, *args, **kwargs)
    finally:
        if had_instance_step:
            schedule.step = original_step
        else:
            del schedule.step


class EMATrainer(Trainer):
    """Trainer with upstream EMA configuration and optimizer wiring."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        ema_weights: EMAOptimizersContainer.Config = field(default_factory=EMAOptimizersContainer.Config)
        checkpoint: EMACheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=EMACheckpointManager.Config
        )

    def __init__(self, config: Config) -> None:
        config.checkpoint.ema_weights = config.ema_weights
        super().__init__(config)

        self.ema_optimizer = cast("Any", self.checkpointer).ema_optimizer
        self.optimizers.register_step_pre_hook(lambda *_args, **_kwargs: self.ema_optimizer.wait_for_param_reads())
        self.optimizers.register_step_post_hook(lambda *_args, **_kwargs: self.ema_optimizer.step(self.step))


def apply() -> None:
    logger.info("[PATCH] Trainer.post_dataloading_process -> patched_post_dataloading_process")
    Trainer.post_dataloading_process = patched_post_dataloading_process
    logger.info("[PATCH] Trainer.pp_forward_backward_step -> patched_pp_forward_backward_step")
    Trainer.pp_forward_backward_step = patched_pp_forward_backward_step
    torchtitan.trainer.Trainer = EMATrainer


apply()
