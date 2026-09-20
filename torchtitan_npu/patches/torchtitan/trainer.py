# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985
# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4421

import functools
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, cast

import torchtitan.trainer
from torchtitan.distributed import full_dtensor
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.trainer import Trainer

from torchtitan_npu.patches.torchtitan.components.checkpoint import EMACheckpointManager
from torchtitan_npu.patches.torchtitan.components.ema import EMAOptimizersContainer

logger = logging.getLogger(__name__)

original_post_dataloading_process = Trainer.post_dataloading_process


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


# Backport TorchTitan #4421 by rewriting only the two v0.3.0 H2D copies.
def patch_async_h2d() -> None:
    src = textwrap.dedent(inspect.getsource(Trainer.train_step))
    assert src.count("value.to(self.device)") == src.count("labels.to(self.device)") == 1
    src = src.replace("value.to(self.device)", "value.to(self.device, non_blocking=True)")
    src = src.replace("labels.to(self.device)", "labels.to(self.device, non_blocking=True)")
    scope = dict(Trainer.train_step.__globals__)
    exec(src, scope)
    Trainer.train_step = scope["train_step"]


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
    logger.info("[PATCH] Trainer.train_step -> async H2D source rewrite (upstream #4421)")
    Trainer.post_dataloading_process = patched_post_dataloading_process
    patch_async_h2d()
    torchtitan.trainer.Trainer = EMATrainer


apply()
