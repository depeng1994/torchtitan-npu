# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests import OverrideDefinitions

# DeepSeek-V4.1 reference recipe: the eager/reference operator overrides the
# model runs with.  The case exercises the reduced-width full-structure debug
# model (real 40-layer compression/source layout, debug widths) on 2 cards, so
# the CI smoke pool can schedule it.  V4.1 keeps no dedicated loss anchor --
# like the upstream torchtitan model, it is covered by the unit suite plus this
# runnable case, and the 8-card shape stays available via run_train.sh.
#
# The case is a smoke run, not a numeric anchor: the runner only enables
# deterministic mode and the fixed seed when ``check_loss`` is set, so the final
# loss is not reproducible run to run (end-of-run values have landed at 8.59 /
# 8.71 / 8.86 across tips).  Pass ``--debug.deterministic --debug.seed=42``
# explicitly when a stable number is needed for a manual comparison.
GOLDEN_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rope.workaround",
    "torchtitan_npu.override.common.optimizer.virtual",
)

# Environment of the reference path.  MODULE/CONFIG route the trainer to the
# V4.1 multimodal debug model regardless of the runner's CLI defaults
# (per-case env vars take precedence); the recipe itself selects the
# eager/reference operators (the AscendC fused attention does not accept
# V4.1's ratio-1 shared KV yet); the dataloader tokenizer is the committed
# tests/assets deepseek_v3 mini tokenizer.  The fixture image is wired by the
# config registry, not by env.
GOLDEN_ENV = {
    "MODULE": "torchtitan_npu.models.deepseek_v4_1",
    "CONFIG": "deepseek_v4_1_debugmodel",
    "DSV41_TOKENIZER_PATH": "tests/assets/deepseek_v3",
}


def build_deepseek_v4_1_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            override_args=[
                GOLDEN_OVERRIDES
                + (
                    "--training.steps=30",
                    "--training.local-batch-size=1",
                    "--training.global-batch-size=2",
                    "--training.seq-len=512",
                    "--parallelism.spmd-backend=partial_dtensor",
                    "--parallelism.data-parallel-shard-degree=2",
                    "--parallelism.data-parallel-replicate-degree=1",
                    "--parallelism.expert-parallel-degree=2",
                    "--parallelism.tensor-parallel-degree=1",
                    "--parallelism.context-parallel-degree=1",
                    "--parallelism.pipeline-parallel-degree=1",
                    "--parallelism.context-parallel-load-balancer=None",
                    "--debug.no-moe-force-load-balance",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--optimizer.implementation=fused",
                    "--optimizer.param-groups.0.optimizer-name=AdamW",
                    "--optimizer.param-groups.0.optimizer-kwargs.lr=1e-5",
                    "--optimizer.param-groups.0.optimizer-kwargs.betas",
                    "0.9",
                    "0.95",
                    "--optimizer.param-groups.0.optimizer-kwargs.eps=1e-6",
                    "--optimizer.param-groups.0.optimizer-kwargs.weight-decay=0.1",
                    # The LR recipe (warmup 25 / total 40) is part of the frozen
                    # anchor: the trajectory was produced with the 40-step
                    # baseline schedule. Keep it unchanged when shortening the smoke
                    # run so its 30 steps remain an exact prefix of that baseline.
                    "--lr-scheduler.warmup-steps=25",
                    "--lr-scheduler.total-steps=40",
                    "--lr-scheduler.decay-type=cosine",
                    "--lr-scheduler.decay-ratio=1.0",
                    "--lr-scheduler.min-lr-factor=0.01",
                    "--checkpoint.no-enable",
                    "--comm.init-timeout-seconds=7200",
                    "--comm.train-timeout-seconds=600",
                    "--metrics.disable-color-printing",
                )
            ],
            test_descr="DeepSeek-V4.1 debugmodel 2p fsdp2 ep2 reference run",
            test_name="dsv41_debugmodel_2p_ep2_fsdp2",
            ngpu=2,
            env_vars=GOLDEN_ENV,
            use_golden=True,
            check_loss=False,
            timeout=7200,
        ),
    ]
