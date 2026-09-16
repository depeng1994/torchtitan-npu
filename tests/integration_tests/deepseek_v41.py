# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests import OverrideDefinitions

# DeepSeek-V4.1 golden reference recipe: the same operator overrides the
# real-width baseline was verified with (forward verified boundary-by-boundary
# against the ds-code inference implementation).  The case runs the
# reduced-width full-structure debug model (real 40-layer compression/source
# layout, debug widths) and pins its deterministic 30-step loss trajectory
# in tests/assets/losses/dsv41_golden_2p_ep2_fsdp2.txt (2 cards, so the CI
# smoke pool can schedule it).  The 8-card shape stays available as a manual
# A/B regression via run_train.sh; its anchor is intentionally not committed.
GOLDEN_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rope.workaround",
    "torchtitan_npu.override.common.optimizer.virtual",
)

# Environment of the frozen baseline.  MODULE/CONFIG route the trainer to
# the V4.1 multimodal debug model regardless of the runner's CLI defaults
# (per-case env vars take precedence); the golden switches select the golden
# reference operator path (USE_GOLDEN with the two legacy names as
# fallbacks); the dataloader tokenizer is the committed tests/assets
# deepseek_v3 mini tokenizer (the same one the DeepSeek-V4 golden cases
# use).  The fixture image is wired by the config registry, not by env.
GOLDEN_ENV = {
    "MODULE": "torchtitan_npu.models.deepseek_v41",
    "CONFIG": "deepseek_v41_debugmodel",
    "TORCHTITAN_NPU_VISION_GOLDEN": "1",
    "TORCHTITAN_NPU_GOLDEN_TRAINING": "1",
    "DSV41_TOKENIZER_PATH": "tests/assets/deepseek_v3",
}


def _build_golden_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            override_args=[
                GOLDEN_OVERRIDES
                + (
                    "--training.steps=30",
                    "--no-engram-enabled",
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
                    "--optimizer.param-groups.1.optimizer-name=AdamW",
                    "--optimizer.param-groups.1.optimizer-kwargs.lr=1e-5",
                    "--optimizer.param-groups.1.optimizer-kwargs.betas",
                    "0.9",
                    "0.95",
                    "--optimizer.param-groups.1.optimizer-kwargs.eps=1e-6",
                    "--optimizer.param-groups.1.optimizer-kwargs.weight-decay=0.1",
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
            test_descr="DeepSeek-V4.1 golden debugmodel 2p fsdp2 ep2 exact loss",
            test_name="dsv41_golden_2p_ep2_fsdp2",
            ngpu=2,
            env_vars=GOLDEN_ENV,
            use_golden=True,
            check_loss=True,
            timeout=7200,
        ),
    ]


def _build_engram_case(*, ep_degree=2, fsdp_degree=2, ascendc=False):
    overrides = ("--override.imports", "torchtitan_npu.override.common.rope.workaround")
    if ascendc:
        overrides += ('torchtitan_npu.override.deepseek_v41.engram.host_offload={"num_max_tokens_per_rank":4096}',)
    common = overrides + (
        "--training.steps=4",
        "--training.seq-len=512",
        "--training.local-batch-size=1",
        f"--training.global-batch-size={fsdp_degree}",
        "--parallelism.spmd-backend=partial_dtensor",
        f"--parallelism.data-parallel-shard-degree={fsdp_degree}",
        f"--parallelism.expert-parallel-degree={ep_degree}",
        "--parallelism.context-parallel-degree=1",
        "--parallelism.tensor-parallel-degree=1",
        "--parallelism.pipeline-parallel-degree=1",
        "--parallelism.context-parallel-load-balancer=None",
        "--hf-assets-path=tests/assets/deepseek_v3",
        "--debug.no-moe-force-load-balance",
        "--checkpoint.enable",
        "--checkpoint.interval=2",
        "--checkpoint.no-last-save-model-only",
    )
    return OverrideDefinitions(
        override_args=[common, common + ("--checkpoint.load-step=2",)],
        test_name=f"dsv41_engram_{'ascendc' if ascendc else 'torch'}_ep{ep_degree}_fsdp{fsdp_degree}_resume",
        test_descr="V4.1 multimodal Engram FullAC training and exact DCP continuation",
        ngpu=fsdp_degree,
        use_golden=False,
        check_loss=False,
        check_resume=True,
        expected_steps=((1, 2, 3, 4), (3, 4)),
        requires_engram_ops=ascendc,
        env_vars={**GOLDEN_ENV, "CONFIG": "deepseek_v41_debugmodel"},
        timeout=600,
    )


def build_deepseek_v41_test_list():
    # Four ranks add sparse-table replicas across E-DP, beyond the two-rank EP path.
    return _build_golden_test_list() + [_build_engram_case(), _build_engram_case(fsdp_degree=4)]


def build_engram_ascendc_test_list():
    return [_build_engram_case(ascendc=True)]
