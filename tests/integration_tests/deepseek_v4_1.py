# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests import OverrideDefinitions

# DeepSeek-V4.1 golden reference recipe: the same operator overrides the
# real-width baseline was verified with (forward verified boundary-by-boundary
# against the ds-code inference implementation).  The case runs the
# reduced-width full-structure debug model (real 40-layer compression/source
# layout, debug widths) and pins its deterministic 100-step loss trajectory
# in tests/assets/losses/dsv41_golden_8p_ep8.txt.
GOLDEN_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rope.workaround",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.golden",
    "torchtitan_npu.override.deepseek_v4_1.golden_moe.golden",
    "torchtitan_npu.override.common.optimizer.virtual",
)

# Environment of the frozen baseline.  MODULE/CONFIG route the trainer to
# the V4.1 multimodal debug model regardless of the runner's CLI defaults
# (per-case env vars take precedence); the golden vision switches select the
# SDPA baseline attention and golden training path; the fixture image is the
# committed tests/assets copy; the dataloader tokenizer is the committed
# tests/assets/deepseek_v3 mini tokenizer (the same one the DeepSeek-V4
# golden cases use).
GOLDEN_ENV = {
    "MODULE": "torchtitan_npu.models.deepseek_v4_1",
    "CONFIG": "deepseek_v4_1_debugmodel",
    "TORCHTITAN_NPU_VISION_GOLDEN": "1",
    "TORCHTITAN_NPU_VISION_SDPA_BASELINE": "1",
    "TORCHTITAN_NPU_GOLDEN_TRAINING": "1",
    "DSV4_TOKENIZER_PATH": "tests/assets/deepseek_v3",
    "DSV4_VISION_IMAGE_PATHS": "tests/assets/dsv4_vit_test.jpeg",
    "DSV4_VISION_LAYERS": "32",
    "DSV4_TRAIN_GOLDEN_REDUCED": "0",
    "TTNPU_DSA_ATTN_CHUNK": "32",
}


def build_deepseek_v4_1_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            override_args=[
                GOLDEN_OVERRIDES
                + (
                    "--training.steps=100",
                    "--training.local-batch-size=1",
                    "--training.global-batch-size=8",
                    "--training.seq-len=512",
                    "--parallelism.spmd-backend=partial_dtensor",
                    "--parallelism.data-parallel-shard-degree=8",
                    "--parallelism.data-parallel-replicate-degree=1",
                    "--parallelism.expert-parallel-degree=8",
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
            test_descr="DeepSeek-V4.1 golden debugmodel 8p fsdp8 ep8 exact loss",
            test_name="dsv41_golden_8p_ep8",
            ngpu=8,
            env_vars=GOLDEN_ENV,
            use_golden=True,
            check_loss=True,
            timeout=7200,
        ),
    ]
