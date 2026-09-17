#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on a single node.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh --training.steps 5
# The eager/reference operator path is what V4.1 runs: the AscendC fused
# sparse-attention kernels do not yet accept the ratio-1 shared/global-KV
# contract of CSA2 layers 20-39.

set -euo pipefail

NGPU="${NGPU:-8}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}"
CONFIG="${CONFIG:-deepseek_v4_1_flash_40layers_16experts_vision}"

# Dataloader & Checkpoint
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-tests/assets/c4_test}" # your data path
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/tokenizer/dsv4_tokenizer}" # your tokenizer path

# Parallelism
TP=1
PP=1
EP=8
CP=1
DP_SHARD=8
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training
# The reference fallback is intentionally kept at seq_len=512 for the 8-card
# validation recipe.
SEQ_LEN=512
MBS=1
GBS=8
STEPS=40

# Debug
DEBUG_ARGS="
    --debug.no-moe-force-load-balance
    --debug.print-config
"
if [[ "${DETERMINISTIC:-1}" == "1" ]]; then
    DEBUG_ARGS="${DEBUG_ARGS}
    --debug.seed 42
    --debug.deterministic"
fi

# HF assets
HF_ASSETS_ARGS="
    --hf-assets-path ${HF_ASSETS_PATH}
"

# Dataloader
DATALOADER_ARGS="
    --dataloader.dataset ${DATASET}
    --dataloader.dataset-path ${DATASET_PATH}
"

# Parallelism
PARALLELISM_ARGS="
    --parallelism.spmd-backend ${SPMD_BACKEND}
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.pipeline-parallel-degree ${PP}
    --parallelism.context-parallel-load-balancer None
"

# Compile
COMPILE_ARGS="
    --compile.no-enable
"

# Training
TRAINING_ARGS="
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
"

# Checkpoint
CHECKPOINT_ARGS="
    --checkpoint.no-enable
"

# Profiler
PROFILER_ARGS="
    --profiler.no-enable-profiling
    --profiler.profile-freq 1
    --profiler.profiler-warmup 0
    --profiler.profiler-active 1
    --profiler.profiler-repeat 1
    --profiler.profiler-skip-first 4
    --profiler.extension.profile-ranks 0
    --profiler.extension.no-enable-online-parse
"

# Communication
COMM_ARGS="
    --comm.init-timeout-seconds 7200
    --comm.train-timeout-seconds 600
"

# Optimizer & LR scheduler
OPTIMIZER_ARGS="
    --optimizer.implementation fused
    --optimizer.param-groups.0.optimizer-name AdamW
    --optimizer.param-groups.0.optimizer-kwargs.lr 1.0e-5
    --optimizer.param-groups.0.optimizer-kwargs.betas 0.9 0.95
    --optimizer.param-groups.0.optimizer-kwargs.eps 1.0e-6
    --optimizer.param-groups.0.optimizer-kwargs.weight-decay 1.0e-1
    --lr-scheduler.warmup-steps 25
    --lr-scheduler.decay-type cosine
    --lr-scheduler.decay-ratio 1.0
    --lr-scheduler.min-lr-factor 1.0e-2
    --lr-scheduler.total-steps 40
"
# The upstream scheduler clamps warmup to training.steps, so a short comparison
# run would silently get a different LR curve; total-steps above pins the
# schedule length to the validation recipe instead.
OPTIMIZER_OVERRIDES="
    torchtitan_npu.override.common.optimizer.virtual
"

NPU_OPS_OVERRIDES=(
    torchtitan_npu.override.common.rope.workaround
)

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NGPU="${NGPU}" \
bash scripts/run_train.sh \
    $COMPILE_ARGS \
    $HF_ASSETS_ARGS \
    $DATALOADER_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    "$@"
