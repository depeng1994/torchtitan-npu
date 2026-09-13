#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on a single node.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh --training.steps 5
# USE_GOLDEN=1 (the default here) selects the Golden reference operators the
# frozen Stage-01 baseline was produced with; USE_GOLDEN=0 selects the AscendC
# kernels. The frozen baseline is deterministic, so --debug.seed 42 and
# --debug.deterministic are on by default (DETERMINISTIC=0 to drop them).

set -euo pipefail

# Enable model compilation by default; callers can override the backend.
# V4.1 Golden leaves this empty: the frozen baseline was produced without
# compilation, and --compile.enable would change the traced graph.
export COMPILE_BACKEND="${COMPILE_BACKEND:-}"

NGPU="${NGPU:-8}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v41}"
CONFIG="${CONFIG:-deepseek_v41_flash_40layers_16experts_vision}"

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
# V4.1 ratio=2 has no CANN SparseFlashMla metadata kernel yet; the verified
# reference fallback is sized at 512.
SEQ_LEN=512
MBS=1
GBS=8
STEPS=40

# Debug
# DeepSeek-V4.1 uses the golden/reference path only: its ratio-1 CSA2 shared
# global KV contract is not yet supported by the AscendC sparse-attention
# kernels.  USE_GOLDEN=0 (the AscendC path) is rejected below.
export USE_GOLDEN="${USE_GOLDEN:-1}"
if [[ "${USE_GOLDEN}" != "1" ]]; then
    echo "FATAL: DeepSeek-V4.1 currently supports the golden/reference path only."
    echo "The AscendC sparse-attention path does not yet accept the V4.1"
    echo "ratio-1 shared global KV contract (CSA2 layer 20-39)."
    echo "Set USE_GOLDEN=1 (default) or unset USE_GOLDEN."
    exit 2
fi
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

# Training
TRAINING_ARGS="
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
"

# Checkpoint
# The frozen Stage-01 recipe does no checkpoint I/O, so this entry keeps
# `--checkpoint.no-enable` and passes no folder; override on the CLI if needed.
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
# schedule length to the baseline recipe instead.
OPTIMIZER_OVERRIDES="
    torchtitan_npu.override.common.optimizer.virtual
"

# Only the golden/reference path is supported (see the USE_GOLDEN guard above).
NPU_OPS_OVERRIDES=(
    torchtitan_npu.override.common.rope.workaround
    torchtitan_npu.override.deepseek_v4.sparse_attn.golden
    torchtitan_npu.override.deepseek_v41.golden_moe.golden
)

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NGPU="${NGPU}" \
bash scripts/run_train.sh \
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
