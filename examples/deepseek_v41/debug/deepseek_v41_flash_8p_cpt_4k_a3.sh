#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on a single node.
# Default entry: the real CC12M caption data (40-layer crop) with the
# verified fusion stack enabled by default.  USE_GOLDEN=1 selects the
# pure reference operator path.  The synthetic vision fixture stays available as the Golden path via
#   CONFIG=deepseek_v41_flash_40layers_16experts_vision ./<this script>
# (its own defaults — STEPS=40, warmup 25, default attention chunk —
# are untouched by the CC12M wiring below).

set -euo pipefail

NGPU="${NGPU:-8}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v41}"
CONFIG="${CONFIG:-deepseek_v41_flash_40layers_16experts_cc12m}"

# Dataloader & Checkpoint
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-tests/assets/c4_test}" # your data path
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/tokenizer/dsv4_tokenizer}" # your tokenizer path

# CC12M data entry wiring (active when a *_cc12m recipe is selected).
CC12M_ARGS=()
if [[ "${CONFIG}" == *_cc12m ]]; then
    export CC12M_MANIFEST_PATH="${CC12M_MANIFEST_PATH:-/data/p00465316/fused/datasets/cc12m/subset_8k/manifest.jsonl}"
    export CC12M_DATA_DIR="${CC12M_DATA_DIR:-/data/p00465316/fused/datasets/cc12m/subset_8k}"
    export CC12M_TOKENIZER_PATH="${CC12M_TOKENIZER_PATH:-/data/p00465316/fused/dsv41_tokenizer}"
    if [[ ! -f "${CC12M_MANIFEST_PATH}" ]]; then
        echo "FATAL: CC12M manifest not found at ${CC12M_MANIFEST_PATH}."
        echo "Prepare the subset once with examples/deepseek_v41/prepare_cc12m.py"
        echo "(see examples/deepseek_v41/readme.md)."
        exit 2
    fi
    # Verified-shape pins: the 40-layer backward workspace does not fit
    # with the default attention chunk 256 (verified OOM); chunk 128 runs
    # at 73.5% HBM.
    export TTNPU_DSA_ATTN_CHUNK="${TTNPU_DSA_ATTN_CHUNK:-128}"
    export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
    export HF_ASSETS_PATH="${CC12M_TOKENIZER_PATH}"
    # Run length: STEPS is the only knob — it keeps training.steps and
    # the LR schedule (total-steps=STEPS, warmup=2) in sync.  Direct CLI
    # overrides are rejected so a run can never desynchronize them.
    for arg in "$@"; do
        case "$arg" in
            --training.steps|--training.steps=*|\
            --lr-scheduler.total-steps|--lr-scheduler.total-steps=*|\
            --lr-scheduler.warmup-steps|--lr-scheduler.warmup-steps=*)
                echo "FATAL: '${arg}' would desynchronize the LR schedule from the run length."
                echo "Use STEPS=<n> instead; it configures training.steps, total-steps and"
                echo "warmup-steps together."
                exit 2
                ;;
        esac
    done
    STEPS_CC12M="${STEPS:-40}"
    CC12M_ARGS=(--training.steps "${STEPS_CC12M}" --lr-scheduler.total-steps "${STEPS_CC12M}" --lr-scheduler.warmup-steps 2)
fi

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
# validation recipe.  STEPS stays hardcoded for the synthetic fixture
# (schedule pinned below); the CC12M entry overrides steps/total/warmup
# together via CC12M_ARGS.
SEQ_LEN=512
MBS=1
GBS=8
STEPS=40

# Debug
export USE_GOLDEN="${USE_GOLDEN:-0}"
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
# schedule length to the validation recipe instead.  The CC12M entry appends
# its own steps/total/warmup overrides (see CC12M_ARGS above), which take
# precedence as later CLI values.
OPTIMIZER_OVERRIDES="
    torchtitan_npu.override.common.optimizer.virtual
"

NPU_OPS_OVERRIDES=()
if [[ "${USE_GOLDEN}" == "1" ]]; then
    # Golden/reference path (USE_GOLDEN=1): pure reference operators,
    # bit-equivalent to the FUSION50 B0 baseline.
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rope.workaround
    )
else
    # Fused path (default): the FUSION50-accepted stack — rope/moe
    # showed no observable trajectory difference in the containing
    # combination; rms_norm/mhc post differ by <= 4.8e-3 random-signed
    # with no drift (accepted).
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.deepseek_v41.rms_norm.ascendc
        torchtitan_npu.override.deepseek_v41.rope.ascendc
        torchtitan_npu.override.deepseek_v41.moe.ascendc
        torchtitan_npu.override.deepseek_v41.mhc.asc_hc_post
    )
    # A5-only fused operators, one switch: sparse attention (ratio-2
    # cmp_topk=256 rejected on A3) + mHC Sinkhorn (operator package
    # missing on A3).  The A5 wrapper enables this by default; Golden
    # always stays reference.
    if [[ "${ENABLE_A5_FUSION:-0}" == "1" ]]; then
        NPU_OPS_OVERRIDES+=(
            torchtitan_npu.override.deepseek_v41.sparse_attn.asc_metadata
            torchtitan_npu.override.deepseek_v41.sparse_attn.asc
            torchtitan_npu.override.deepseek_v41.mhc.asc_sinkhorn
        )
    fi
fi

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
    ${CC12M_ARGS[@]+"${CC12M_ARGS[@]}"} \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    "$@"
