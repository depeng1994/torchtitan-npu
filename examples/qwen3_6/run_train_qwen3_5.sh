#!/usr/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen3.5 multimodal NPU launcher.
#
# Examples:
#   ./examples/qwen3_6/run_train_qwen3_5.sh --training.steps 5
#   CONFIG=qwen35_debugmodel_moe ./examples/qwen3_6/run_train_qwen3_5.sh \
#     --parallelism.data-parallel-shard-degree 1 \
#     --parallelism.tensor-parallel-degree 1 \
#     --parallelism.pipeline-parallel-degree 1

set -euo pipefail

if [[ -n "${ASCEND_SET_ENV_PATH:-}" ]]; then
    source "${ASCEND_SET_ENV_PATH}"
elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/cann-9.2.0/set_env.sh ]]; then
    source /usr/local/Ascend/cann-9.2.0/set_env.sh
elif [[ -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
    source /usr/local/Ascend/cann-9.1.0/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo "Ascend CANN set_env.sh not found; set ASCEND_SET_ENV_PATH" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export TORCHINDUCTOR_NPU_BACKEND="${TORCHINDUCTOR_NPU_BACKEND:-ascendc}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export MULTI_STREAM_MEMORY_RESERVE="${MULTI_STREAM_MEMORY_RESERVE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

NGPU=${NGPU:-1}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan_npu.models.qwen3_5"}
CONFIG=${CONFIG:-"qwen35_debugmodel"}

REQUIRED_OVERRIDES="torchtitan_npu.override.qwen3_5.gated_delta.npu"
NPU_MOE_DISPATCHER_OVERRIDE="torchtitan_npu.override.common.token_dispatcher.asc"

if [[ ${OVERRIDE_IMPORTS+x} ]]; then
    USER_SET_OVERRIDE_IMPORTS=1
else
    USER_SET_OVERRIDE_IMPORTS=0
    OVERRIDE_IMPORTS="${REQUIRED_OVERRIDES}"
fi

if [[ "${ENABLE_NPU_MOE_DISPATCHER:-0}" == "1" \
      && "${USER_SET_OVERRIDE_IMPORTS}" == "0" ]]; then
    OVERRIDE_IMPORTS="${OVERRIDE_IMPORTS},${NPU_MOE_DISPATCHER_OVERRIDE}"
fi

# TorchTitan is supplied by the container installation.  TORCHTITAN_REPO is
# reserved for tokenizer/dataset assets;
TORCHTITAN_REPO="${TORCHTITAN_REPO:-${TORCHTITAN_DIR:-${ROOT_DIR}/../torchtitan}}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-${TORCHTITAN_REPO}/tests/assets/tokenizer}" # your tokenizer path
DATASET="${DATASET:-cc12m-test}"
DATASET_PATH="${DATASET_PATH:-${TORCHTITAN_REPO}/tests/assets/cc12m_test}" # your data path

ARGS=(
    --module "${MODULE}"
    --config "${CONFIG}"
    --debug.print-config
    --hf-assets-path "${HF_ASSETS_PATH}"
    --dataloader.dataset "${DATASET}"
    --dataloader.dataset-path "${DATASET_PATH}"
)
# Always pass the CLI value, including an explicit empty string.  The CLI
# parser treats "" as an empty import list, which intentionally disables
# config-level overrides for compatibility/debugging runs.
ARGS+=(--override.imports "${OVERRIDE_IMPORTS}")

if [[ -n "${COMPILE_BACKEND:-}" ]]; then
    ARGS+=(
        --compile.enable
        --compile.components model
        --compile.backend "${COMPILE_BACKEND}"
    )
fi

timestamp=$(date +%Y%m%d%H%M%S)
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/../log}"
mkdir -p "${LOG_DIR}"
logfile="qwen3_5_${CONFIG}_${timestamp}.log"

cd "${ROOT_DIR}"
torchrun --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    "${ARGS[@]}" \
    "$@" 2>&1 | tee -a "${LOG_DIR}/${logfile}"
