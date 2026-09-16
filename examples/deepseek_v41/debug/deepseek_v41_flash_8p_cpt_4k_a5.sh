#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 entry: same training parameters as the A3 script, with the
# A5-only fused operators (sparse attention + mHC Sinkhorn) enabled by
# default.  USE_GOLDEN=1 still selects the pure reference path;
# ENABLE_A5_FUSION=0 disables them on A5.  All other arguments pass
# through unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_A5_FUSION="${ENABLE_A5_FUSION:-1}"

# Use npu-smi info -t topo to check CPU affinity on the target host.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

exec bash "${SCRIPT_DIR}/deepseek_v41_flash_8p_cpt_4k_a3.sh" "$@"
