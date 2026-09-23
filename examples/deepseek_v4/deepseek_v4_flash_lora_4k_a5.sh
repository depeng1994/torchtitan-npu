#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-deepseek_v4_flash_lora}"
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a5.sh" \
    --checkpoint.no-load-only \
    --checkpoint.save-training-state \
    "$@"
