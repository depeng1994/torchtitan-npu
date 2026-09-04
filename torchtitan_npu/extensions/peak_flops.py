# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ascend peak-FLOPS support for TorchTitan metrics."""

from torchtitan.tools import utils as titan_utils

_upstream_get_peak_flops = titan_utils.get_peak_flops

# MFU uses the BF16 Cube peak rather than the larger total-throughput figure.
_ASCEND_BF16_PEAK_FLOPS = {
    "Ascend910_9392": 353.8944e12,
    "Ascend910B1": 373.88e12,
    "Ascend910B2": 353.8944e12,
    "Ascend910B3": 294.912e12,
    "Ascend910B4": 245.76e12,
}


def get_peak_flops(device_name: str) -> float:
    for model, peak_flops in _ASCEND_BF16_PEAK_FLOPS.items():
        if model in device_name:
            return peak_flops
    return _upstream_get_peak_flops(device_name)


titan_utils.get_peak_flops = get_peak_flops
