# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torchtitan.components.metrics as titan_metrics
from torchtitan.tools import utils as titan_utils

from torchtitan_npu.extensions import peak_flops


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("Ascend910_9392", 353.8944e12),
        ("Ascend910B1", 373.88e12),
        ("Ascend910B2", 353.8944e12),
        ("Ascend910B3", 294.912e12),
        ("Ascend910B4", 245.76e12),
    ],
)
def test_ascend_peak_flops_bypasses_upstream(monkeypatch, device_name, expected):
    def unexpected_upstream_call(device_name):
        raise AssertionError(f"unexpected upstream lookup for {device_name}")

    monkeypatch.setattr(
        peak_flops,
        "_upstream_get_peak_flops",
        unexpected_upstream_call,
    )

    assert titan_utils.get_peak_flops(device_name) == expected


def test_unknown_device_delegates_to_upstream(monkeypatch):
    calls = []

    def upstream_get_peak_flops(device_name):
        calls.append(device_name)
        return 1.0

    monkeypatch.setattr(
        peak_flops,
        "_upstream_get_peak_flops",
        upstream_get_peak_flops,
    )

    assert titan_utils.get_peak_flops("H100") == 1.0
    assert calls == ["H100"]


def test_metrics_processor_uses_installed_peak_flops_hook():
    assert titan_utils.get_peak_flops is peak_flops.get_peak_flops
    assert titan_metrics.utils.get_peak_flops is peak_flops.get_peak_flops
