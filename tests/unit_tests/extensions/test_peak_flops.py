# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def peak_flops_extension():
    calls = []

    utils = types.ModuleType("torchtitan.tools.utils")

    def upstream_get_peak_flops(device_name: str) -> float:
        calls.append(device_name)
        return 1.0

    utils.get_peak_flops = upstream_get_peak_flops
    tools = types.ModuleType("torchtitan.tools")
    tools.utils = utils
    torchtitan = types.ModuleType("torchtitan")
    torchtitan.tools = tools

    modules = {
        "torchtitan": torchtitan,
        "torchtitan.tools": tools,
        "torchtitan.tools.utils": utils,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)

    path = Path(__file__).resolve().parents[3] / "torchtitan_npu" / "extensions" / "peak_flops.py"
    spec = importlib.util.spec_from_file_location("peak_flops_test_impl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._calls = calls
    module._utils = utils

    yield module

    sys.modules.pop(spec.name, None)
    for name, previous_module in previous.items():
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


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
def test_ascend_peak_flops_bypasses_upstream(peak_flops_extension, device_name, expected):
    assert peak_flops_extension.get_peak_flops(device_name) == expected
    assert peak_flops_extension._calls == []


def test_unknown_device_delegates_to_upstream(peak_flops_extension):
    assert peak_flops_extension.get_peak_flops("H100") == 1.0
    assert peak_flops_extension._calls == ["H100"]


def test_extension_installs_peak_flops_hook(peak_flops_extension):
    assert peak_flops_extension._utils.get_peak_flops is peak_flops_extension.get_peak_flops
