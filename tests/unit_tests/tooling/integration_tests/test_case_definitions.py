# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests.deepseek_v4 import build_deepseek_v4_test_list
from tests.integration_tests.deepseek_v41 import build_deepseek_v41_test_list, build_engram_ascendc_test_list


def test_engram_only_runs_on_v41_with_explicit_operator_selection():
    assert all("engram" not in case.test_name for case in build_deepseek_v4_test_list())
    default = build_deepseek_v41_test_list()
    optional = build_engram_ascendc_test_list()
    assert not any(case.requires_engram_ops for case in default)
    assert all(case.requires_engram_ops for case in optional)
    for case in default + optional:
        if "engram" not in case.test_name:
            continue
        assert case.env_vars["MODULE"] == "torchtitan_npu.models.deepseek_v41"
        assert case.check_resume and case.expected_steps == ((1, 2, 3, 4), (3, 4))
        assert len(case.override_args) == 2
        assert all("--parallelism.context-parallel-degree=1" in args for args in case.override_args)
