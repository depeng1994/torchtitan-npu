# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only GitHub Actions Inputs -> bound JSON Artifact contract."""

from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[4] / ".github/scripts/lite_actions/prepare_request.py"
CASE = {"test_id": "dsv4_flash_a3_8p_example"}


class InputsArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        spec = importlib.util.spec_from_file_location("prepare_ci_request", SCRIPT)
        cls.prepare = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "cann-framework/torchtitan-npu"}):
            spec.loader.exec_module(cls.prepare)

    def test_valid_multiple_cases(self):
        self.assertEqual(self.prepare.parse_cases(json.dumps([CASE, CASE])), [CASE, CASE])

    def test_reject_unsafe_ids_and_removed_parameter_fields(self):
        for test_id in ("../bin/sh", "test;curl", "Invalid-ID"):
            with self.subTest(test_id=test_id), self.assertRaises(ValueError):
                self.prepare.parse_cases(json.dumps([{**CASE, "test_id": test_id}]))
        with self.assertRaises(ValueError):
            self.prepare.parse_cases(json.dumps([{**CASE, "params": {}}]))

    def test_selectors_are_not_arbitrarily_capped(self):
        many = [{"test_id": f"case_{i:03d}"} for i in range(40)]
        self.assertEqual(self.prepare.parse_cases(json.dumps(many)), many)
        with self.assertRaises(ValueError):
            self.prepare.parse_cases("[]")


if __name__ == "__main__":
    unittest.main()
