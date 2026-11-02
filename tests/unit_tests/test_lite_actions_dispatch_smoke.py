# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the BSD-style license found in the LICENSE file.

import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test import runner
from tests.integration_tests.tools.lite_actions.contract import normalize
from tests.integration_tests.tools.lite_actions.entrypoint import catalog, select


class LiteActionsDispatchSmokeTests(unittest.TestCase):
    def setUp(self):
        self.case = OverrideDefinitions(test_name="renamed_smoke", ci_kind="npu_smoke", ngpu=8, timeout=120)
        self.defaults = dict(worker="blue", target_pool="a3_shared_blue", docker="image:tag")

    def test_registered_topologies_and_training_independence(self):
        cases = catalog()
        for test_id, nodes in (("dispatch_smoke_8p", 1), ("dispatch_smoke_16p", 2)):
            case = select(cases, test_id=test_id)[0]
            self.assertEqual((case.nnodes, case.ngpu), (nodes, 8))
            self.assertTrue(runner.is_dispatch_smoke(case))
            self.assertFalse(case.check_loss)
            self.assertIsNone(case.expected_steps)

    def test_kind_not_name_selects_smoke(self):
        self.assertTrue(runner.is_dispatch_smoke(replace(self.case, test_name="dispatch_smoke_32p", nnodes=4)))
        self.assertFalse(runner.is_dispatch_smoke(replace(self.case, test_name="dispatch_smoke_8p", ci_kind="train")))

    def test_single_manual_launch_uses_standalone_and_declared_timeout(self):
        for timeout in (120, None):
            case = replace(self.case, timeout=timeout)
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
                with patch.object(runner.subprocess, "run") as launch:
                    runner.run_dispatch_smoke(case, output_dir=Path(tmp))
                self.assertIn("--standalone", launch.call_args.args[0])
                self.assertEqual(
                    launch.call_args.kwargs["timeout"], normalize(case, self.defaults, 0)["execution_timeout_seconds"]
                )
                runner.verify_dispatch_smoke(case, Path(tmp))

    def test_manual_multinode_uses_explicit_rank_ports_and_derived_world(self):
        case = replace(self.case, nnodes=4, ngpu=2)
        env = dict(
            NODE_IPS="192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4",
            NODE_RANK="2",
            LITE_CI_MASTER_PORT="16002",
            LITE_CI_HCCL_IF_BASE_PORT="20256",
            LITE_CI_HCCL_HOST_PORT_RANGE="24256-24319",
            LITE_CI_HCCL_NPU_PORT_RANGE="28256-28319",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            with patch.object(runner.subprocess, "run") as launch:
                runner.run_dispatch_smoke(case, output_dir=Path(tmp))
            argv = launch.call_args.args[0]
            self.assertEqual(argv[argv.index("--node-rank") + 1], "2")
            self.assertEqual(launch.call_args.kwargs["env"]["LITE_SMOKE_EXPECTED_WORLD"], "8")
            self.assertEqual(launch.call_args.kwargs["env"]["HCCL_IF_BASE_PORT"], "20256")
            runner.verify_dispatch_smoke(case, Path(tmp))

    def test_multinode_missing_explicit_port_fails_before_launch(self):
        env = dict(NODE_IPS="192.0.2.1,192.0.2.2", NODE_RANK="0")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            with patch.object(runner.subprocess, "run") as launch:
                with self.assertRaisesRegex(RuntimeError, "explicit LITE_CI_MASTER_PORT"):
                    runner.run_dispatch_smoke(replace(self.case, nnodes=2), output_dir=Path(tmp))
                launch.assert_not_called()

    def test_ci_never_falls_back_to_standalone_without_complete_lease(self):
        env = dict(LITE_CI_JOB_ID="test-job", NODE_IPS="192.0.2.1", LITE_CI_MASTER_PORT="16002")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            with patch.object(runner.subprocess, "run") as launch:
                with self.assertRaisesRegex(RuntimeError, "complete dispatcher port lease"):
                    runner.run_dispatch_smoke(self.case, output_dir=Path(tmp))
                launch.assert_not_called()

    def test_invalid_hccl_port_is_rejected_by_shared_validator(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LITE_CI_HCCL_IF_BASE_PORT": "bad"}, clear=True),
        ):
            with patch.object(runner.subprocess, "run") as launch:
                with self.assertRaisesRegex(ValueError, "LITE_CI_HCCL_IF_BASE_PORT"):
                    runner.run_dispatch_smoke(self.case, output_dir=Path(tmp))
                launch.assert_not_called()

    def test_timeout_error_uses_same_budget_and_never_writes_pass(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("torchrun", 120)):
                with self.assertRaisesRegex(RuntimeError, "120-second"):
                    runner.run_dispatch_smoke(self.case, output_dir=Path(tmp))
            self.assertFalse((Path(tmp) / "dispatch_smoke_result.json").exists())

    def test_verify_rejects_wrong_test_or_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dict(status="PASS", test_id="another_case", world_size=8, nodes=1, devices_per_node=8)
            (Path(tmp) / "dispatch_smoke_result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError, "invalid dispatch smoke result"):
                runner.verify_dispatch_smoke(self.case, Path(tmp))


if __name__ == "__main__":
    unittest.main()
