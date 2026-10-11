# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only CI plan, independent Matrix, waiter and workflow admission."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
PREP = ROOT / ".github/scripts/lite_actions/prepare_request.py"
WAIT = ROOT / ".github/scripts/lite_actions/wait_result.py"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("ci_prepare", PREP)
with patch.dict(os.environ, {"GITHUB_ACTOR": "ci-maintainer"}):
    waiter = load("ci_waiter", WAIT)


class CIContract(unittest.TestCase):
    def test_matrix_and_artifact_identical(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            try:
                os.chdir(td)
                env = {"CI_TEST_CASES": '[{"suite":"a3_8p_tests"}]',
                       "CI_MAX_SELECTORS": "2",
                       "CI_MAX_CASES": "4", "CI_PIPELINE": "a3-8p",
                       "GITHUB_REPOSITORY": "torchtitan-npu/torchtitan-npu",
                       "GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "1",
                       "GITHUB_SHA": "a"*40,
                       "GITHUB_OUTPUT": str(Path(td) / "gh-output.txt")}
                with patch.dict(os.environ, env):
                    # This runs against real model-owned OverrideDefinitions.
                    prepare.main()
                request = json.loads(Path(".ci-request/ci-request.json").read_text())
                matrix = json.loads(Path("gh-output.txt").read_text().splitlines()[0].split("=",1)[1])
                self.assertEqual(request["schema"], 2)
                self.assertEqual(request["repo"], env["GITHUB_REPOSITORY"])
                self.assertEqual(matrix["include"][0]["test_id"],
                                 "dsv4_flash_a3_8p_example")
                self.assertEqual(len(matrix["include"]), 2)
                self.assertEqual([x["test_id"] for x in matrix["include"]],
                                 [x["test_id"] for x in request["cases"]])
                unsigned={k:v for k,v in request.items() if k!="plan_digest"}
                self.assertEqual(request["plan_digest"], prepare.digest_plan(unsigned))
            finally:
                os.chdir(cwd)

    def test_selector_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare.expand_cases([{"suite": "a3_8p_tests"},
                                  {"test_id": "dsv4_flash_a3_8p_adamw"}], 4)

    def test_matrix_result_identity_and_failed_case_isolated(self):
        ref={"schema":2,"repo":"torchtitan-npu/torchtitan-npu",
             "sha":"a"*40,"run_id":91,"attempt":1,"pipeline":"a3-8p",
             "test_id":"dsv4_flash_a3_8p_adamw","plan_digest":"b"*64,
             "status":"FAIL","exit_code":42,"last_20_lines":["OOM"]}
        comment={"user":{"login":"ci-maintainer"},
                 "body":"Lite-CI-RESULT 91:1:dsv4_flash_a3_8p_adamw\n"+json.dumps(ref)}
        q=dict(sha="a"*40,run_id=91,attempt=1,pipeline="a3-8p",
               test_id="dsv4_flash_a3_8p_adamw",plan_digest="b"*64)
        self.assertEqual(waiter.find_case_result([comment],**q),ref)
        for change in ({"attempt":2},{"plan_digest":"c"*64},
                       {"test_id":"dsv4_flash_a3_8p_example"},{"sha":"c"*40}):
            self.assertIsNone(waiter.find_case_result([comment],**{**q,**change}))
        comment["user"]["login"]="attacker"
        self.assertIsNone(waiter.find_case_result([comment],**q))

    def test_waiter_progress_heartbeat_checks_author_and_identity(self):
        args = dict(sha="a"*40,run_id=123,attempt=2,pipeline="a3-8p",
                    test_id="dsv4_flash_a3_8p_example",plan_digest="b"*64)
        self.assertEqual(waiter.find_progress([],**args), "QUEUED")
        marker = "LCI-R 123:2:dsv4_flash_a3_8p_example"
        comment = {"user":{"login":"ci-maintainer"}, "body":marker}
        self.assertEqual(waiter.find_progress([comment],**args),"RUNNING")
        for change in ({"attempt":3},{"test_id":"other_case"}):
            with self.subTest(change=change):
                self.assertEqual(waiter.find_progress([comment],**{**args,**change}),
                                 "QUEUED")
        comment["user"]["login"]="forged-author"
        self.assertEqual(waiter.find_progress([comment],**args),"QUEUED")
        self.assertEqual(
            waiter.heartbeat_line(status="QUEUED",current_case=args["test_id"],
                                  waited=60,run_id=123,attempt=2),
            "CI QUEUED dsv4_flash_a3_8p_example 60s")

    def test_partial_matrix_rerun_without_artifact_fails_fast(self):
        import io
        with patch.object(waiter.urllib.request, "urlopen",
                          return_value=io.BytesIO(
                              b'{"artifacts":[{"name":"lite-ci-request-1",'
                              b'"expired":false}]}')):
            with self.assertRaisesRegex(ValueError, "Partial Matrix reruns"):
                waiter.ensure_attempt_artifact("test-token",1234,2)
        with patch.object(waiter.urllib.request, "urlopen",
                          return_value=io.BytesIO(
                              b'{"artifacts":[{"name":"lite-ci-request-2",'
                              b'"expired":false}]}')):
            waiter.ensure_attempt_artifact("test-token",1234,2)

    def test_org_only_repository_and_comment_author_binding(self):
        org = "torchtitan-npu/torchtitan-npu"
        self.assertEqual(prepare.REPO, org)
        self.assertEqual(waiter.REPO, org)
        cases = [{"test_id":"dsv4_flash_a3_8p_example","nnodes":1,"ngpu":8}]
        plan = prepare.create_plan(cases=cases,run_id=123,attempt=1,
                                   sha="a"*40,repo=org,pipeline="a3-8p")
        self.assertEqual(plan["repo"],org)
        report = {"schema":2,"repo":org,"run_id":123,"attempt":1,
                  "sha":"a"*40,"pipeline":"a3-8p","test_id":cases[0]["test_id"],
                  "plan_digest":"b"*64,"status":"PASS","exit_code":0,
                  "last_20_lines":[]}
        comment = {"user":{"login":"ci-maintainer"},
                   "body":f"Lite-CI-RESULT 123:1:{cases[0]['test_id']}\n"+
                          json.dumps(report)}
        args = dict(sha="a"*40,run_id=123,attempt=1,pipeline="a3-8p",
                    test_id=cases[0]["test_id"],plan_digest="b"*64)
        self.assertEqual(waiter.find_case_result([comment],**args),report)
        comment["user"]["login"]="forged-user"
        self.assertIsNone(waiter.find_case_result([comment],**args))
        with self.assertRaisesRegex(ValueError,"untrusted repository"):
            prepare.create_plan(cases=cases,run_id=123,attempt=1,
                                sha="a"*40,repo="other/repo",pipeline="a3-8p")
    def test_workflow_is_org_scoped_and_ci_pr_branch_dispatchable(self):
        for file in ("a3-8p-lite-actions.yml", "a3-16p-lite-actions.yml"):
            data = (ROOT / ".github/workflows" / file).read_text()
            for expected in ("workflow_dispatch:",
                             "fromJSON(needs.prepare.outputs.matrix)",
                             "fail-fast: false", "--test-id",
                             "actions/upload-artifact@v4",
                             "torchtitan-npu/torchtitan-npu",
                             "pull_request:",
                             "github.event.pull_request.head.sha"):
                self.assertIn(expected, data)
            self.assertNotIn("CI_ALLOWED_ACTOR", data)
            self.assertNotIn("refs/heads/fix/", data)
            self.assertNotIn("runs-on: self-hosted", data)
        a3_8p = (ROOT / ".github/workflows/a3-8p-lite-actions.yml").read_text()
        self.assertIn("refs/heads/master|refs/heads/ci/*)", a3_8p)
        a3_16p = (ROOT / ".github/workflows/a3-16p-lite-actions.yml").read_text()
        for required in ("github.repository == 'torchtitan-npu/torchtitan-npu'",
                         "github.event_name == 'pull_request'",
                         "github.base_ref == 'master'",
                         "github.event_name == 'workflow_dispatch'",
                         "github.ref == 'refs/heads/master'",
                         "startsWith(github.ref, 'refs/heads/ci/')",
                         'CI_MAX_CASES: "2"',
                         "CI_PLAN_DIGEST", "CI_WAIT_TIMEOUT_SECONDS"):
            self.assertIn(required, a3_16p)
        for redundant in ('run-name:', 'max-parallel:', 'CI_MAX_SELECTORS:',
                          'Validate CI admission', 'concurrency:'):
            self.assertNotIn(redundant, a3_16p)

    def test_new_ci_files_have_huawei_copyright(self):
        files = [
            ".github/scripts/lite_actions/prepare_request.py",
            ".github/scripts/lite_actions/wait_result.py",
            *(".github/workflows/" + name for name in (
                "a3-16p-lite-actions.yml", "a3-8p-lite-actions.yml",
                "a5-64p-lite-actions.yml", "lite-actions-dispatcher-smoke.yml")),
            *("tests/integration_tests/nightly_all_models_test/" + name for name in (
                "a3_16p_tests.py", "a3_8p_tests.py", "a5_64p_tests.py", "runner.py")),
            "tests/integration_tests/tools/lite_actions/entrypoint.py",
            "tests/unit_tests/models/deepseek_v4/test_nightly_flash_recipes.py",
            *("tests/unit_tests/tooling/lite_actions/" + name for name in (
                "test_ci_contract.py", "test_entrypoint.py", "test_request.py")),
        ]
        self.assertEqual(len(files), 15)
        copyright = "# Copyright (c) 2026 Huawei Technologies Co., Ltd."
        for name in files:
            with self.subTest(path=name):
                content = (ROOT / name).read_text()
                if content.startswith("#!"):
                    content = content.split("\n", 1)[1]
                self.assertTrue(content.startswith(copyright), name)
                self.assertIn("BSD-style license", "\n".join(content.splitlines()[:6]))

    def test_a5_workflow_is_validly_disabled_without_nonstandard_concurrency_keys(self):
        data=(ROOT/".github/workflows/a5-64p-lite-actions.yml").read_text()
        self.assertIn('CI_CHANNEL_ENABLED: "false"', data)
        self.assertIn("cancel-in-progress: false", data)
        self.assertNotIn("queue:", data)
        self.assertNotIn("runs-on: self-hosted", data)


if __name__ == "__main__":
    unittest.main()
