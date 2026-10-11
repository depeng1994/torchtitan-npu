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
                       "CI_PIPELINE": "a3-8p",
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

    def test_expanded_suite_has_no_configured_budget(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        module = 'tests.integration_tests.tools.lite_actions.entrypoint'
        cases = [SimpleNamespace(test_name=f'case_{i:03d}', nnodes=1, ngpu=8,
                   test_descr=f'Test case {i}') for i in range(15)]
        with patch(f'{module}.catalog', return_value={'synthetic_tests': cases}):
            expanded = prepare.expand_cases([{'suite': 'synthetic_tests'}])
        self.assertEqual(len(expanded), 15)

    def test_ci_port_override_wins_over_multinode_recipe_defaults(self):
        from unittest.mock import patch
        from tests.integration_tests.nightly_all_models_test.runner import dispatcher_port_env
        with patch.dict("os.environ", {"LITE_CI_MASTER_PORT": "16007",
                                    "LITE_CI_HCCL_IF_BASE_PORT": "20448"}):
            self.assertEqual(dispatcher_port_env(),
                             {"MASTER_PORT": "16007", "HCCL_IF_BASE_PORT": "20448"})
        for invalid in ("-1", "0", "65536", "not-a-port"):
            with patch.dict("os.environ", {"LITE_CI_MASTER_PORT": invalid}):
                with self.assertRaises(ValueError):
                    dispatcher_port_env()

    def test_selector_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare.expand_cases([{"suite": "a3_8p_tests"},
                                  {"test_id": "dsv4_flash_a3_8p_adamw"}])

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
            waiter.heartbeat_line(status="QUEUED", current_case=args["test_id"],
                                  when=waiter.dt.datetime(2026,10,11,13,7,10,
                                                          tzinfo=waiter.dt.timezone.utc)),
            "2026/10/11 21:07:10 dsv4_flash_a3_8p_example CI queued.")

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
    def test_ci_workflow_reuse_contract(self):
        import yaml
        root = ROOT / ".github/workflows"
        def read(name):
            return yaml.load((root / name).read_text(), Loader=yaml.BaseLoader)
        common = read("lite_actions_common.yml")
        self.assertEqual(list(common["on"]), ["workflow_call"])
        self.assertEqual(set(common["jobs"]), {"prepare", "case", "smoke"})
        inputs = common["on"]["workflow_call"]["inputs"]
        self.assertEqual(inputs["pipeline"]["required"], "true")
        self.assertEqual(inputs["pipeline"]["type"], "string")
        self.assertNotIn("max_cases", inputs)
        prepare = common["jobs"]["prepare"]
        matrix = common["jobs"]["case"]
        smoke = common["jobs"]["smoke"]
        self.assertIn("torchtitan-npu/torchtitan-npu", prepare["if"])
        self.assertIn("github.base_ref == 'master'", prepare["if"])
        self.assertIn("startsWith(github.ref, 'refs/heads/ci/')", prepare["if"])
        self.assertEqual(matrix["needs"], "prepare")
        self.assertIn("needs.prepare.result == 'success'",matrix["if"])
        self.assertEqual(matrix["strategy"]["fail-fast"], "false")
        self.assertIn("fromJSON(needs.prepare.outputs.matrix)", matrix["strategy"]["matrix"])
        self.assertEqual(matrix["timeout-minutes"], "300")
        plan = prepare["steps"][1]
        self.assertNotIn("CI_MAX_SELECTORS", plan["env"])
        self.assertNotIn("CI_MAX_CASES", plan["env"])
        self.assertIn("inputs.pipeline", plan["env"]["CI_PIPELINE"])
        self.assertIn("github.event.pull_request.head.sha", plan["env"]["CI_SOURCE_SHA"])
        self.assertIn("inputs.default_cases", plan["env"]["CI_TEST_CASES"])
        artifact = prepare["steps"][2]
        self.assertEqual(artifact["uses"], "actions/upload-artifact@v4")
        self.assertEqual(artifact["with"]["if-no-files-found"], "error")
        self.assertEqual(artifact["with"]["retention-days"], "1")
        waiter = matrix["steps"][1]
        for env in ("CI_SOURCE_SHA", "CI_RESULT_AUTHOR", "CI_PLAN_DIGEST",
                    "CI_WAIT_TIMEOUT_SECONDS"):
            self.assertIn(env, waiter["env"])
        self.assertIn("--test-id", waiter["run"])
        self.assertIn("--case dispatcher-smoke", smoke["steps"][1]["run"])
        self.assertEqual(smoke["timeout-minutes"], "90")
        self.assertIn("github.base_ref == 'master'", smoke["if"])
        for filename,pipeline,default in (
            ("a3_8p_lite_actions.yml", "a3-8p", "a3_8p_tests"),
            ("a3_16p_lite_actions.yml", "a3-16p", "a3_16p_tests"),
            ("a5_64p_lite_actions.yml", "a5-64p", "a5_64p_tests"),
        ):
            with self.subTest(filename=filename):
                data=read(filename)
                self.assertEqual(list(data["jobs"]), ["ci"])
                caller=data["jobs"]["ci"]
                self.assertEqual(caller["uses"], "./.github/workflows/lite_actions_common.yml")
                self.assertEqual(caller["with"]["pipeline"], pipeline)
                self.assertNotIn("max_cases", caller["with"])
                self.assertIn(default, caller["with"]["default_cases"])
                self.assertIn("inputs.test_cases", caller["with"]["test_cases"])
                self.assertEqual(data["permissions"], common["permissions"])
                self.assertIn("workflow_dispatch", data["on"])
                if pipeline=="a5-64p":
                    self.assertNotIn("pull_request", data["on"])
                    self.assertEqual(caller["if"], chr(36)+"{{ false }}")
                else:
                    self.assertEqual(data["on"]["pull_request"]["branches"], ["master"])
        smoke_caller=read("lite_actions_dispatcher_smoke.yml")
        self.assertEqual(
            smoke_caller["jobs"]["ci"]["uses"],"./.github/workflows/lite_actions_common.yml")
        self.assertEqual(smoke_caller["jobs"]["ci"]["with"]["pipeline"],"dispatcher-smoke")
        self.assertEqual(smoke_caller["permissions"],common["permissions"])
        self.assertEqual(smoke_caller["jobs"]["ci"]["with"]["wait_seconds"],"4800")

    def test_model_owned_plan_is_unchanged_after_reuse(self):
        selected=prepare.expand_cases([{"suite":"a3_16p_tests"}])
        self.assertEqual([c["test_id"] for c in selected],
                         ["dsv4_flash_a3_16p_example"])
        self.assertEqual((selected[0]["nnodes"],selected[0]["ngpu"]),(2,8))

    def test_new_ci_files_have_huawei_copyright(self):
        files = [
            ".github/scripts/lite_actions/prepare_request.py",
            ".github/scripts/lite_actions/wait_result.py",
            *(".github/workflows/" + name for name in (
                "a3_16p_lite_actions.yml", "a3_8p_lite_actions.yml",
                "a5_64p_lite_actions.yml", "lite_actions_dispatcher_smoke.yml",
                "lite_actions_common.yml")),
            *("tests/integration_tests/nightly_all_models_test/" + name for name in (
                "a3_16p_tests.py", "a3_8p_tests.py", "a5_64p_tests.py", "runner.py")),
            "tests/integration_tests/tools/lite_actions/entrypoint.py",
            "tests/unit_tests/models/deepseek_v4/test_nightly_flash_recipes.py",
            *("tests/unit_tests/tooling/lite_actions/" + name for name in (
                "test_ci_contract.py", "test_entrypoint.py", "test_request.py")),
        ]
        self.assertEqual(len(files), 16)
        copyright = "# Copyright (c) 2026 Huawei Technologies Co., Ltd."
        for name in files:
            with self.subTest(path=name):
                content = (ROOT / name).read_text()
                if content.startswith("#!"):
                    content = content.split("\n", 1)[1]
                self.assertTrue(content.startswith(copyright), name)
                self.assertIn("BSD-style license", "\n".join(content.splitlines()[:6]))

    def test_unrelated_workflows_remain_independent(self):
        for name in ("deploy-docs.yml","sync-upstream.yml"):
            data=(ROOT/".github/workflows"/name).read_text()
            self.assertNotIn("lite_actions_common.yml",data)
            self.assertIn("secrets.DEPLOY_KEY",data)



if __name__ == "__main__":
    unittest.main()
