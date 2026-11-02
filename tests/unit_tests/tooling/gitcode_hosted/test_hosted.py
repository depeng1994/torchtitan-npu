"""CPU boundary tests; no platform or hardware acceptance is implied."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "hosted", ROOT / ".github/scripts/lite_actions/gitcode/hosted.py"
)
hosted = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hosted)


class HostedTests(unittest.TestCase):
    def setUp(self):
        self.policy = SimpleNamespace(
            github_repo_id=1210038933,
            control_sha="a" * 40,
            enabled=True,
            workflow_path=".github/workflows/lite_actions_dispatcher_smoke.yml",
            max_duration_seconds=100,
        )
        repo, branch = hosted.MODEL_REPOSITORIES[self.policy.github_repo_id]
        self.env = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_REPOSITORY": repo,
            "GITHUB_REPOSITORY_ID": "1210038933",
            "GITHUB_REF": "refs/heads/" + branch,
            "GITHUB_SHA": "a" * 40,
            "GITHUB_WORKFLOW_SHA": "a" * 40,
            "GITHUB_WORKFLOW_REF": repo + "/" + self.policy.workflow_path + "@refs/heads/" + branch,
            "GITHUB_RUN_ID": "900",
        }

    def test_prepare_delegates_exact_run_and_separated_paths(self):
        producer = Mock(policy=self.policy)
        producer.prepare.return_value = {"prepared": True}
        result = hosted.prepare(
            environment=self.env,
            producer=producer,
            intent={"data": "only"},
            runner_path=Path("/control/runner"),
            workspace=Path("/new"),
        )
        self.assertEqual(result, {"prepared": True})
        producer.prepare.assert_called_once_with(
            run_id=900,
            intent={"data": "only"},
            runner_path=Path("/control/runner"),
            workspace=Path("/new"),
        )

    def test_each_context_mismatch_blocks_before_prepare(self):
        for key in self.env:
            with self.subTest(key=key):
                producer = Mock(policy=self.policy)
                environment = {**self.env, key: "wrong"}
                with self.assertRaises(hosted.HostedError):
                    hosted.prepare(
                        environment=environment,
                        producer=producer,
                        intent={},
                        runner_path=Path("/runner"),
                        workspace=Path("/new"),
                    )
                producer.prepare.assert_not_called()

    def test_disabled_release_blocks(self):
        self.policy.enabled = False
        with self.assertRaisesRegex(hosted.HostedError, "DISABLED"):
            hosted.validate_context(self.env, self.policy)

    def test_rerun_points_to_bridge(self):
        for attempt in ("2", "01", "", None):
            with self.assertRaisesRegex(hosted.HostedError, "USE_GITCODE_BRIDGE"):
                hosted.validate_context({**self.env, "GITHUB_RUN_ATTEMPT": attempt}, self.policy)

    def test_uploader_single_invocation_and_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tar.gz"
            path.write_bytes(b"source")
            with patch.object(
                hosted.subprocess, "run", return_value=SimpleNamespace(stdout=b'{"id": 82}')
            ) as run:
                self.assertEqual(
                    hosted.ArtifactUploader(["node", "/trusted/upload.cjs"])("source-900-1", path),
                    82,
                )
                self.assertEqual(run.call_count, 1)
                self.assertEqual(
                    run.call_args.args[0],
                    ["node", "/trusted/upload.cjs", "source-900-1", str(path)],
                )
                self.assertNotIn("shell", run.call_args.kwargs)

    def test_uploader_unknown_outcomes_are_recoverable_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tar.gz"
            path.write_bytes(b"source")
            for reply in (b"progress", b'{"id": true}', b'{"id": 0}', b'{"id": 1, "extra": 2}'):
                with patch.object(
                    hosted.subprocess, "run", return_value=SimpleNamespace(stdout=reply)
                ) as run:
                    with self.assertRaises(OSError):
                        hosted.ArtifactUploader(["trusted"])("source-1", path)
                    self.assertEqual(run.call_count, 1)
            for failure in (
                subprocess.TimeoutExpired("trusted", 1),
                subprocess.CalledProcessError(1, "trusted"),
            ):
                with patch.object(hosted.subprocess, "run", side_effect=failure) as run:
                    with self.assertRaises(OSError):
                        hosted.ArtifactUploader(["trusted"])("source-1", path)
                    self.assertEqual(run.call_count, 1)

    def test_uploader_rejects_symlink_without_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"x")
            link = Path(directory) / "link"
            link.symlink_to(path)
            with patch.object(hosted.subprocess, "run") as run:
                with self.assertRaises(hosted.HostedError):
                    hosted.ArtifactUploader(["trusted"])("source-1", link)
                run.assert_not_called()

    def wait_args(self, reader):
        self.now = 100

        def sleep(seconds):
            self.now += seconds

        return dict(
            environment=self.env,
            reader=reader,
            workspace="/verified",
            current=lambda _: True,
            clock=lambda: self.now,
            sleep=sleep,
            timeout_seconds=30,
            poll_seconds=15,
        )

    def test_wait_requires_loaded_verified_request_before_poll(self):
        loaded = Mock(verified=object())
        reader = Mock(policy=self.policy)
        reader.load.side_effect = [None, loaded]
        waiter = Mock(plan={"deadline": 200})
        waiter.poll.side_effect = [{"test_result": "PENDING"}, {"test_result": "PASS"}]
        args = self.wait_args(reader)
        args["timeout_seconds"] = 45
        with patch.object(hosted, "GitCodeWaiter", return_value=waiter) as factory:
            result = hosted.wait(**args)
        self.assertEqual(result, {"test_result": "PASS"})
        self.assertEqual(reader.load.call_count, 2)
        loaded.verify_files.assert_called_once()
        factory.assert_called_once_with(
            loaded.verified, reader.channel, current=args["current"], resolve_canonical=None
        )
        self.assertEqual(self.now, 130)

    def test_wait_timeout_never_infers_success_from_absent_artifact(self):
        reader = Mock(policy=self.policy)
        reader.load.return_value = None
        with (
            patch.object(hosted, "GitCodeWaiter") as factory,
            self.assertRaisesRegex(hosted.HostedError, "TIMEOUT"),
        ):
            hosted.wait(**self.wait_args(reader))
        factory.assert_not_called()
        self.assertEqual(self.now, 130)

    def test_failed_prepare_and_network_errors_propagate(self):
        for failure in (ValueError("PREPARE_NOT_SUCCESSFUL"), OSError("network")):
            reader = Mock(policy=self.policy)
            reader.load.side_effect = failure
            with patch.object(hosted, "GitCodeWaiter") as factory:
                with self.assertRaises(type(failure)):
                    hosted.wait(**self.wait_args(reader))
                factory.assert_not_called()

    def test_terminal_fail_and_stale_not_rewritten_to_pass(self):
        for state in ("FAIL", "STALE"):
            reader = Mock(policy=self.policy)
            waiter = Mock(plan={"deadline": 200})
            waiter.poll.return_value = {"test_result": state, "merge_gate": "NOT_PROVIDED"}
            with patch.object(hosted, "GitCodeWaiter", return_value=waiter):
                self.assertEqual(hosted.wait(**self.wait_args(reader)), waiter.poll.return_value)

    def test_invalid_budget_rejected_before_loading(self):
        for budget in (0, -1, 101, True):
            reader = Mock(policy=self.policy)
            args = self.wait_args(reader)
            args["timeout_seconds"] = budget
            with self.assertRaises(hosted.HostedError):
                hosted.wait(**args)
            reader.load.assert_not_called()

    def test_node_adapter_returns_only_id_and_exact_single_file(self):
        import json
        import os

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "node_modules/@actions/artifact"
            package.mkdir(parents=True)
            (package / "package.json").write_text('{"type":"module","exports":"./index.js"}')
            (package / "index.js").write_text(
                "import fs from 'node:fs'; export class DefaultArtifactClient { async uploadArtifact(name,files,root,options) {"
                "fs.writeFileSync(process.env.CAPTURE,JSON.stringify({name,files,root,options}));"
                "console.log('sdk progress'); return {id: 91}; }}"
            )
            uploader = root / "upload_artifact.cjs"
            uploader.write_bytes(
                (ROOT / ".github/scripts/lite_actions/gitcode/upload_artifact.cjs").read_bytes()
            )
            source = root / "source.tar.gz"
            source.write_bytes(b"opaque source; never executable")
            capture = root / "args.json"
            result = subprocess.run(
                [
                    "node",
                    str(uploader),
                    "source-900-1",
                    str(source),
                ],
                env={**os.environ, "CAPTURE": str(capture)},
                capture_output=True,
                check=True,
            )
            self.assertEqual(result.stdout, b'{"id":91}\n')
            self.assertIn(b"sdk progress", result.stderr)
            self.assertEqual(
                json.loads(capture.read_text()),
                {
                    "name": "source-900-1",
                    "files": [str(source)],
                    "root": str(root),
                    "options": {"retentionDays": 7},
                },
            )

    def test_plan_deadline_bounds_waiter_sleep(self):
        reader = Mock(policy=self.policy)
        waiter = Mock(plan={"deadline": 103})
        waiter.poll.return_value = {"test_result": "PENDING"}
        with (
            patch.object(hosted, "GitCodeWaiter", return_value=waiter),
            self.assertRaisesRegex(hosted.HostedError, "TIMEOUT"),
        ):
            hosted.wait(**self.wait_args(reader))
        self.assertEqual(self.now, 103)
        self.assertEqual(waiter.poll.call_count, 1)

    def test_slow_load_crossing_plan_deadline_cannot_poll(self):
        reader = Mock(policy=self.policy)
        waiter = Mock(plan={"deadline": 105})
        args = self.wait_args(reader)

        def load(*args, **kwargs):
            self.now = 106
            return Mock(verified=object())

        reader.load.side_effect = load
        with (
            patch.object(hosted, "GitCodeWaiter", return_value=waiter),
            self.assertRaisesRegex(hosted.HostedError, "TIMEOUT"),
        ):
            hosted.wait(**args)
        waiter.poll.assert_not_called()

    def test_slow_poll_crossing_global_deadline_cannot_pass(self):
        reader = Mock(policy=self.policy)
        waiter = Mock(plan={"deadline": 200})
        args = self.wait_args(reader)

        def poll(**kwargs):
            self.now = 130
            return {"test_result": "PASS"}

        waiter.poll.side_effect = poll
        with (
            patch.object(hosted, "GitCodeWaiter", return_value=waiter),
            self.assertRaisesRegex(hosted.HostedError, "TIMEOUT"),
        ):
            hosted.wait(**args)
        waiter.poll.assert_called_once_with(now=100)


class HostedDependencyTests(unittest.TestCase):
    def test_artifact_sdk_and_transitives_are_locked_without_install_scripts(self):
        import json

        path = ROOT / ".github/scripts/lite_actions/gitcode"
        package = json.loads((path / "package.json").read_text())
        lock = json.loads((path / "package-lock.json").read_text())
        self.assertEqual(package["dependencies"], {"@actions/artifact": "6.3.1"})
        self.assertEqual(lock["packages"]["node_modules/@actions/artifact"]["version"], "6.3.1")
        self.assertEqual(lock["packages"][""]["dependencies"], package["dependencies"])
        for name, entry in lock["packages"].items():
            if name:
                self.assertTrue(entry["resolved"].startswith("https://registry.npmjs.org/"), name)
                self.assertTrue(entry["integrity"].startswith("sha512-"), name)
        self.assertEqual((path / ".gitignore").read_text(), "node_modules/\n")

    def test_uploader_receives_no_platform_or_app_secret(self):
        import os

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tar.gz"
            source.write_bytes(b"source")
            with (
                patch.dict(
                    os.environ,
                    {
                        "GH_TOKEN": "synthetic-gh-secret",
                        "LITE_GITCODE_APP_PRIVATE_KEY": "synthetic-app-secret",
                        "LITE_ACTIONS_GITCODE_TOKEN": "synthetic-gc-secret",
                        "NODE_OPTIONS": "--require=untrusted",
                        "ACTIONS_RUNTIME_TOKEN": "synthetic-artifact-credential",
                    },
                ),
                patch.object(
                    hosted.subprocess, "run", return_value=SimpleNamespace(stdout=b'{"id": 82}')
                ) as run,
            ):
                self.assertEqual(
                    hosted.ArtifactUploader(["node", "/trusted/upload.cjs"])(
                        "source-900-1", source
                    ),
                    82,
                )
                actual = run.call_args.kwargs["env"]
                self.assertEqual(actual["ACTIONS_RUNTIME_TOKEN"], "synthetic-artifact-credential")
                for key in (
                    "GH_TOKEN",
                    "LITE_GITCODE_APP_PRIVATE_KEY",
                    "LITE_ACTIONS_GITCODE_TOKEN",
                    "NODE_OPTIONS",
                ):
                    self.assertNotIn(key, actual)
