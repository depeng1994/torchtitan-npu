"""CPU tests for the trusted runner candidate; no platform or hardware evidence."""

import argparse
import copy
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "trusted_gitcode_" + name, ROOT / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build_runner")
runner = load("runner")


class BundleTests(unittest.TestCase):
    def test_bundle_is_deterministic_and_contains_no_proposed_payload(self):
        first = builder.build()
        self.assertEqual(first, builder.build())
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
            self.assertEqual(archive.getnames(), list(builder.FILES))
            self.assertNotIn("smoke_payload.py", archive.getnames())
            for member in archive:
                self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
                self.assertTrue(member.isfile())

    def test_fixed_smoke_and_nightly_profiles(self):
        document = json.loads((ROOT / "catalog.json").read_text())
        builder.validate_catalog(document)
        self.assertEqual(
            document["pipelines"]["smoke"],
            ["worker_os_blue", "dispatch_smoke_8p", "dispatch_smoke_16p"],
        )
        self.assertEqual(
            document["cases"]["dispatch_smoke_16p"]["resources"],
            {"nodes": 2, "devices_per_node": 8},
        )
        profile = document["profiles"]["dsv4_flash_a3_16p_example"]
        args = profile["train_args"]
        for flag, value in (
            ("--parallelism.expert-parallel-degree", "16"),
            ("--training.global-batch-size", "128"),
            ("--training.steps", "5"),
            ("--optimizer.name", "AdamW"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(profile["expected_steps"], [1, 2, 3, 4, 5])
        self.assertNotIn("MASTER_PORT", profile["env_vars"])

    def test_changed_resource_owner_container_timeout_or_case_set_is_rejected(self):
        original = json.loads((ROOT / "catalog.json").read_text())
        for field, value in (
            ("worker", "yellow"),
            ("docker", "vllm_hccl_smoke"),
            ("execution_timeout_seconds", 999),
        ):
            document = copy.deepcopy(original)
            document["cases"]["dispatch_smoke_8p"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                builder.validate_catalog(document)
        document = copy.deepcopy(original)
        document["pipelines"]["smoke"].pop()
        with self.assertRaises(ValueError):
            builder.validate_catalog(document)

    def test_secret_or_source_path_injection_is_rejected(self):
        original = json.loads((ROOT / "catalog.json").read_text())
        document = copy.deepcopy(original)
        document["profiles"]["dispatch_smoke_8p"]["env_vars"]["GH_TOKEN"] = (
            "not-a-real-token"
        )
        with self.assertRaises(ValueError):
            builder.validate_catalog(document)
        document = copy.deepcopy(original)
        document["profiles"]["dispatch_smoke_8p"]["source_paths"] = ["../../outside"]
        with self.assertRaises(ValueError):
            builder.validate_catalog(document)

    def test_module_import_and_bundle_build_do_not_import_torch(self):
        before = set(sys.modules)
        load("runner")
        load("smoke_rank")
        load("_lite_gitcode_train")
        builder.build()
        self.assertFalse(
            any(
                name == "torch" or name.startswith("torch.")
                for name in set(sys.modules) - before
            )
        )


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.args = argparse.Namespace(
            test_id="dispatch_smoke_8p",
            source_sha="a" * 40,
            tree_sha="b" * 40,
            lease_id="lease-1",
            task_marker="task-1",
            fencing_token=3,
            node_count=1,
            devices_per_node=8,
            node_rank=0,
            master_addr="192.0.2.2",
            master_port=16000,
        )
        self.env = patch.dict(
            os.environ,
            {
                "LITE_CI_JOB_ID": "task-1",
                "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "LITE_CI_MASTER_PORT": "16000",
                "LITE_CI_HCCL_IF_BASE_PORT": "20000",
                "LITE_CI_HCCL_HOST_PORT_RANGE": "24000-24063",
                "LITE_CI_HCCL_NPU_PORT_RANGE": "28000-28063",
                "NODE_IPS": "192.0.2.2",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            },
            clear=True,
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.case, self.profile = runner.catalog_case(self.args.test_id)
        self.identity = runner.identities(self.args, self.case)

    def payload(self, name=None):
        path = self.source / (name or self.profile["payload"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise RuntimeError('must never execute on controller')\n")
        return path

    def test_launch_command_uses_trusted_supervisor_and_fixed_leased_topology(self):
        self.payload()
        command = runner.command(
            self.args, self.case, self.profile, self.source, self.output
        )
        self.assertIn(str(ROOT / "smoke_rank.py"), command)
        self.assertEqual(command[command.index("--nproc-per-node") + 1], "8")
        self.assertEqual(command[command.index("--master-port") + 1], "16000")
        self.assertIn(str(self.source / self.profile["payload"]), command)
        self.assertNotIn(
            "tests.integration_tests.tools.lite_actions.entrypoint", " ".join(command)
        )

    def test_os_never_enters_npu_runner(self):
        with self.assertRaisesRegex(ValueError, "OS probes"):
            runner.catalog_case("worker_os_blue")

    def test_wrong_rank_topology_or_marker_fails(self):
        for field, value in (
            ("node_count", 2),
            ("devices_per_node", 4),
            ("node_rank", 2),
            ("task_marker", "other"),
            ("fencing_token", 0),
            ("source_sha", "main"),
        ):
            args = copy.copy(self.args)
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.identities(args, self.case)

    def test_platform_credentials_rejected_not_forwarded(self):
        for name in runner.FORBIDDEN_ENV:
            with patch.dict(os.environ, {name: "fixture-not-a-real-secret"}):
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "credential"),
                ):
                    runner.identities(self.args, self.case)

    def test_partial_or_invalid_port_lease_and_duplicate_devices_are_rejected(self):
        for field, value in (
            ("LITE_CI_HCCL_IF_BASE_PORT", "0"),
            ("LITE_CI_HCCL_HOST_PORT_RANGE", "1-2"),
            ("LITE_CI_MASTER_PORT", "16001"),
            ("ASCEND_RT_VISIBLE_DEVICES", "0,0,2,3,4,5,6,7"),
        ):
            with patch.dict(os.environ, {field: value}):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.identities(self.args, self.case)

    def test_required_source_symlink_and_escape_rejected(self):
        path = self.payload()
        target = self.root / "outside.py"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            runner.source_path(self.source, self.profile["payload"])
        with self.assertRaises(ValueError):
            runner.source_path(self.source, "../../outside.py")

    def test_trusted_recipe_applies_no_platform_token_and_original_five_steps(self):
        case, profile = runner.catalog_case("dsv4_flash_a3_8p_adamw")
        self.payload(profile["train_script"])
        args = copy.copy(self.args)
        args.test_id = case["test_id"]
        command = runner.command(args, case, profile, self.source, self.output)
        self.assertEqual(command[0], "bash")
        self.assertEqual(command[command.index("--training.steps") + 1], "5")
        environment = runner.environment(
            args, profile, self.identity, self.source, self.output
        )
        self.assertEqual(environment["TRAIN_FILE"], "_lite_gitcode_train")
        self.assertEqual(
            environment["OPTIMIZER_OVERRIDES"],
            "torchtitan_npu.override.common.optimizer.virtual",
        )
        self.assertFalse(set(runner.FORBIDDEN_ENV) & set(environment))

    def test_training_shim_replaces_source_rendezvous_with_owned_port(self):
        shim = load("torchrun_lease")
        env = runner.environment(
            self.args, self.profile, self.identity, self.source, self.output
        )
        command = shim.launch_args(
            [
                "--nproc_per_node=8",
                "--rdzv_backend",
                "c10d",
                "--rdzv_endpoint=localhost:0",
                "-m",
                "_lite_gitcode_train",
                "--module",
                "torchtitan_npu.models.deepseek_v4",
            ],
            env,
        )
        self.assertEqual(command[command.index("--master-port") + 1], "16000")
        self.assertNotIn("localhost:0", command)
        self.assertEqual(command[command.index("--nnodes") + 1], "1")
        self.assertEqual(command[command.index("-m", 3) + 1], "_lite_gitcode_train")
        with self.assertRaisesRegex(ValueError, "bypassed"):
            shim.launch_args(["-m", "untrusted_other_module"], env)

    def test_success_requires_matching_actual_hccl_and_source_identity(self):
        self.payload()
        self.output.mkdir()
        evidence = {
            **self.identity,
            "status": "PASS",
            "all_reduce_sum": 28,
            "payload_path": str(self.source / self.profile["payload"]),
        }
        path = self.output / "smoke-rank0.json"
        path.write_text(json.dumps(evidence))
        runner.verify(
            self.args, self.case, self.profile, self.source, self.output, self.identity
        )
        for key, value in (
            ("all_reduce_sum", 120),
            ("source_sha", "c" * 40),
            ("fencing_token", 4),
        ):
            path.write_text(json.dumps({**evidence, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.verify(
                    self.args,
                    self.case,
                    self.profile,
                    self.source,
                    self.output,
                    self.identity,
                )


if __name__ == "__main__":
    unittest.main()
