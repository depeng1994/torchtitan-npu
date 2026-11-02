"""Upload command contracts: reject ambiguous input before any subprocess."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "hosted_upload_contract", ROOT / ".github/scripts/lite_actions/gitcode/hosted.py"
)
hosted = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hosted)


class UploadCommandContractTests(unittest.TestCase):
    def test_upload_command_requires_explicit_argument_vector(self):
        invalid = (
            "node",
            b"node",
            {"node": True},
            {"node"},
            iter(["node"]),
            None,
            1,
            True,
            [],
            (),
            [""],
            ["node", 1],
            ["node\0"],
        )
        with patch.object(hosted.subprocess, "run") as run:
            for command in invalid:
                with (
                    self.subTest(type=type(command).__name__),
                    self.assertRaisesRegex(hosted.HostedError, "TRUSTED_UPLOAD_COMMAND_REQUIRED"),
                ):
                    hosted.ArtifactUploader(command)
            run.assert_not_called()

    def test_list_and_tuple_argv_are_accepted_without_invocation(self):
        with patch.object(hosted.subprocess, "run") as run:
            for command in (["node", "/trusted/upload.cjs"], ("node", "/trusted/upload.cjs")):
                self.assertEqual(hosted.ArtifactUploader(command).command, tuple(command))
            run.assert_not_called()

    def test_mutating_caller_list_cannot_change_captured_command(self):
        command = ["node", "/trusted/upload.cjs"]
        uploader = hosted.ArtifactUploader(command)
        command[1] = "/untrusted/upload.cjs"
        self.assertEqual(uploader.command, ("node", "/trusted/upload.cjs"))
