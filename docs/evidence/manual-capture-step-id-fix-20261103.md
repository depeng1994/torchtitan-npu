# Manual capture step identity contract correction — 2026-11-03

## Failure and evidence boundary

Run #10: GitCode `bac91c70278e4141a21366ef667b67e5`; manual_forward job `f6fbc6eb51f74b65993234a13dc78b80`. PM's existing web observation (00:28:17–00:28:31 UTC) reports overall failure after 2m38s, manual_capture 12s, manual_forward 5s, and:
`[2026-11-03 08:26:34.337+08:00] MANUAL_FORWARD_FAILED:CAPTURE_STEP_IDENTITY_MISMATCH`.

The exact sanitized bac91c job/step metadata projection is not present in the local workspace and no new run-status GET was performed. Therefore the exact bac91c field value remains unconfirmed.

## Independently verified fixed-contract mismatch

- Current generated model workflow used `id: manual_capture` for the capture step.
- Pinned SDK `be1a37d67a977dbee9c7bb1ef13f445a438e20ec` requires `CAPTURE_STEP = "1"` in `ci_core/gitcode_manual/common.py`.
- SDK `dispatch.py` rejects a capture step whose `identifier` differs from that fixed constant, raising `CAPTURE_STEP_IDENTITY_MISMATCH`.
- Saved actual platform metadata fixture from prior run b472 reports the user shell step's API `identifier: "1"`, with `name: "Capture manual request"`, `task: "official_shell"`, `sequence: 1`, `job_run_id` equal to capture job id, and COMPLETED status.
- G's independent snapshot review confirmed the job key, name, shell, working directory and capture script digest align; its sole static mismatch is generated workflow step ID `manual_capture` vs SDK expected `1`.

This is a strong static cause consistent with the runtime reason, but not claimed as an independently observed exact bac91c response field.

## Minimal correction

Changed only model generator capture step ID to `1`; the job identifier remains `manual_capture`. The generated workflow is regenerated from the builder. No SDK identity checks were weakened or removed. Name, shell, working directory, generated script SHA-256, task, sequence, job binding, status, command digest, and retained evidence re-read remain strict.

At this historical step-ID correction, the capture bootstrap digest was
`8eb518a2fc3708de3f8e41015fb952d68f3b767754e699eb1697c33df5ab442d`.
That digest is superseded by the later reviewed phase-diagnostic bootstrap:
`2a468c43fa38a69d6e037965c6f4b003039261e5f3307396d0d5f3e94367e1ca`.
The latter is the current generated digest for all six formal jobs and is the
value carried by the current PR18 workflow generator. The historical test and
step-ID evidence above remains valid for its original revision only.

## Regression evidence

New model test:
`tests/unit_tests/test_gitcode_manual_capture_step_contract.py`

It compares generated workflow step ID/name with the saved real platform metadata fixture, verifies task/sequence/job_run_id/status, checks the separate job key, and checks shell/cwd/script digest.

Fixture:
`tests/fixtures/lite_actions/manual_capture_platform_steps_b472.json`
This is real sanitized b472 platform metadata, not bac91c.

Command:
`python3 -m unittest tests.unit_tests.test_gitcode_manual_capture_step_contract tests.unit_tests.test_lite_actions_dispatch_smoke -v`

Result: 12/12 PASS in 0.166s. A preceding pytest invocation could not load repository conftest because this CPU-only Remote environment lacks torch; direct unittest then ran the target modules successfully.

Hashes:
- builder: `fd32ae5b54fa8a203259c7a59553c9616dd38b714cf064a43548e62b696b8056`
- generated workflow: `606c5d30ecab70e13dfaf813624d52728243faa0828559075d56f43152b44436`
- test: `386e0e709ab3d2ee03a3be54eaff43abcfdd073d88fe2700be9c49a92c1e93ff`
- fixture: `ba3e33a133a93227d1beecfb5961b07c7096010523dc95530c14e9729e2676a9`
- test log: `cab57d5801eb4fcdd2b852f29f3c67e87c7ef6c11d6e54a8c53a9dce5bd531a1`

No source commit, push, PR, release, deployment, or new dispatch has yet been made for this correction.
