# PR18 bootstrap diagnostic cleanup and PR-trigger safety plan — 2026-11-03

## Current source and evidence boundary

- PR: https://github.com/cann-framework/torchtitan-npu/pull/18
- Current bootstrap script SHA-256: 2a468c43fa38a69d6e037965c6f4b003039261e5f3307396d0d5f3e94367e1ca
- SDK commit/archive pins remain unchanged.
- The one-job Hosted diagnostic is temporary evidence tooling, not a permanent product workflow.
- It has no secrets, no artifact upload/download, and does not enter CLI business logic.
- The formal six-job manual-forward workflow remains separate and keeps the strict LITE_GITCODE_CAPTURE_SCRIPT_SHA256 binding.

## Delete-after-evidence procedure

After exactly one manual run of TorchTitan SDK Bootstrap Diagnostic on the intended GitCode personal master, preserve:
1. GitCode run ID, attempt, workflow revision/control SHA, job ID, start/end timestamps and terminal conclusion.
2. The complete job log captured through the authorized web UI, with credentials/headers absent.
3. The single bounded diagnostic line (MANUAL_SDK_BOOTSTRAP_OK or MANUAL_SDK_BOOTSTRAP_FAILED) including phase, fixed error code and elapsed milliseconds.
4. The downloaded archive digest/size result as emitted by the fixed script; do not add arbitrary response bodies, URLs, headers, or environment dumps.
5. The job/workflow source SHA and the generated script SHA above.

Then, in a follow-up source commit after the evidence is durably recorded:
1. Delete .gitcode/workflows/torchtitan_bootstrap_diagnostic.yml.
2. Remove render_diagnostic(), DIAGNOSTIC_OUTPUT, and its write/print call from .gitcode/scripts/build_hosted_manual_workflow.py.
3. Regenerate .gitcode/workflows/torchtitan_forward_manual.yml from the single bootstrap template.
4. Verify the six formal job bootstrap bodies remain byte-identical and each retains the exact generated SHA binding.
5. Re-run the eight offline bootstrap failure fixtures and the model workflow contract tests; verify YAML and git diff --check.
6. Commit the deletion as a distinct cleanup commit; do not leave the diagnostic-only environment switch or workflow in the product tree.

Do not delete the diagnostic before the one-job evidence is captured. Do not substitute a full manual Forward or a Blue/NPU run for this Hosted-only diagnostic.

## Why pull requests were repeatedly dispatching hardware

Both .github/workflows/lite_actions_dispatcher_smoke.yml and
.github/workflows/nightly_fulltests.yml subscribe to pull_request on
master. Their reusable workflow's prepare job substitutes default_cases
whenever inputs.test_cases is empty. A pull-request event has no
workflow_dispatch input value, so it selects the default hardware matrix:
- Dispatcher-Smoke: worker OS probe + 8P + 16P.
- Nightly: 8P + 16P training suites.

The reusable case matrix then runs wait_result.py and can consume Blue
resources. An empty CI_RESULT_AUTHOR causes result-wait failure, but it is
not a pre-dispatch guard and therefore did not prevent the Blue listener from
accepting the request. This is the common cause of PR-open/update hardware
side effects.

PR18 now guards the reusable case job with
github.event_name != 'pull_request': PRs retain immutable plan expansion and
static validation but do not dispatch NPU cases. Explicit workflow_dispatch
continues to run the selected/default matrix. The caller result jobs accept
both success and skipped, so a deliberately skipped PR matrix is not
misreported as a failed hardware test. Regression assertions cover both
workflow callers and the common case guard.

## Merge and release boundary

The repository's active default ruleset has no required-status-check rule
and no required approving review count; the branch-protection endpoint reports
Branch not protected. Do not label the currently red checks as required.
Use the actual current PR checks, mergeability and ruleset response; do not
ApproveReview or bypass protection.

Before a formal Forward, bind the final merged control SHA and the exact
bootstrap digest above together in the release candidate and Blue configuration.
The Hosted-only diagnostic does not need a Blue release/config/process cutover.
