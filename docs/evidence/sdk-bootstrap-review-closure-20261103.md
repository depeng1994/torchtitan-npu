# SDK bootstrap review closure — 2026-11-03

Task G identified no P0 and two P1 items. The exact archive network smoke is now covered by the model's existing `test_real_isolated_sdk_archive_bootstrap_executes_aggregate_without_credentials`: it downloads the exact public archive for SDK commit `be1a37d67a977dbee9c7bb1ef13f445a438e20ec`, verifies the pinned archive digest, extracts/imports the published SDK in the actual subprocess bootstrap, runs aggregate PASS without either GitCode or GitHub credential, and verifies current authority/merge gate remain NOT_PROVIDED. It passed in the local model test run.

The archive extraction hardening now binds the exact 58-file `ci_core/` path set in the generator and rejects missing/unexpected files, duplicate normalized paths (including duplicate `ci_core` root), non-normalized paths, unexpected directories, symlinks/hardlinks/devices, file/parent-directory collisions, traversal/absolute paths, and extraction beyond the existing 8 MiB cap. The complete compressed archive remains pinned by exact commit SHA and SHA-256 with the existing 4 MiB download cap; each member is extracted only after the entire inventory preflight succeeds, using O_EXCL/O_NOFOLLOW and private temporary files/directories. No per-file duplicate manifest or new protocol/service/credential is introduced; the exact path set is sufficient because the complete archive digest authenticates all file contents.

Added CPU regression cases cover duplicate normalized member, file/directory prefix collision, and missing/unexpected SDK files. Existing real-archive subprocess smoke covers the published immutable archive. See `tests/unit_tests/tooling/gitcode_hosted/test_manual_sdk_bootstrap.py`.

Release candidate `tt-personal-org-manual-release-6eae06-sdkbe1a37-bootstrap.json` was updated only as a local candidate: SDK SHA remains `be1a37d...`; `manual_capture.script_sha256` is updated to the generated capture bootstrap digest. The live Blue config and process were not changed in this review closure.


## Superseding bootstrap diagnostic revision — 2026-11-03

The generated capture bootstrap digest from the earlier closure was superseded
by PR18's redacted phase/error/elapsed diagnostic bootstrap:
`2a468c43fa38a69d6e037965c6f4b003039261e5f3307396d0d5f3e94367e1ca`. The generator now emits this exact digest into the formal manual-forward
workflow; all six inline formal job scripts are byte-identical. The separate
one-job Hosted diagnostic embeds the same script and exits before CLI business
logic. This does not change the pinned SDK SHA/archive SHA or any live Blue
configuration. The current release/Blue pin must be bound to the final merged
control SHA plus this digest before a formal Forward; the Hosted-only
diagnostic does not require a Blue cutover.
