# Pitchblend Model-Catalog Review Plan

## Goal

Allow Pitchblend to approve low- or medium-complexity model-catalog changes,
including hosted OpenRouter additions such as PR #330, while retaining the
existing deterministic safety gates and semantic review.

## Scope

- Recognize catalog-only changes through a narrow file allowlist that requires
  `agents/model_catalog.py` and permits only its focused tests and model docs.
- Add an explicit approval-matrix route for catalog bug fixes, features, and
  refactors when the classifier reports no migration or security change and a
  regression test is present.
- Ignore Pitchblend's own pending approval-gate status while still rejecting
  every other pending or unsuccessful commit status.
- Keep size, mergeability, blocked-path, patch-availability, test-file, and
  check-run gates unchanged.

## Validation

- Cover the exact PR #330 file set and expected classifier shape.
- Reject near misses that change any non-allowlisted implementation file or do
  not modify `agents/model_catalog.py`.
- Verify the approval gate does not deadlock on its own pending status and that
  unrelated failing statuses still block review.
- Run the focused Pitchblend unit tests and repository formatting checks.
