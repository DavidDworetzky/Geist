# Open PR Stack Repair

## Scope

Repair the security and correctness findings identified during the August 2026
review of the open Geist pull-request stack without broadening the feature scope
of the affected branches.

## Work

1. Require explicit approval for dynamically discovered MCP tools and cover the
   default permission posture with focused tests.
2. Validate optional Agent Plugin MCP fields so malformed local manifests are
   skipped instead of breaking discovery.
3. Claim scheduled routines atomically and execute them for their owning user.
4. Encrypt provider credentials at rest while retaining a migration path for
   existing plaintext rows.
5. Keep sibling Alembic revisions linear by rebasing and updating
   `down_revision` immediately before each migration-bearing PR is merged.

## Verification

- Run the focused backend tests for each changed service and contract.
- Run Ruff and Mypy on changed Python files where the branch baseline permits.
- Run Docker, UI, and native smoke checks only where the changed behavior makes
  them applicable, and report environmental blockers precisely.
- Recheck each pushed head and its CI status before recommending merge order.
