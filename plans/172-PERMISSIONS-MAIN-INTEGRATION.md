# Permissions and main integration

Preserve main's authenticated workspace identity, intent routing and invocation-
bound approval resume while adding the existing permission modes and allowlist.
Remove obsolete numeric-user settings routes. Preserve permission fields when
reconstructing tool context after approval. Join migration branches without
rewriting published migrations, and adopt legacy permission columns only after
schema validation and SQLite backup.

Verify permission modes, allowlists, approval resume, cancellation, settings API
and reset, migrations, frontend tests/build, and isolated runtime startup.

Validation: 389 Docker tests and 217 frontend tests pass; production build and
eight-file project mypy pass. Isolated Docker UI/settings routes return HTTP200,
with default permission mode and an empty allowlist. Browser is blocked by the
locked Mac. Normal commit hooks pass except full frontend ESLint's89 existing
test-style violations; targeted production ESLint is checked separately and
only that baseline hook is skipped for this merge commit.

Review5576249648: runtime-discovered definitions cannot redeem name-only standing
grants; the registry sets eligibility rather than trusting source claims. Expose
that eligibility in the catalog and allow UI revocation, not creation, of inert
saved dynamic grants. Explicit auto-approve remains an operator-wide trust choice;
the higher execution layer's non-waivable host approvals remain stronger. Retain
invocation-specific fingerprint checks. Guard injected-loader failures by requiring
approval, improve route/migration tests, align permission typing and recovery copy.

Follow-up:393 Docker tests pass,221 frontend tests/build pass; five-file project
mypy and production component ESLint pass. Includes versioned compute-parent
upgrade and permissions-only legacy adoption, injected-loader failure, changed
dynamic definitions, and disabled-new/revocable-old grant UI coverage.
# Review5576410109 follow-up

Auto-approve copy explicitly covers runtime MCP/plugin definitions changing
without notice. Require-approval copy explains grant eligibility, and an old
ineligible grant is labeled as stored but not honored, with revocation available.
The normalizer now returns a TypedDict so PermissionMode is checked at its
construction boundary. A catalog contract test pins false eligibility for a
source-provided tool. Source eligibility deliberately remains fail-closed for
first-party adapter sources too; registrars do not gain an extra trust override.
In #307, mandatory per-call approval remains ahead of both global auto-approve
and standing-grant checks. This ordering is a stack contract, not an optional UI rule.
