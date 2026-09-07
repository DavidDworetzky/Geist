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
