# Compute selection and current main integration

Preserve compute-device selection while adopting main's workspace identity,
runtime discovery, model-load lifecycle and dedicated Models UI. Resolve both
schema branches with a merge revision. Adopt only schema-validated legacy
settings gaps, backing up SQLite data before repairs; do not migrate user data
while testing. Preserve all migration data effects for workspace identity.

Verify compute settings, native runner and startup contracts, database upgrades,
frontend build/tests, isolated Docker startup, and native MLX-mode validation.
Keep permissions, approval resume, agentic harness and per-chat workspaces in
their higher stack layers.

Discovery runs in one daemon worker per service with bounded caller waits,
including the initiating caller. A wedged OS probe cannot accumulate workers or
persist a provisional CPU choice. Warm placeholders preserve real errors; cold
placeholders respect runner overrides and perform path checks outside the lock.
Refresh remains coalesced/rate limited; UI wording explicitly describes a latest
available snapshot instead of claiming a fresh enumeration.

Validation: 977 Docker tests passed, 7 skipped; 165 focused native MLX-mode
tests passed. Docker and native isolated chat routes returned HTTP200. Browser
interaction is blocked by the locked Mac; no real Vulkan device is available,
and no second large MLX model was loaded alongside the user's active server.

Normal commit hooks found baseline issues: the cached mypy hook reports Any in
unchanged whisper_adapter.py (project-venv focused mypy passes), Bandit reports
the existing subprocess import and main's unchanged 0.0.0.0 bind, and full UI
eslint reports 84 existing test-style violations. Targeted production ESLint,
Ruff, project mypy and native Windows-contract mypy are checked separately;
these three baseline hooks are skipped for this integration commit only.
