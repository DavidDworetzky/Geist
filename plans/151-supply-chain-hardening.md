# Supply Chain Hardening Plan

## Goal

Reduce dependency supply-chain risk for frontend npm installs and backend Python/conda installs by making safe install behavior explicit and enforceable in local hooks.

## Scope

- Pin frontend direct dependencies to exact versions already resolved in `package-lock.json`.
- Use `npm ci --ignore-scripts` for frontend Docker installs.
- Add a local pre-commit dependency policy check.
- Update install instructions so contributors avoid ad hoc dependency resolution.

## Policy

- Frontend installs must use the committed lockfile with lifecycle scripts disabled.
- New frontend dependencies must be added with exact versions and reviewed in both `package.json` and `package-lock.json`.
- Backend pip dependencies in environment files must be exact `==` pins.
- Conda dependencies in environment files must include exact version/build pins.
- Dependency changes should be audited before merge.

## Validation

- Run `python scripts/verify_dependency_policy.py`.
- Run `pre-commit run dependency supply-chain policy --all-files` after installing hooks.
- For frontend dependency changes, run `npm audit --package-lock-only` from `client/geist`.
- For backend dependency changes, audit the pinned pip section of the environment file.
