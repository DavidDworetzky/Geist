# Workspace Principal and Multi-Node Readiness

## Objective

Separate three identities that have historically been conflated:

1. Pitchblend's cloud account principal identifies the signed-in human.
2. Geist's request principal authenticates a client process and authorizes capabilities.
3. Geist's workspace identity owns local chats, files, memory, settings, workflows, jobs, and tool configuration.

Geist remains a single-workspace local server. It does not maintain email/password user accounts. The design must remain extensible to Pitchblend controlling multiple Geist workstations over an authenticated transport such as Tailscale without sharing one credential across nodes.

## Invariants

- Preserve the deployed `geist_user` table and `user_id` database columns in this stack so existing SQLite databases and Pitchblend memory sync remain compatible.
- Preserve the default workspace row's numeric primary key and every existing foreign key.
- Do not map a Pitchblend OIDC account into the Geist database.
- Require an authenticated request principal for Geist API access, with only narrowly defined health/readiness endpoints exempt.
- Keep capability authorization separate from workspace ownership.
- Keep Geist bound to loopback in standalone and Pitchblend-managed modes. A future Tailscale proxy terminates remote transport and forwards only authenticated requests to loopback.
- Do not share a static operator token between workstations.

## Current Stack Allocation

### Geist #323: canonical workspace semantics

- Introduce `WorkspaceModel` and `get_default_workspace()` as the canonical application vocabulary.
- Keep the physical `geist_user.user_id` schema as a compatibility detail.
- Remove unused general-purpose user CRUD helpers.
- Update ordinary workspace-scoped routes and services to resolve the default workspace rather than a placeholder current user.
- Remove public user-settings-by-ID routes; the local API exposes only the current workspace settings.
- Retain legacy aliases only where they are necessary for a transition test or an already published integration.
- Document that the personal seed cleanup is a one-time compatibility migration, not authentication.

### Pitchblend #48: workspace-aware local integration

- Rename sync concepts from default user to default workspace while continuing to query the compatible `geist_user.user_id` schema.
- Keep OIDC identity separate from Geist workspace selection.
- Continue injecting the per-launch Geist request token only for the exact managed Geist origin.
- Record the supported Geist schema revision strategy explicitly so a later physical table rename can support old and new schemas during a rollout window.

### Geist #322: request principal boundary

- Rename `OperatorPrincipal.user_id` to `workspace_id`.
- Split authentication from authorization: authenticate requests globally, then require capabilities on privileged routes.
- Exempt only health/readiness and documentation endpoints that must operate before the managed client is ready.
- Keep local token-file and Pitchblend per-launch token authenticators behind one principal-construction interface.
- Include stable subject and authentication-method fields without treating them as a human account.
- Reserve optional controller-node, target-node, audience, expiry, and credential-ID claims for future authenticators; do not persist cloud users locally.

### Geist #303 and #312: workspace-owned tools and plugins

- Use `workspace_id` throughout tool context, approvals, MCP configuration, caches, and route ownership.
- Because MCP tables are not deployed yet, use `workspace_id` as the new physical column while referencing the compatible workspace table primary key.
- Describe the future multi-node rule: remote requests receive only explicitly granted capabilities, and stdio execution remains local-node-only.
- Keep plugin discovery and plugin-provided MCP servers scoped to the target workspace and authenticated request principal.

## Request Flow

### Standalone Geist

1. Startup creates or loads a random local operator token file.
2. The local frontend proxy injects the token server-side.
3. Geist authenticates the request into an operator principal.
4. Geist resolves the singleton workspace and attaches its ID to the principal.
5. Route dependencies authorize the requested capability.

### Pitchblend-managed Geist

1. Pitchblend creates a random token for each Geist process launch.
2. The token is passed only to the child process and the managed Electron session boundary.
3. The browser never receives the token in JavaScript.
4. Geist authenticates the exact managed-origin request and resolves the local workspace.
5. Pitchblend OIDC remains independent and controls cloud features rather than local database identity.

### Future multi-node control

1. Each workstation enrolls as a distinct node with its own key material and stable node ID.
2. Pitchblend Cloud records which account may control which nodes.
3. Tailscale grants restrict which controller nodes may reach tagged Geist nodes and ports.
4. Pitchblend obtains a short-lived authorization scoped to a target node, audience, expiry, and capabilities.
5. Geist validates both the transport/device assertion and application authorization before constructing the request principal.
6. The target Geist node maps the request to its singleton local workspace; it does not create a local human user.

## Deferred Physical Schema Migration

The physical rename is intentionally a later coordinated Geist/Pitchblend release:

- `geist_user` becomes `workspace`.
- `user_id` ownership columns become `workspace_id`.
- `username`, `email`, and `password` are removed.
- `name` becomes an optional workspace display name.
- Pitchblend supports both schema revisions during the compatibility window.
- Released Alembic migrations remain immutable; runtime-only legacy personal lookup code can be removed after the supported upgrade window.

## Security Requirements

- Fail closed when configured token sources disagree, are malformed, or are unavailable.
- Compare bearer material in constant time and never log it.
- Bind tokens or future assertions to one target Geist node; reject cross-node replay.
- Use short expirations and credential identifiers for future remote authorizations.
- Keep direct local server ports loopback-only even when Tailscale is enabled through a local reverse proxy.
- Tailscale reachability is defense in depth, not a replacement for Geist application authorization.
- Privileged capabilities remain explicit and deny by default.

## Test Plan

### Geist #323

- Fresh database creates exactly one neutral workspace.
- Legacy personal seed migrates in place and preserves ownership IDs.
- Existing workspace and unrelated users are not overwritten.
- Chats, files, settings, workflows, jobs, and memory remain attached to the preserved ID.
- User-settings-by-ID routes no longer exist.

### Pitchblend #48

- Workspace lookup uses `workspace_key` for the new revision and the legacy lookup only for supported older revisions.
- OIDC login/logout does not modify Geist workspace identity.
- Packaged, attach, restart, and child-exit token-boundary tests pass on macOS, Linux, and Windows validation paths.

### Geist #322

- Requests without a valid principal fail across protected API routes.
- Health/readiness exemptions remain reachable.
- Standalone proxy and Pitchblend-managed requests succeed.
- Mismatched token sources fail with 503.
- Capability-protected routes return 403 for an authenticated but insufficient principal.
- Principal fields contain workspace and node-ready request context without cloud-account persistence.

### Geist #303/#312

- MCP CRUD, approvals, caches, and tool execution cannot cross workspace IDs.
- Remote/wrapper principals cannot start local stdio MCP processes.
- Plugin refresh and plugin MCP configuration require the correct workspace-scoped capability.
- Existing SSRF, redirect, subprocess environment, timeout, cancellation, and symlink-boundary tests continue to pass.

### Runtime QA

- Run focused backend and frontend tests for every modified branch.
- Run Ruff, format checks, and mypy on changed Python files.
- Run Docker startup/log/curl and browser chat/settings smoke tests for the final Geist stack.
- Run native `make run MLX_BACKEND=1` because startup/auth proxy behavior affects native local-model use.
- Verify Pitchblend CI's exact Geist integration SHA after the operator branch is rewritten.

## Merge and Rollout

1. Merge Geist #323.
2. Merge Pitchblend #48.
3. Rebase Geist #322 onto Geist main, refresh Pitchblend's temporary integration SHA, and merge #322 after CI.
4. Remove the temporary Pitchblend integration SHA once #322 is present in Geist main.
5. Rebase and merge Geist #303.
6. Rebase and merge Geist #312.
7. Schedule the physical workspace-table rename as a separate coordinated compatibility stack.
