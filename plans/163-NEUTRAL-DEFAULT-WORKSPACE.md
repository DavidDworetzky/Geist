# Neutral Default Workspace and Operator Identity Plan

## Goal

Remove the personal name and email currently used as Geist's implicit local
identity without changing the durable `geist_user.user_id` that owns existing
chats, memories, files, settings, workflows, jobs, and tool configuration.
Keep local workspace ownership separate from request authentication so the
Pitchblend wrapper and future remote deployments can supply an operator
principal without turning profile metadata into credentials.

## Stack and compatibility

1. Geist adds a stable `workspace_key` to `geist_user`, migrates the generated
   legacy default row in place, and bootstraps a neutral default workspace.
2. Pitchblend resolves the new `workspace_key` while retaining a bounded
   compatibility path for the current Geist schema revision.
3. Geist adds the request-scoped operator principal above the workspace branch.
4. The MCP/tool-registry PR is rebased above the operator principal.
5. The Agent Plugins PR remains stacked above the MCP/tool-registry PR.

The Pitchblend adapter must recognize the legacy memory schema, this migration,
and the MCP migration revision used by the top of the stack. It must never
infer the default workspace from a person's email on a new Geist revision.

## Geist implementation

- Add nullable, uniquely indexed `geist_user.workspace_key`.
- Reserve `workspace_key="default"` for the local workspace.
- Change default lookup to use the workspace key rather than email.
- Preserve the existing default row's `user_id` and every foreign-key
  relationship during migration.
- Clear the legacy email and empty password. Replace the generated name and
  username only when they still match the legacy generated values, preserving
  intentional local customization.
- Add an idempotent post-upgrade bootstrap because fresh databases are created
  from SQLAlchemy metadata and stamped at head without running migration data
  operations.
- Remove the duplicate default-user insertion implementation and keep one
  transactional bootstrap path.
- Keep the physical password column for the current Pitchblend compatibility
  window, but stop treating it as authentication state.

## Operator implementation

- Add an immutable request-scoped `OperatorPrincipal` with subject,
  authentication method, workspace ID, and explicit capabilities.
- Enforce principal authentication across HTTP and WebSocket routes, exempting
  only health/readiness and API documentation endpoints.
- Keep token-file, wrapper-token, loopback, and future node assertions behind a
  pluggable authenticator interface.
- Treat loopback-only standalone Geist as a trusted local operator.
- Give Docker/native development proxies a generated local operator token;
  keep it in a private file consumed only by the backend and server-side proxy.
- Accept a per-launch wrapper credential for Pitchblend-managed Geist and map
  it to the neutral default workspace without using Pitchblend OIDC tokens.
- Require authenticated operator capabilities on MCP CRUD/test routes.
- Keep remote stdio MCP disabled by deployment policy even for an authenticated
  operator unless a future design explicitly adds a separate opt-in.
- Preserve Pitchblend's existing customer OIDC login for licensing and sync;
  it is not the local machine-administrator credential.
- Reserve controller node, target node, audience, expiry, and credential ID
  fields so future Tailscale transport assertions can remain node-scoped and
  short-lived without persisting cloud users in Geist.

## Documentation

Expand the tool-registry feature documentation with the trust boundaries among
the browser guest, Geist server, Pitchblend wrapper, Pitchblend Cloud identity,
operator principal, local workspace ownership, MCP transports, approvals, and
remote deployment restrictions.

## Tests

- Fresh database creates exactly one neutral default workspace.
- Legacy generated row migrates in place and retains its `user_id`.
- Existing related rows retain their owner IDs.
- Customized local display metadata survives migration.
- Bootstrap is idempotent and rejects ambiguous duplicate workspace keys.
- Pitchblend resolves legacy email-backed and new workspace-key-backed schemas.
- Pitchblend login/logout and identity token boundaries remain unchanged.
- MCP routes reject unauthenticated remote callers and resolve ownership from
  the authenticated principal.
- Loopback and Pitchblend wrapper sessions remain usable without a second
  customer login.
- Existing MCP, tool approval, frontend, Docker, and browser smoke checks pass.

## Review remediation

- Recognize unversioned databases created before the workspace migration,
  including the existing pre-local-artifact compatibility shape.
- Make the workspace migration establish a default workspace without relying on
  the application startup bootstrap.
- Keep database failures outside the compatibility-token validation boundary so
  storage outages surface as server errors rather than authentication failures.
- Keep request-time workspace resolution read-only after startup establishes
  the default workspace invariant.
- Adopt a sole existing owner row when its legacy email was customized, and
  fail explicitly instead of guessing when multiple unkeyed owners are
  present.
- Cover and safely adopt an unversioned branch database that already has
  `workspace_key` but is missing only `default_local_artifact_id`.
