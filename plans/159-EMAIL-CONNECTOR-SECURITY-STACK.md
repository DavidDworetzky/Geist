# Email connector security stack

## Goal

Build a three-PR stack that makes external MCP servers usable for curated email
accounts while treating every external server and every email payload as
untrusted by default. Security inspection is enforced at connector boundaries
without propagating a general taint state across chat turns.

## Product decisions

- External MCP servers default to untrusted, including Microsoft and Google.
- Email content remains untrusted even when an operator promotes its server.
- Inspect tool metadata, outbound calls, and inbound results.
- The security inspector mirrors the active chat model configuration, has no
  tools, receives the complete boundary payload, and fails closed.
- Failures include model/transport errors, timeouts, invalid verdict schemas,
  instruction-pattern matches, and an `inspector_compromised` verdict.
- The safe result preserves useful text while stripping HTML and omitting
  active content, remote images, and attachment bodies.
- Security inspection layers on top of existing approval behavior; it never
  grants approval. Every email capability is inspected.
- Writes use idempotency keys, recipient controls, and rate limits.
- Policies are stored for the default Geist user. The Settings security surface
  stays small: enablement, application-policy toggles, security model status,
  and a single append-only audit log.

## PR 1: MCP base

- Rebase PR #303 on current `main` and resolve the Settings integration.
- Keep stdio and streamable HTTP MCP transport, persisted server configuration,
  tool discovery, and the unified registry.
- Redact stored secrets from API responses and enforce default-user ownership
  for read, update, delete, test, and enabled-server discovery.
- Add working approval/resume semantics for dynamically discovered tools.
- Keep all external MCP servers disabled and untrusted by default.

## PR 2: Curated email connectors

- Add connector records and account onboarding for personal Gmail, managed
  Google Workspace, Outlook/Microsoft 365, and paid Proton accounts through
  Proton Mail Bridge.
- Support provider-specific local/remote MCP launch settings without installing
  third-party packages automatically.
- Add secure credential references, connection testing, capability policies,
  recipient allowlists, and write-rate limits.
- Keep curated connectors feature-gated until the security layer is present.
- Add backend API/service tests and frontend connector-settings tests.

## PR 3: Security boundary

- Add a boundary inspector for MCP tool metadata, outbound calls, and inbound
  results.
- Use the active chat model configuration in a tool-free inspector request.
- Require a strict structured verdict and a small deterministic instruction
  pattern scanner.
- Return `allow`, `block`, or `inspector_compromised`; any internal ambiguity
  blocks the operation.
- Produce lightly normalized safe results and store blocked originals only for
  direct user reveal outside model context.
- Add per-user enforcement policies, a compact Security Settings tab, and a
  single append-only audit log with metadata and hashes rather than message
  bodies.
- Add hostile tool-description, argument, email-body, HTML, and failure-mode
  tests.

## Validation

- Focused backend tests for MCP, approval/idempotency, connectors, and security.
- Ruff, formatting, and targeted mypy checks.
- Frontend component tests and production build.
- Docker backend/frontend startup, logs, and `curl http://localhost:3000`.
- Browser smoke of MCP, connector, and Security Settings tabs.
- Native MLX smoke because the security inspector can mirror a local model;
  report environment blockers explicitly.

## Merge order

1. MCP base (`main` target; existing PR #303).
2. Curated email connectors (target the MCP base branch).
3. Security boundary and Settings (target the connector branch).
