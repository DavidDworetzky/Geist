# Bundled Proton Bridge email MCP

Stack: #315 -> #316 -> this change. Ship a stdlib-only local stdio MCP server,
with operator-only guided Proton setup instead of a third-party command.

## Contracts
- Persist non-secret Bridge settings and a random credential reference on McpServer.
  Store passwords in macOS Keychain or Linux Secret Service; fail clearly if unavailable.
- Bind Bridge connections to literal loopback addresses; require STARTTLS or TLS and
  the exported public Bridge certificate. Never accept a remote endpoint or skip verification.
- Dedicated test/create/update routes accept Bridge credentials; responses never echo
  passwords. Save only after both IMAP and SMTP authentication succeed; start disabled.
- Generate the child command internally. Send credentials over its private stdin during
  initialization, not command arguments, process environment, database, or tool schemas.
- Expose list_folders, search_messages, read_message, send_message. Bound inputs/results,
  use UID + UIDVALIDITY for stable reads, BODY.PEEK to avoid marking mail read.
- Retain #316 inspection and per-call approvals. Only the bundled known read tools are
  classified read-only. Sending is subject to recipient and write policies; maintain a
  durable send-attempt ledger so ambiguous SMTP completion is not automatically retried.
- Update/delete invalidate running sessions; generic endpoints cannot rewrite managed
  launch/credential configuration. Delete removes the associated OS credential.

## Verification
Focused protocol/API/policy/migration/frontend tests, real stdio subprocess discovery,
local TLS IMAP/SMTP fixture integration, Docker when available, native MLX and browser
smoke. Real Proton login requires user-supplied Bridge credentials and remains a manual
acceptance test. No Python/Node/system packages added or installed.
