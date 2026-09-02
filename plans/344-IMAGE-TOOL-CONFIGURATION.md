# Image tool configuration metadata

## Scope

- Keep image credentials environment-managed until Geist has a dedicated
  secrets-management design.
- Extend the tool catalogue response with a redacted, read-only summary for
  `image.generate`: provider, resolved model, resolved base URL, API-key status,
  and the relevant environment variable names.
- Render that summary in a configuration modal without exposing or accepting
  secret values.
- Remove unfinished built-in tools from the catalogue and enable read-only
  Markdown listing and reading by default.

## Verification

- Backend registry/catalogue tests confirm visibility, defaults, and redaction.
- Frontend tests cover opening and closing the image configuration modal.
- Run focused lint, typecheck, production build, and browser smoke checks.
