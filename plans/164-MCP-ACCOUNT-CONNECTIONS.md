# MCP account connections

## Goal

Let users connect supported MCP providers from setup, starting with Google
OAuth for Gmail and Google Workspace. Provide a guided Proton Mail Bridge
setup for Proton accounts and use each other provider's supported auth method.

## Existing work and implementation base

Implementation is based on `main` at `800a28f`, on the new branch
`codex/mcp-account-connections`, as requested. Main contains the existing
workspace-scoped MCP server model/client, operator authentication, and Tools
catalogue. This change extends that stack directly.

## Implementation

1. Extend provider metadata with supported connection methods and setup
   requirements. Distinguish authorization to a remote MCP server from a
   server's own upstream Google/Microsoft account authorization. Do not send
   provider tokens to arbitrary user-entered MCP endpoints.
2. Add workspace-scoped connection metadata and private pending-authorization records, migration,
   request/response contracts, and a service for authorization start, callback,
   status, refresh, and disconnect. Use short-lived, single-use state bound
   to the initiating operator-authorized browser and server, PKCE where supported, fixed
   registered callback URLs, and vetted provider endpoints/scopes. Persist
   credentials using an appropriate protected backend store; never return
   token values in connection responses, browser storage, or logs.
3. Implement Google account consent and offline access with the configured
   OAuth application. Verify Gmail MCP availability and its required scopes
   against Google's documentation before enabling the catalogue endpoint.
   Show missing deployment configuration or account eligibility accurately.
4. Pass valid access tokens to the matching MCP connection, refresh expired
   credentials, and surface reconnect states after revocation or refresh
   failure. Disconnect deletes stored authorization and performs provider
   revocation where supported. Preserve existing email tool activation gates.
5. Add Connect with Google, connected status, reconnect, and disconnect UI.
   Present consent cancellation and setup failures without losing the saved
   MCP configuration. Extend the same flow only to other catalogue providers
   whose authorization contracts have been verified. Keep credential-based
   and unauthenticated setup for providers that need them.
6. For Proton, guide users to install and sign in to Proton Mail Bridge,
   configure a vetted local IMAP/SMTP MCP server with Bridge-issued credentials,
   and test discovery. Document the paid Mail plan requirement, Bridge's
   separate password, and local-versus-container network reachability.
   Do not label this as Proton OAuth or request the Proton account password.
   Do not install Bridge or another package without explicit user approval.

## Validation

- Backend contract/service tests for consent start/callback, state expiry and
  replay, browser/user isolation, denied consent, token redaction, endpoint
  binding, refresh failure/rotation, and disconnect cleanup.
- UI tests for Google connection, cancellation, reconnect, missing OAuth
  application configuration, and Proton Bridge setup requirements.
- Follow the Geist test-loop skill for relevant Docker startup, logs,
  localhost:3000 response, settings, and browser smoke tests. Test MCP tool
  registry integration if modified. Native MLX testing is needed only if
  implementation changes native inference paths.
- Live account consent and Bridge verification require configured services
  and user-controlled sign-in; record any unavailable checks explicitly.

## References checked

- Google OAuth web-server authorization:
  https://developers.google.com/identity/protocols/oauth2/web-server
- Proton Bridge, paid Mail requirement, and generated credentials:
  https://proton.me/mail/bridge
- Proton IMAP/SMTP integration:
  https://proton.me/support/imap-smtp-and-pop3-setup

## Status

Implemented Google and GitHub OAuth presets, a provider registry for other
operator-configured HTTP MCP endpoints, callback/state/PKCE validation,
backend token refresh and disconnect, a private credential store, migration and
legacy database adoption, OAuth UI controls, and the Proton Bridge checklist.
No dependencies were added. The private file store supports macOS/Linux;
Windows OAuth storage remains unsupported. Provider registrations are explicit,
with no dynamic discovery or dynamic client registration in this release.

Validation covers mocked provider exchanges and runtime setup with no account
credentials. Live Google/GitHub consent and Proton Bridge mailbox access require
operator app registration or an installed signed-in Bridge and MCP server and
were not exercised. No native MLX inference paths changed.


## Validation evidence

- Backend: **82 passed**, covering OAuth exchanges, PKCE/state/cookie binding,
  replay and expiry, workspace ownership, refresh serialization and rotation,
  credential redaction, endpoint binding, cached MCP dispatch, deletion,
  GitHub and custom registrations, migration/adoption, and operator middleware.
- Frontend: **19 passed**, including save-and-connect failures, reconnect,
  disconnect, provider labels, Proton instructions, and OAuth return navigation.
- Targeted Ruff and mypy checks pass. Frontend webpack compiled successfully
  with no type issues. Existing dependency deprecation warnings remain.
- Docker smoke startup succeeded on isolated host ports 3535/5535 using existing
  images. The first attempts hit Docker disk capacity; tmpfs storage resolved
  the test infrastructure failure without deleting other tasks' data.
- Curl returned HTTP 200 for the isolated frontend and existing localhost:3000
  frontend. Only port 3535 served this change; the existing port 3000 stack was
  left untouched. An unauthenticated invalid callback returned 400 and an
  unauthenticated account status request returned 401. The backend access log
  omitted the callback query, including the dummy authorization code.
- Browser: Gmail setup preserved the disabled server when app configuration was
  missing; Proton instructions rendered correctly; no browser console errors.
  Compact Mode saved, persisted across reload, and was restored. Chat opened;
  generation was not exercised because the default model could not be downloaded
  into the disposable runtime's limited storage.
- Live provider consent and Proton mailbox access were not run: no test OAuth
  app credentials or signed-in Bridge/MCP mail service were configured. Native
  `make run MLX_BACKEND=1` was not applicable; no inference paths changed.

Commands used for the passing Docker test suites (dotenv loading is disabled so
validation does not read local secret files):

```sh
docker run --rm --network none --tmpfs /tmp:mode=1777,size=536870912 --entrypoint python -v /Users/daviddworetzky/Documents/repos/Geist-wt/geist-5:/opt/geist:ro -w /opt/geist -e GEIST_DATABASE_PROVIDER=sqlite -e SQLITE_DATABASE_PATH=/tmp/geist-oauth-tests.sqlite3 -e GEIST_DATA_DIR=/tmp/geist-oauth-tests -e GEIST_JOB_WORKER_ENABLED=false -e PYTHONDONTWRITEBYTECODE=1 geist-tools-mcp-pr-backend -c 'import dotenv; dotenv.load_dotenv = lambda *args, **kwargs: False; import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/api/test_mcp_oauth.py", "tests/api/test_mcp_api.py", "tests/services/test_mcp_client.py", "tests/services/test_mcp_tool_source.py", "tests/database/test_database_upgrade.py", "tests/security/test_operator_middleware.py"]))'

docker run --rm --network none --tmpfs /tmp:mode=1777,size=268435456 --entrypoint npm -v /Users/daviddworetzky/Documents/repos/Geist-wt/geist-5/client/geist/src:/app/src:ro -w /app -e CI=true geist-tools-mcp-pr-frontend test -- --watchAll=false --runInBand --runTestsByPath src/Components/__tests__/McpServersSection.test.tsx src/Components/__tests__/McpOAuthAccount.test.tsx src/Tools.test.tsx

docker compose --env-file /dev/null -p geist-oauth-smoke -f /tmp/geist-oauth-smoke-compose.yaml up -d --force-recreate
docker compose --env-file /dev/null -p geist-oauth-smoke -f /tmp/geist-oauth-smoke-compose.yaml logs --tail=15 backend frontend
curl -fsS -o /dev/null -w 'Isolated frontend: HTTP %{http_code}\n' http://localhost:3535
curl -fsS -o /dev/null -w 'Existing localhost:3000 frontend: HTTP %{http_code}\n' http://localhost:3000
```

The smoke compose file was a temporary validation harness using existing images,
read-only source mounts, a disposable database, a test-only operator token, and
no OAuth credentials. Its containers were removed after verification.

### Publication checks

- Re-ran all 19 frontend tests after lint cleanup: passed. React test `act`
  warnings and existing Babel dependency warnings remain.
- Changed frontend files pass ESLint with `--max-warnings=0` in the existing
  Docker image. Full frontend ESLint reports 81 diagnostics versus 83 on `main`,
  with no new diagnostics (compared by file, rule, and message).
- Ruff lint/format and mypy `--follow-imports=silent` pass for all 14 changed
  application Python files. The full mypy hook reproduces the same pre-existing
  `adapters/whisper_adapter.py:38` `no-any-return` error on `main` in its cached
  hook environment.
- Bandit finds no new issues against an archived `main` baseline. Its three
  existing findings are B104 in `app/main.py` and B404/B603 in the MCP subprocess
  client. Two explicit B106 suppressions identify public OAuth token endpoint
  URLs that the heuristic otherwise mistakes for passwords.
- Installed the Git hook using the existing virtual environment; all hook
  environments were already cached. No packages or dependencies were installed.
  Commit-time mypy and Bandit are skipped because of the verified baseline
  findings; frontend-eslint is replaced by the passing changed-file Docker
  command because local frontend dependencies are unavailable. Other hooks,
  including staged secret scanning, run normally.

Additional check commands (file lists are the changed application/frontend files):

```sh
PATH="$PWD/.venv/bin:$PATH" SKIP=frontend-eslint .venv/bin/pre-commit run
.venv/bin/ruff check <changed-python-files>
.venv/bin/ruff format --check <changed-python-files>
/Users/daviddworetzky/.cache/pre-commit/repohww_ontr/py_env-default/bin/mypy --follow-imports=silent <changed-python-files>
/Users/daviddworetzky/.cache/pre-commit/repopzzhupy9/py_env-python3/bin/bandit -c pyproject.toml -b /tmp/geist-oauth-main-bandit.json <changed-python-files>
docker run --rm --network none --tmpfs /tmp:mode=1777,size=268435456 --entrypoint /app/node_modules/.bin/eslint -v /Users/daviddworetzky/Documents/repos/Geist-wt/geist-5/client/geist/src:/app/src:ro -w /app geist-tools-mcp-pr-frontend <changed-frontend-files> --max-warnings=0
```
