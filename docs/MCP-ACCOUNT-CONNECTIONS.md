# Connect MCP accounts

In **Tools → MCP Servers**, configure a Gmail, Google Workspace Mail, or GitHub
profile and choose **Save and connect**. After approving the provider’s consent
screen, Geist returns to Tools. Test the server, then enable it when ready.
Signing in does not enable a new server or bypass the existing tool approval
policy. Existing manual bearer headers remain supported; remove the manual
Authorization header before switching a server to OAuth.

Saved HTTP servers also show Connect, Reconnect, and Disconnect controls when
the backend has an OAuth registration for their exact endpoint. Disconnect
removes local credentials and disables the server. For Google it also attempts
provider revocation. If revocation fails or the provider has no configured
revocation endpoint, the UI links the remaining action to the provider’s account
settings. Deleting an OAuth server also cleans up its saved credentials.

## Google deployment setup

Configure the backend process with:

- `GEIST_GOOGLE_OAUTH_CLIENT_ID`
- `GEIST_GOOGLE_OAUTH_CLIENT_SECRET`
- `GEIST_PUBLIC_URL`: the browser-facing origin, such as
  `http://localhost:3000` (the default) or `https://geist.example.com`.

Register a **Web application** OAuth client with the exact callback URL:
`GEIST_PUBLIC_URL` followed by `/api/v1/mcp/oauth/callback`. Open Geist using that
same origin when signing in. Do not place the client secret in frontend variables.

Enable the Gmail API and Gmail MCP API, configure the Google consent screen,
and enroll the project/account in the Workspace Developer Preview program if
required. Workspace administrators may need to approve the application. The
built-in profile requests the documented `gmail.readonly` and `gmail.compose`
scopes, and offline access so Geist can refresh tokens without another sign-in.
Google’s consent screen describes the permissions actually granted by those
scopes; OAuth authorization alone does not establish MCP service eligibility.

References: [Gmail MCP configuration](https://developers.google.com/workspace/gmail/api/guides/configure-mcp-server)
and [Google OAuth](https://developers.google.com/identity/protocols/oauth2/web-server).

## GitHub deployment setup

Register a GitHub OAuth application using the same callback URL and configure
`GEIST_GITHUB_OAUTH_CLIENT_ID` and `GEIST_GITHUB_OAUTH_CLIENT_SECRET` on the backend.
The preset binds the grant to `https://api.githubcopilot.com/mcp/` and requests
`repo`, which includes private repository access and repository write permissions.
Review the consent screen and your organization’s application policies before
connecting. Geist’s MCP tool calls still require approval. Token expiration and
rotation are supported when the OAuth application enables expiring tokens.
Remove the application grant in GitHub settings to revoke access after disconnect.

Reference: [GitHub OAuth application flow](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps).

## Other OAuth MCP servers, including Outlook

An operator can supply additional explicit registrations in the backend process
variable `GEIST_MCP_OAUTH_PROVIDERS` as a JSON array. Example for an organization’s
mail MCP service (replace all example values with the service’s documented registration):

```json
[
  {
    "id": "work-mail",
    "label": "Work Mail",
    "endpoint": "https://mail.example.com/mcp",
    "authorization_url": "https://auth.example.com/authorize",
    "token_url": "https://auth.example.com/token",
    "revocation_url": "https://auth.example.com/revoke",
    "client_id_env": "WORK_MAIL_CLIENT_ID",
    "client_secret_env": "WORK_MAIL_CLIENT_SECRET",
    "scopes": ["mail:read"],
    "resource": "https://mail.example.com/mcp"
  }
]
```

Configure the referenced client credentials on the backend. All service URLs
must use HTTPS. Registrations support authorization-code flow with S256 PKCE,
JSON bearer token responses, refresh tokens, optional resource indicators, and
optional revocation. Public clients can omit `client_secret_env`; confidential
clients use `client_secret_post`. The registry is operator-managed: Geist does
not discover authorization servers or dynamically register clients in this release.

For Outlook/Microsoft 365, use the authentication contract of the specific MCP
service, including the correct token audience. A Graph access token must not be
forwarded to an arbitrary third-party MCP endpoint. The MCP service may handle
its upstream Microsoft account consent separately. There is no universal Outlook
MCP endpoint in the catalogue. Context7 and local Playwright can continue using
their existing setup methods; a login is not required for every MCP server.

## Proton Mail

Proton’s documented integration uses **Proton Mail Bridge**, not a Google-style
OAuth grant. Select **Configure Proton Mail** for a step-by-step checklist:

1. Install Bridge on the machine running the IMAP/SMTP MCP server and sign in
   inside Bridge. A paid plan that includes Proton Mail is required.
2. Configure your installed and vetted MCP server using Bridge’s host, ports,
   username, and generated password. Use the variable names documented by that
   server; these are not standardized across IMAP/SMTP MCP implementations.
3. Enter its command and arguments in Geist, save, and test tool discovery.
   Bridge must stay running. Enable the server when ready.

The Bridge-generated password differs from your Proton account password. Do not
enter the Proton account password into Geist. Geist does not install Bridge or
an MCP server. In Docker, container localhost is not host localhost: running the
MCP server alongside Bridge is the straightforward arrangement. Keep Bridge
local; do not expose its mail ports publicly to work around container networking.
An MCP discovery test establishes that the MCP server responds; whether it also
checks Bridge authentication depends on the selected server implementation.

References: [Proton Bridge](https://proton.me/mail/bridge) and
[Proton IMAP/SMTP setup](https://proton.me/support/imap-smtp-and-pop3-setup).

## Storage and troubleshooting

OAuth access/refresh tokens and pending verifiers are stored outside the database
and tool workspace under the runtime data directory’s `mcp-credentials` folder.
On macOS/Linux the folder is mode 0700 and files are mode 0600, owned by the
runtime account. File writes are atomic and per-server file locks serialize
refresh and disconnect across workers sharing the same runtime directory. These
files are **not encrypted at rest**; protect the runtime volume and its backups.
Windows OAuth credential storage is not supported yet. Existing non-OAuth MCP
configuration continues to work on Windows.

The database stores only provider/endpoint/client identifiers and an opaque
credential-file reference. Keep the database and private runtime directory on
the same deployment. Restoring the database without credentials requires
reconnecting accounts. Do not put this directory inside a model-accessible
workspace or grant an MCP process access to it.

Authorization state expires after ten minutes, is single use, and is bound to an
HttpOnly SameSite browser cookie. A cancelled, expired, or invalid callback shows
a safe error and a link back to Tools. Account tokens never enter API responses,
manual MCP headers, browser storage, or tool arguments. The backend suppresses
callback query strings in its access log; configure any upstream proxy to omit
query strings for the callback too.

If provider credentials, endpoints, or scopes change, disconnect and reconnect.
If a refresh grant is revoked, the saved account shows Reconnect required after
its next refresh attempt. Network failures preserve the refresh token so the next
request can retry. A successful consent screen followed by a failed MCP test can
mean missing provider service access, administrator approval, or insufficient
permissions; it does not necessarily mean the OAuth exchange failed.
