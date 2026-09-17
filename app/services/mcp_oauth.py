"""Browser OAuth and token lifecycle for explicitly registered MCP providers."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import uuid
from urllib.parse import urlencode

import httpx

from app.models.database.database import SessionLocal
from app.models.database.mcp_oauth import McpOAuthConnection
from app.models.database.mcp_server import McpServer, McpServerModel
from app.models.mcp_oauth import OAuthProviderInfo, OAuthStatus
from app.services.mcp_oauth_providers import (
    OAuthError,
    OAuthProvider,
    callback_uri,
    get_provider,
    providers,
)
from app.services.mcp_oauth_store import (
    CredentialStoreError,
    connection_lock,
    delete_credentials,
    read_credentials,
    write_credentials,
)


STATE_LIFETIME = 600


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _provider_revision(provider: OAuthProvider) -> str:
    return _digest(provider.model_dump_json())


def _validate_binding(
    connection: McpOAuthConnection, server: McpServer | McpServerModel
) -> OAuthProvider:
    provider = get_provider(connection.provider)
    if (
        server.transport != "http"
        or server.url != connection.endpoint
        or server.url != provider.endpoint
        or provider.client_credentials()["client_id"] != connection.client_id
    ):
        raise OAuthError("OAuth configuration changed; disconnect and reconnect the account")
    return provider


def status(server: McpServerModel) -> OAuthStatus:
    result = OAuthStatus(redirect_uri=callback_uri())
    for provider in providers():
        if server.transport != "http" or provider.endpoint != server.url:
            continue
        message = None
        try:
            provider.client_credentials()
        except OAuthError as error:
            message = str(error)
        result.providers.append(
            OAuthProviderInfo(
                id=provider.id,
                label=provider.label,
                scopes=provider.scopes,
                ready=message is None,
                setup_message=message,
            )
        )
    if not server.oauth_configured:
        return result
    with connection_lock(server.mcp_server_id), SessionLocal() as session:
        connection = session.get(McpOAuthConnection, server.mcp_server_id)
        if connection is None:
            return result
        result.provider = connection.provider
        credentials = read_credentials(connection.credential_id)
        result.status = "reconnect_required"
        try:
            provider = _validate_binding(connection, server)
            if credentials.get("provider_revision") != _provider_revision(provider):
                return result
        except OAuthError:
            return result
        pending = credentials.get("pending") or {}
        if pending.get("expires_at", 0) > time.time():
            result.status = "pending"
        elif credentials.get("access_token") and (
            credentials.get("expires_at", 0) > time.time() or credentials.get("refresh_token")
        ):
            result.status = "connected"
        return result


def start(server: McpServerModel, provider_id: str, browser_token: str) -> str:
    provider = get_provider(provider_id)
    client = provider.client_credentials()
    redirect_uri = callback_uri()
    if server.transport != "http" or server.url != provider.endpoint:
        raise OAuthError("This provider can authorize only its registered MCP endpoint")
    if any(name.lower() == "authorization" for name in server.headers):
        raise OAuthError("Remove the manual Authorization header before connecting with OAuth")
    state = f"{server.mcp_server_id}.{secrets.token_urlsafe(32)}"
    verifier = secrets.token_urlsafe(64)
    with connection_lock(server.mcp_server_id), SessionLocal() as session:
        # Recheck the persisted endpoint under the same lock used for account changes.
        current = session.get(McpServer, server.mcp_server_id)
        if (
            current is None
            or current.workspace_id != server.workspace_id
            or current.url != server.url
            or current.transport != "http"
            or any(key.lower() == "authorization" for key in (current.headers or {}))
        ):
            raise OAuthError("MCP server changed; reload its configuration")
        connection = session.get(McpOAuthConnection, server.mcp_server_id)
        if connection is None:
            connection = McpOAuthConnection(
                mcp_server_id=server.mcp_server_id,
                provider=provider.id,
                endpoint=server.url,
                client_id=client["client_id"],
                credential_id=uuid.uuid4().hex,
            )
            session.add(connection)
        elif connection.provider != provider.id or connection.client_id != client["client_id"]:
            raise OAuthError("Disconnect the previous OAuth account before changing providers")
        credentials = read_credentials(connection.credential_id)
        if credentials and credentials.get("provider_revision") != _provider_revision(provider):
            raise OAuthError("OAuth provider configuration changed; disconnect before reconnecting")
        credentials["provider_revision"] = _provider_revision(provider)
        credentials["pending"] = {
            "state_hash": _digest(state),
            "browser_hash": _digest(browser_token),
            "verifier": verifier,
            "expires_at": time.time() + STATE_LIFETIME,
            "redirect_uri": redirect_uri,
        }
        current.enabled = False
        write_credentials(connection.credential_id, credentials)
        session.commit()
    params = {
        "response_type": "code",
        "client_id": client["client_id"],
        "redirect_uri": redirect_uri,
        "scope": " ".join(provider.scopes),
        "state": state,
        "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode(),
        "code_challenge_method": "S256",
    }
    if provider.id == "google":
        params.update(access_type="offline", prompt="consent")
    if provider.resource:
        params["resource"] = provider.resource
    return provider.authorization_url + "?" + urlencode(params)


def _token_request(
    provider: OAuthProvider, data: dict[str, str], timeout_seconds: float = 20
) -> dict:
    payload = {**provider.client_credentials(), **data}
    if provider.resource:
        payload["resource"] = provider.resource
    try:
        with (
            httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client,
            client.stream(
                "POST", provider.token_url, data=payload, headers={"Accept": "application/json"}
            ) as response,
        ):
            body = b""
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > 65536:
                    raise OAuthError("OAuth provider returned an oversized response")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError()
            if result.get("error") == "invalid_grant":
                raise OAuthGrantExpiredError(
                    "Account authorization expired or was revoked; reconnect the account"
                )
            if response.status_code != 200 or result.get("error"):
                raise OAuthError(
                    "OAuth provider rejected the request; check the app registration and consent"
                )
            return result
    except (httpx.HTTPError, ValueError) as error:
        raise OAuthError("Could not complete the OAuth token exchange; try reconnecting") from error


class OAuthGrantExpiredError(OAuthError):
    pass


def _tokens(result: dict, *, previous_refresh: str | None = None) -> dict:
    access = result.get("access_token")
    refresh = result.get("refresh_token", previous_refresh)
    if (
        not isinstance(access, str)
        or not access
        or len(access) > 8192
        or any(c.isspace() or ord(c) < 32 or ord(c) > 126 for c in access)
        or str(result.get("token_type", "")).lower() != "bearer"
        or (refresh is not None and (not isinstance(refresh, str) or len(refresh) > 8192))
    ):
        raise OAuthError("OAuth provider returned invalid credentials")
    try:
        lifetime = float(result.get("expires_in", 315360000))
        if not 0 < lifetime <= 315360000:
            raise ValueError()
    except (ValueError, TypeError) as error:
        raise OAuthError("OAuth provider returned an invalid token lifetime") from error
    return {"access_token": access, "refresh_token": refresh, "expires_at": time.time() + lifetime}


def finish(server_id: int, state: str, browser_token: str, code: str | None, denied: bool) -> None:
    with connection_lock(server_id), SessionLocal() as session:
        connection = session.get(McpOAuthConnection, server_id)
        server = session.get(McpServer, server_id)
        if connection is None or server is None:
            raise OAuthError("Authorization expired; start the connection again")
        credentials = read_credentials(connection.credential_id)
        pending = credentials.get("pending") or {}
        if (
            not browser_token
            or not secrets.compare_digest(pending.get("state_hash", ""), _digest(state))
            or not secrets.compare_digest(pending.get("browser_hash", ""), _digest(browser_token))
            or pending.get("expires_at", 0) <= time.time()
        ):
            raise OAuthError("Authorization expired or belongs to another browser; start again")
        credentials.pop("pending", None)
        write_credentials(connection.credential_id, credentials)
        if denied:
            raise OAuthError("Sign-in was cancelled or denied. You can try connecting again")
        if not code:
            raise OAuthError("The provider did not return an authorization code")
        provider = _validate_binding(connection, server)
        if credentials.get("provider_revision") != _provider_revision(provider):
            raise OAuthError("OAuth provider configuration changed; start again")
        result = _token_request(
            provider,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": pending["redirect_uri"],
                "code_verifier": pending["verifier"],
            },
        )
        if "scope" in result and not set(provider.scopes).issubset(
            str(result["scope"]).replace(",", " ").split()
        ):
            raise OAuthError(
                "Required permissions were not granted; reconnect and approve the requested scopes"
            )
        tokens = _tokens(result)
        # Never carry a refresh token from the previously connected Google account.
        if provider.id == "google" and not tokens["refresh_token"]:
            raise OAuthError(
                "Google did not grant offline access; remove the app grant and reconnect"
            )
        credentials = {"provider_revision": _provider_revision(provider), **tokens}
        write_credentials(connection.credential_id, credentials)


def access_token(
    server_id: int, workspace_id: int, endpoint: str, timeout_seconds: float = 20
) -> str:
    deadline = time.monotonic() + timeout_seconds
    with connection_lock(server_id, timeout_seconds=timeout_seconds), SessionLocal() as session:
        server = session.get(McpServer, server_id)
        connection = session.get(McpOAuthConnection, server_id)
        if (
            server is None
            or connection is None
            or server.workspace_id != workspace_id
            or server.url != endpoint
        ):
            raise OAuthError("MCP account is disconnected or its endpoint changed")
        provider = _validate_binding(connection, server)
        credentials = read_credentials(connection.credential_id)
        if credentials.get("provider_revision") != _provider_revision(provider):
            raise OAuthError("OAuth configuration changed; reconnect the account")
        if credentials.get("access_token") and credentials.get("expires_at", 0) > time.time() + 60:
            return str(credentials["access_token"])
        refresh = credentials.get("refresh_token")
        if not refresh:
            raise OAuthError("Reconnect the MCP account before using its tools")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OAuthError("OAuth refresh exceeded the MCP request deadline")
        try:
            result = _token_request(
                provider,
                {"grant_type": "refresh_token", "refresh_token": refresh},
                timeout_seconds=min(20, remaining),
            )
        except OAuthGrantExpiredError:
            credentials.pop("access_token", None)
            credentials.pop("refresh_token", None)
            write_credentials(connection.credential_id, credentials)
            raise
        credentials.update(_tokens(result, previous_refresh=refresh))
        write_credentials(connection.credential_id, credentials)
        return str(credentials["access_token"])


def disconnect(server: McpServerModel) -> bool:
    revoked = False
    with connection_lock(server.mcp_server_id), SessionLocal() as session:
        connection = session.get(McpOAuthConnection, server.mcp_server_id)
        current = session.get(McpServer, server.mcp_server_id)
        if current is None or current.workspace_id != server.workspace_id:
            raise OAuthError("MCP server no longer exists")
        current.enabled = False
        if connection is not None:
            try:
                credentials = read_credentials(connection.credential_id)
            except CredentialStoreError:
                credentials = {}
            try:
                provider = _validate_binding(connection, current)
                if credentials.get("provider_revision") != _provider_revision(provider):
                    raise OAuthError("OAuth provider configuration changed")
                token = credentials.get("refresh_token") or credentials.get("access_token")
                if provider.revocation_url and token:
                    with httpx.Client(timeout=10, follow_redirects=False) as client:
                        response = client.post(provider.revocation_url, data={"token": token})
                        revoked = response.status_code == 200
            except (OAuthError, httpx.HTTPError):
                pass
            delete_credentials(connection.credential_id)
            session.delete(connection)
        session.commit()
    return revoked
