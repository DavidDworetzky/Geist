import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from app.models.database.database import SessionLocal
from app.models.database.mcp_oauth import McpOAuthConnection
from app.models.database.mcp_server import get_mcp_server
from app.security.operator import (
    ALL_OPERATOR_CAPABILITIES,
    OperatorPrincipal,
    get_operator_principal,
)
from app.services import mcp_oauth
from app.services.mcp_client import McpClientManager, McpError
from app.services.mcp_oauth_providers import GMAIL_ENDPOINT, OAuthError
from app.services.mcp_oauth_store import read_credentials, write_credentials
from app.services.mcp_tool_source import config_from_model
from tests.api.test_mcp_api import mcp_client  # noqa: F401


@pytest.fixture
def oauth(mcp_client, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setenv("GEIST_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("GEIST_GOOGLE_OAUTH_CLIENT_ID", "test-client")
    monkeypatch.setenv("GEIST_GOOGLE_OAUTH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("GEIST_PUBLIC_URL", "http://127.0.0.1")
    monkeypatch.delenv("GEIST_MCP_OAUTH_PROVIDERS", raising=False)
    server = mcp_client.post(
        "/api/v1/mcp/servers",
        json={
            "name": "gmail",
            "transport": "http",
            "url": GMAIL_ENDPOINT,
        },
    ).json()
    return mcp_client, server["mcp_server_id"]


def begin(client, server_id):
    response = client.post(
        f"/api/v1/mcp/servers/{server_id}/oauth/start", json={"provider": "google"}
    )
    assert response.status_code == 200, response.text
    params = parse_qs(urlsplit(response.json()["authorization_url"]).query)
    return params, response


def callback(client, params, **kwargs):
    return client.get(
        "/api/v1/mcp/oauth/callback",
        params={
            "state": params["state"][0],
            "code": "test-code",
            **kwargs,
        },
        headers={"Authorization": ""},
        follow_redirects=False,
    )


def credentials(server_id):
    with SessionLocal() as session:
        reference = session.get(McpOAuthConnection, server_id).credential_id
    return reference, read_credentials(reference)


def token_server(monkeypatch, responses=None):
    calls = []
    responses = list(
        responses
        or [
            {
                "access_token": "account-access",
                "refresh_token": "account-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        ]
    )

    def handler(request):
        calls.append(request)
        if request.url.path == "/revoke":
            return httpx.Response(200)
        return httpx.Response(200, json=responses.pop(0))

    real_client = httpx.Client
    monkeypatch.setattr(
        mcp_oauth.httpx,
        "Client",
        lambda **kwargs: real_client(
            **kwargs,
            transport=httpx.MockTransport(handler),
        ),
    )
    return calls


def test_google_consent_pkce_callback_and_private_storage(oauth, monkeypatch, tmp_path):
    client, server_id = oauth
    calls = token_server(monkeypatch)
    params, started = begin(client, server_id)
    assert params["access_type"] == ["offline"]
    assert params["prompt"] == ["consent"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["redirect_uri"] == ["http://127.0.0.1/api/v1/mcp/oauth/callback"]
    assert "HttpOnly" in started.headers["set-cookie"]
    assert "SameSite=lax" in started.headers["set-cookie"]
    assert "test-client-secret" not in started.text
    reference, pending = credentials(server_id)
    verifier = pending["pending"]["verifier"]
    assert (
        params["code_challenge"][0]
        == base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    response = callback(client, params)
    assert response.status_code == 303
    assert response.headers["location"] == "/tools?mcp_oauth=connected"
    assert response.headers["cache-control"] == "no-store"
    assert parse_qs(calls[0].content.decode())["code_verifier"] == [verifier]
    assert "pending" not in read_credentials(reference)
    assert callback(client, params).status_code == 400
    assert len(calls) == 1
    account = client.get(f"/api/v1/mcp/servers/{server_id}/oauth")
    assert account.json()["status"] == "connected"
    saved = client.get(f"/api/v1/mcp/servers/{server_id}").json()
    assert saved["headers"] == {}
    assert saved["oauth_configured"] is True
    assert saved["enabled"] is False
    assert "account-access" not in account.text
    assert "account-refresh" not in account.text
    credential_file = tmp_path / "runtime" / "mcp-credentials" / f"{reference}.json"
    assert credential_file.stat().st_mode & 0o777 == 0o600
    assert credential_file.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("tamper", ["cookie", "state", "expired", "denied"])
def test_callback_rejects_invalid_or_denied_authorization(oauth, monkeypatch, tamper):
    client, server_id = oauth
    calls = token_server(monkeypatch)
    params, _ = begin(client, server_id)
    if tamper == "cookie":
        client.cookies.clear()
    elif tamper == "state":
        params["state"] = [f"{server_id}." + "x" * 43]
    elif tamper == "expired":
        reference, data = credentials(server_id)
        data["pending"]["expires_at"] = 0
        write_credentials(reference, data)
    response = callback(
        client, params, **({"error": "access_denied"} if tamper == "denied" else {})
    )
    assert response.status_code == 400
    assert not calls
    assert "account-access" not in response.text
    if tamper == "denied":
        assert "cancelled" in response.text
        assert "pending" not in credentials(server_id)[1]


def test_oauth_routes_are_workspace_scoped(oauth):
    client, server_id = oauth
    client.app.dependency_overrides[get_operator_principal] = lambda: OperatorPrincipal(
        subject="other",
        authentication_method="test",
        workspace_id=2,
        is_loopback=True,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )
    prefix = f"/api/v1/mcp/servers/{server_id}/oauth"
    assert client.get(prefix).status_code == 404
    assert client.post(prefix + "/start", json={"provider": "google"}).status_code == 404
    assert client.delete(prefix).status_code == 404


def test_wrong_origin_missing_configuration_and_unregistered_endpoint(oauth, monkeypatch):
    client, server_id = oauth
    prefix = f"/api/v1/mcp/servers/{server_id}/oauth"
    response = client.post(
        prefix + "/start", json={"provider": "google"}, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 409
    monkeypatch.delenv("GEIST_GOOGLE_OAUTH_CLIENT_SECRET")
    assert client.get(prefix).json()["providers"][0]["ready"] is False
    assert client.post(prefix + "/start", json={"provider": "google"}).status_code == 409
    monkeypatch.setenv("GEIST_GOOGLE_OAUTH_CLIENT_SECRET", "test-client-secret")
    client.put(f"/api/v1/mcp/servers/{server_id}", json={"url": "https://untrusted.example/mcp"})
    assert client.post(prefix + "/start", json={"provider": "google"}).status_code == 409


def test_manually_supplied_header_and_endpoint_changes_are_rejected(oauth):
    client, server_id = oauth
    prefix = f"/api/v1/mcp/servers/{server_id}"
    client.put(prefix, json={"headers": {"authorization": "Bearer manually-configured"}})
    assert client.post(prefix + "/oauth/start", json={"provider": "google"}).status_code == 409
    client.put(prefix, json={"headers": {}})
    begin(client, server_id)
    assert client.put(prefix, json={"url": "https://untrusted.example/mcp"}).status_code == 409
    assert (
        client.put(prefix, json={"headers": {"AUTHORIZATION": "Bearer another"}}).status_code == 409
    )
    assert client.put(prefix, json={"timeout_seconds": 40}).status_code == 200


def test_refresh_rotation_is_serialized_and_disconnect_invalidates_cached_config(
    oauth, monkeypatch
):
    client, server_id = oauth
    calls = token_server(
        monkeypatch,
        [
            {
                "access_token": "first",
                "refresh_token": "refresh-one",
                "token_type": "Bearer",
                "expires_in": 1,
            },
            {
                "access_token": "second",
                "refresh_token": "refresh-two",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        ],
    )
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 303
    config = config_from_model(get_mcp_server(server_id))
    assert not config.headers
    with ThreadPoolExecutor(max_workers=2) as pool:
        tokens = list(
            pool.map(lambda _: mcp_oauth.access_token(server_id, 1, GMAIL_ENDPOINT), range(2))
        )
    assert tokens == ["second", "second"]
    assert len(calls) == 2
    assert credentials(server_id)[1]["refresh_token"] == "refresh-two"
    with pytest.raises(OAuthError):
        mcp_oauth.access_token(server_id, 2, GMAIL_ENDPOINT)
    with pytest.raises(OAuthError):
        mcp_oauth.access_token(server_id, 1, "https://evil.example")
    reference, _ = credentials(server_id)
    client.put(f"/api/v1/mcp/servers/{server_id}", json={"enabled": True})
    response = client.delete(f"/api/v1/mcp/servers/{server_id}/oauth")
    assert response.json()["revocation_complete"] is True
    assert get_mcp_server(server_id).enabled is False
    assert read_credentials(reference) == {}
    with pytest.raises(McpError, match="disconnected"):
        McpClientManager().list_tools(config)


def test_revoked_refresh_is_reported_as_reconnect_required(oauth, monkeypatch):
    client, server_id = oauth
    token_server(
        monkeypatch,
        [
            {
                "access_token": "first",
                "refresh_token": "refresh-one",
                "token_type": "Bearer",
                "expires_in": 1,
            },
            {"error": "invalid_grant", "error_description": "sensitive-provider-details"},
        ],
    )
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 303
    with pytest.raises(OAuthError, match="revoked"):
        mcp_oauth.access_token(server_id, 1, GMAIL_ENDPOINT)
    assert (
        client.get(f"/api/v1/mcp/servers/{server_id}/oauth").json()["status"]
        == "reconnect_required"
    )
    assert "refresh_token" not in credentials(server_id)[1]


def test_new_account_cannot_reuse_previous_refresh_token(oauth, monkeypatch):
    client, server_id = oauth
    token_server(
        monkeypatch,
        [
            {
                "access_token": "first",
                "refresh_token": "account-one",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            {"access_token": "second", "token_type": "Bearer", "expires_in": 3600},
        ],
    )
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 303
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 400
    assert credentials(server_id)[1]["access_token"] == "first"


def test_custom_registered_provider_uses_bound_resource_and_pkce(oauth, monkeypatch):
    client, server_id = oauth
    monkeypatch.setenv("CUSTOM_CLIENT", "registered-client")
    monkeypatch.setenv(
        "GEIST_MCP_OAUTH_PROVIDERS",
        json.dumps(
            [
                {
                    "id": "work-mail",
                    "label": "Work Mail",
                    "endpoint": "https://mail.example/mcp",
                    "authorization_url": "https://auth.example/authorize",
                    "token_url": "https://auth.example/token",
                    "client_id_env": "CUSTOM_CLIENT",
                    "scopes": ["mail:read"],
                    "resource": "https://mail.example/mcp",
                }
            ]
        ),
    )
    client.put(f"/api/v1/mcp/servers/{server_id}", json={"url": "https://mail.example/mcp"})
    response = client.post(
        f"/api/v1/mcp/servers/{server_id}/oauth/start", json={"provider": "work-mail"}
    )
    params = parse_qs(urlsplit(response.json()["authorization_url"]).query)
    assert params["resource"] == ["https://mail.example/mcp"]
    assert params["code_challenge_method"] == ["S256"]
    assert "client_secret" not in params


def test_changed_registration_cannot_receive_existing_refresh_token(oauth, monkeypatch):
    client, server_id = oauth
    calls = token_server(monkeypatch)
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 303
    original = mcp_oauth.get_provider
    monkeypatch.setattr(
        mcp_oauth,
        "get_provider",
        lambda key: original(key).model_copy(
            update={"token_url": "https://new-provider.example/token"},
        ),
    )
    with pytest.raises(OAuthError, match="configuration changed"):
        mcp_oauth.access_token(server_id, 1, GMAIL_ENDPOINT)
    assert len(calls) == 1


def test_delete_removes_oauth_credentials(oauth, monkeypatch):
    client, server_id = oauth
    token_server(monkeypatch)
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 303
    reference, _ = credentials(server_id)
    assert client.delete(f"/api/v1/mcp/servers/{server_id}").status_code == 204
    assert read_credentials(reference) == {}
    with SessionLocal() as session:
        assert session.get(McpOAuthConnection, server_id) is None


@pytest.mark.parametrize(
    "token_response",
    [
        {
            "access_token": "secret",
            "refresh_token": "refresh",
            "token_type": "Bearer",
            "scope": "unrelated",
        },
        {
            "access_token": "bad\r\nHeader: injected",
            "refresh_token": "refresh",
            "token_type": "Bearer",
        },
        {
            "access_token": "secret",
            "refresh_token": "refresh",
            "token_type": "Bearer",
            "expires_in": -1,
        },
        {"error": "invalid_client", "error_description": "test-client-secret"},
    ],
)
def test_invalid_token_responses_do_not_connect_or_expose_provider_details(
    oauth, monkeypatch, token_response
):
    client, server_id = oauth
    token_server(monkeypatch, [token_response])
    params, _ = begin(client, server_id)
    response = callback(client, params)
    assert response.status_code == 400
    assert "test-client-secret" not in response.text
    assert "access_token" not in credentials(server_id)[1]


def test_github_preset_and_nonexpiring_oauth_tokens(oauth, monkeypatch):
    client, server_id = oauth
    monkeypatch.setenv("GEIST_GITHUB_OAUTH_CLIENT_ID", "github-client")
    monkeypatch.setenv("GEIST_GITHUB_OAUTH_CLIENT_SECRET", "github-secret")
    client.put(
        f"/api/v1/mcp/servers/{server_id}", json={"url": "https://api.githubcopilot.com/mcp/"}
    )
    response = client.post(
        f"/api/v1/mcp/servers/{server_id}/oauth/start", json={"provider": "github"}
    )
    assert response.status_code == 200
    params = parse_qs(urlsplit(response.json()["authorization_url"]).query)
    assert params["scope"] == ["repo"]
    token_server(
        monkeypatch, [{"access_token": "github-account", "token_type": "bearer", "scope": "repo"}]
    )
    assert callback(client, params).status_code == 303
    assert (
        mcp_oauth.access_token(server_id, 1, "https://api.githubcopilot.com/mcp/")
        == "github-account"
    )


def test_https_callback_cookie_is_secure(oauth, monkeypatch):
    client, server_id = oauth
    monkeypatch.setenv("GEIST_PUBLIC_URL", "https://geist.example")
    _, response = begin(client, server_id)
    assert "Secure" in response.headers["set-cookie"]


def test_missing_credential_file_allows_disconnect(oauth):
    client, server_id = oauth
    begin(client, server_id)
    reference, _ = credentials(server_id)
    from app.services.mcp_oauth_store import delete_credentials

    delete_credentials(reference)
    response = client.delete(f"/api/v1/mcp/servers/{server_id}/oauth")
    assert response.status_code == 200
    assert get_mcp_server(server_id).oauth_configured is False


def test_mcp_dispatch_resolves_refreshed_tokens_without_putting_them_in_tool_config(
    oauth, monkeypatch
):
    client, server_id = oauth
    token_server(monkeypatch)
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 303
    config = config_from_model(get_mcp_server(server_id))
    assert "account-access" not in repr(config)
    import app.services.mcp_client as transport

    seen = []

    class FakeConnection:
        def __init__(self, config, **kwargs):
            self.config = config
            seen.append(config)

        def list_tools(self, **kwargs):
            return []

        def close(self):
            pass

    monkeypatch.setattr(transport, "McpConnection", FakeConnection)
    assert McpClientManager().list_tools(config) == []
    assert seen[0].headers["Authorization"] == "Bearer account-access"
    assert config.headers == {}


def test_changed_provider_cannot_relabel_old_credentials_during_reconnect(oauth, monkeypatch):
    client, server_id = oauth
    token_server(monkeypatch)
    params, _ = begin(client, server_id)
    assert callback(client, params).status_code == 303
    original = mcp_oauth.get_provider
    monkeypatch.setattr(
        mcp_oauth,
        "get_provider",
        lambda key: original(key).model_copy(
            update={"token_url": "https://new-provider.example/token"},
        ),
    )
    response = client.post(
        f"/api/v1/mcp/servers/{server_id}/oauth/start", json={"provider": "google"}
    )
    assert response.status_code == 409
    assert "disconnect" in response.json()["detail"]
