import importlib

import pytest
from fastapi.testclient import TestClient

from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser
from app.models.database.mcp_server import get_mcp_server
from app.security.operator import (
    ALL_OPERATOR_CAPABILITIES,
    OperatorPrincipal,
    get_operator_principal,
)
from app.services.mcp_client import McpError


@pytest.fixture()
def mcp_client(tmp_path, monkeypatch):
    operator_token = "m" * 43
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'mcp-api.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "false")
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", operator_token)
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN_FILE", raising=False)
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=1,
                workspace_key="default",
                name="Local Workspace",
            )
        )
        session.commit()
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_operator_principal] = lambda: OperatorPrincipal(
        subject="test-operator",
        authentication_method="test",
        workspace_id=1,
        is_loopback=True,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
        headers={"Authorization": f"GeistOperator {operator_token}"},
    ) as client:
        yield client
    app.dependency_overrides.clear()
    Session.remove()
    Base.metadata.drop_all(bind=engine)
    configure_database(original_config)


def _stdio_payload(name: str = "filesystem") -> dict:
    return {
        "name": name,
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        "env": {"HOME": "/tmp"},
    }


def test_mcp_routes_require_operator_authentication(monkeypatch):
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", "a" * 43)
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "false")
    from app.main import create_app

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/mcp/servers")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "GeistOperator"


def test_mcp_server_crud_flow(mcp_client):
    created = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())
    assert created.status_code == 201
    body = created.json()
    server_id = body["mcp_server_id"]
    assert body["name"] == "filesystem"
    assert body["enabled"] is False  # servers start disabled
    assert body["trusted"] is False
    assert body["connector_kind"] == "custom"

    listed = mcp_client.get("/api/v1/mcp/servers")
    assert listed.status_code == 200
    assert [server["name"] for server in listed.json()] == ["filesystem"]

    updated = mcp_client.put(
        f"/api/v1/mcp/servers/{server_id}",
        json={"enabled": True, "timeout_seconds": 45},
    )
    assert updated.status_code == 200
    assert updated.json()["enabled"] is True
    assert updated.json()["timeout_seconds"] == 45

    fetched = mcp_client.get(f"/api/v1/mcp/servers/{server_id}")
    assert fetched.status_code == 200
    assert fetched.json()["enabled"] is True

    deleted = mcp_client.delete(f"/api/v1/mcp/servers/{server_id}")
    assert deleted.status_code == 204
    assert mcp_client.get(f"/api/v1/mcp/servers/{server_id}").status_code == 404
    assert mcp_client.get("/api/v1/mcp/servers").json() == []


def test_mcp_secrets_are_redacted_and_redacted_updates_preserve_values(mcp_client):
    payload = _stdio_payload()
    payload["env"] = {"ACCESS_TOKEN": "secret-token"}
    created = mcp_client.post("/api/v1/mcp/servers", json=payload)
    server_id = created.json()["mcp_server_id"]

    assert created.json()["env"] == {"ACCESS_TOKEN": "<redacted>"}
    listed = mcp_client.get("/api/v1/mcp/servers").json()[0]
    assert listed["env"] == {"ACCESS_TOKEN": "<redacted>"}

    updated = mcp_client.put(
        f"/api/v1/mcp/servers/{server_id}",
        json={"env": listed["env"], "timeout_seconds": 60},
    )
    assert updated.status_code == 200
    assert get_mcp_server(server_id).env == {"ACCESS_TOKEN": "secret-token"}

    unknown_placeholder = mcp_client.put(
        f"/api/v1/mcp/servers/{server_id}",
        json={"env": {"NEW_TOKEN": "<redacted>"}},
    )
    assert unknown_placeholder.status_code == 422


def test_email_connector_profiles_cover_supported_accounts(mcp_client):
    response = mcp_client.get("/api/v1/mcp/email-connectors/profiles")

    assert response.status_code == 200
    assert {profile["kind"] for profile in response.json()} == {
        "gmail",
        "google_workspace",
        "outlook",
        "proton",
    }


def test_email_connector_starts_disabled_untrusted_and_security_gated(mcp_client):
    payload = {
        **_stdio_payload("work-mail"),
        "connector_kind": "google_workspace",
        "account_label": "person@example.com",
        "working_directory": "/opt/vetted-mail-mcp",
        "recipient_allowlist": ["teammate@example.com"],
        "max_writes_per_hour": 8,
        "enabled": True,
        "trusted": True,
    }

    created = mcp_client.post("/api/v1/mcp/email-connectors", json=payload)

    assert created.status_code == 201
    body = created.json()
    assert body["connector_kind"] == "google_workspace"
    assert body["enabled"] is False
    assert body["trusted"] is False
    assert body["security_required"] is True
    assert body["working_directory"] == "/opt/vetted-mail-mcp"
    assert body["recipient_allowlist"] == ["teammate@example.com"]
    assert body["max_writes_per_hour"] == 8

    enabled = mcp_client.put(
        f"/api/v1/mcp/servers/{body['mcp_server_id']}",
        json={"enabled": True},
    )
    assert enabled.status_code == 200
    assert enabled.json()["enabled"] is True

    disabled_policy = mcp_client.put(
        "/api/v1/security/policy",
        json={"enabled": False},
    )
    assert disabled_policy.status_code == 200
    assert disabled_policy.json()["enabled"] is False
    assert mcp_client.get("/api/v1/mcp/tools").json() == []


def test_security_policy_defaults_to_fail_closed_inspection(mcp_client):
    response = mcp_client.get("/api/v1/security/policy")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["inspect_tool_metadata"] is True
    assert body["inspect_outbound_arguments"] is True
    assert body["inspect_inbound_results"] is True
    assert body["deterministic_scanner"] is True
    assert body["model_mode"] == "mirror"


def test_blocked_original_can_only_be_revealed_through_security_endpoint(mcp_client, monkeypatch):
    from app.services.mcp_security import blocked_originals

    blocked_id = blocked_originals.put(1, "original untrusted email")

    token = "test-operator-token-for-reveal-1234567890"
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", token)
    path = f"/api/v1/security/blocked/{blocked_id}"
    assert mcp_client.get(path).status_code == 401
    assert mcp_client.get(path, headers={"Authorization": "GeistOperator wrong"}).status_code == 401
    headers = {"Authorization": f"GeistOperator {token}"}
    other_id = blocked_originals.put(2, "another user's email")
    assert (
        mcp_client.get(f"/api/v1/security/blocked/{other_id}", headers=headers).status_code == 404
    )
    response = mcp_client.get(path, headers=headers)
    assert response.headers["cache-control"] == "no-store"
    assert response.status_code == 200
    assert response.json()["content"] == "original untrusted email"
    assert (
        mcp_client.get("/api/v1/security/blocked/blocked_missing", headers=headers).status_code
        == 404
    )


def test_mcp_server_routes_hide_other_workspaces_servers(mcp_client):
    created = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())
    server_id = created.json()["mcp_server_id"]
    mcp_client.app.dependency_overrides[get_operator_principal] = lambda: OperatorPrincipal(
        subject="other-test-operator",
        authentication_method="test",
        workspace_id=2,
        is_loopback=True,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )

    assert mcp_client.get(f"/api/v1/mcp/servers/{server_id}").status_code == 404
    assert (
        mcp_client.put(f"/api/v1/mcp/servers/{server_id}", json={"enabled": True}).status_code
        == 404
    )
    assert mcp_client.post(f"/api/v1/mcp/servers/{server_id}/test").status_code == 404
    assert mcp_client.delete(f"/api/v1/mcp/servers/{server_id}").status_code == 404


def test_stdio_server_requires_command(mcp_client):
    response = mcp_client.post(
        "/api/v1/mcp/servers",
        json={"name": "broken", "transport": "stdio"},
    )
    assert response.status_code == 422


def test_remote_operator_cannot_configure_stdio_server(mcp_client):
    mcp_client.app.dependency_overrides[get_operator_principal] = lambda: OperatorPrincipal(
        subject="remote-test-operator",
        authentication_method="test",
        workspace_id=1,
        is_loopback=False,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )

    response = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())

    assert response.status_code == 422
    assert "local operator" in response.json()["detail"]


def test_local_token_file_operator_can_configure_stdio_server_through_proxy(mcp_client):
    mcp_client.app.dependency_overrides[get_operator_principal] = lambda: OperatorPrincipal(
        subject="local-managed",
        authentication_method="local-token-file",
        workspace_id=1,
        is_loopback=False,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )

    response = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload("proxied"))

    assert response.status_code == 201


def test_http_server_requires_http_url(mcp_client):
    response = mcp_client.post(
        "/api/v1/mcp/servers",
        json={"name": "remote", "transport": "http", "url": "ftp://example.com"},
    )
    assert response.status_code == 422

    ok = mcp_client.post(
        "/api/v1/mcp/servers",
        json={"name": "remote", "transport": "http", "url": "https://example.com/mcp"},
    )
    assert ok.status_code == 201


def test_invalid_name_rejected(mcp_client):
    response = mcp_client.post(
        "/api/v1/mcp/servers",
        json={"name": "bad name!", "transport": "stdio", "command": "npx"},
    )
    assert response.status_code == 422


def test_duplicate_name_conflicts(mcp_client):
    assert mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload()).status_code == 201
    duplicate = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())
    assert duplicate.status_code == 409


def test_update_cannot_break_transport_invariant(mcp_client):
    created = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())
    server_id = created.json()["mcp_server_id"]

    response = mcp_client.put(
        f"/api/v1/mcp/servers/{server_id}",
        json={"transport": "http"},  # no URL configured
    )
    assert response.status_code == 422


class _FakeManager:
    def __init__(self, tools=None, error=None):
        self.tools = tools or []
        self.error = error

    def list_tools(self, config):
        if self.error:
            raise McpError(self.error)
        return self.tools

    def invalidate(self, server_id):
        pass


def test_test_route_reports_discovered_tools(mcp_client, monkeypatch):
    import app.api.v1.endpoints.mcp as mcp_endpoints

    created = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())
    server_id = created.json()["mcp_server_id"]

    fake = _FakeManager(tools=[{"name": "read_file", "description": "Read a file"}])
    monkeypatch.setattr(mcp_endpoints, "get_mcp_manager", lambda: fake)

    response = mcp_client.post(f"/api/v1/mcp/servers/{server_id}/test")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["tools"] == [{"name": "read_file", "description": "Read a file"}]


def test_test_route_reports_connection_failure(mcp_client, monkeypatch):
    import app.api.v1.endpoints.mcp as mcp_endpoints

    created = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())
    server_id = created.json()["mcp_server_id"]

    fake = _FakeManager(error="connection refused")
    monkeypatch.setattr(mcp_endpoints, "get_mcp_manager", lambda: fake)

    response = mcp_client.post(f"/api/v1/mcp/servers/{server_id}/test")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "connection refused" in body["error"]


def test_test_route_missing_server_404(mcp_client):
    assert mcp_client.post("/api/v1/mcp/servers/999/test").status_code == 404


@pytest.fixture()
def bridge_store(monkeypatch):
    from app.services import email_credentials, proton_bridge

    store = {}

    def credentials(reference, password=None, *, delete=False):
        if delete:
            store.pop(reference, None)
            return ""
        if password is not None:
            store[reference] = password
            return ""
        return store[reference]

    monkeypatch.setattr(proton_bridge, "credentials", credentials)
    monkeypatch.setattr(email_credentials, "credentials", credentials)
    monkeypatch.setattr(proton_bridge, "test_connection", lambda settings: None)
    return store


def test_bundled_bridge_credential_lifecycle(mcp_client, bridge_store):
    payload = {
        "name": "bridge",
        "username": "user@example.com",
        "sender": "user@example.com",
        "password": "synthetic-bridge-password",
        "certificate": "synthetic-public-cert",
    }
    created = mcp_client.post("/api/v1/mcp/proton-bridge", json=payload)
    assert created.status_code == 201, created.text
    data = created.json()
    server_id = data["mcp_server_id"]
    assert not data["enabled"] and not data["trusted"] and data["security_required"]
    assert "credential_id" not in data["managed_email"]
    assert payload["password"] not in created.text
    assert data["env"] == {} and data["args"] == []
    stored = get_mcp_server(server_id)
    assert "password" not in stored.managed_email
    original_ref = stored.managed_email["credential_id"]
    assert bridge_store[original_ref] == payload["password"]
    blocked = mcp_client.put(f"/api/v1/mcp/servers/{server_id}", json={"command": "other"})
    assert blocked.status_code == 409
    assert (
        mcp_client.put(f"/api/v1/mcp/servers/{server_id}", json={"enabled": True}).status_code
        == 200
    )
    edited = mcp_client.put(
        f"/api/v1/mcp/proton-bridge/{server_id}", json={**payload, "password": None}
    )
    assert edited.status_code == 200, edited.text
    assert not edited.json()["enabled"]
    assert original_ref not in bridge_store
    assert list(bridge_store.values()) == [payload["password"]]
    duplicate = mcp_client.post("/api/v1/mcp/proton-bridge", json=payload)
    assert duplicate.status_code == 409
    assert len(bridge_store) == 1
    assert mcp_client.delete(f"/api/v1/mcp/servers/{server_id}").status_code == 204
    assert bridge_store == {}


def test_bundled_bridge_rejects_remote_hosts(mcp_client, bridge_store):
    response = mcp_client.post(
        "/api/v1/mcp/proton-bridge",
        json={
            "name": "bridge",
            "host": "example.com",
            "username": "user@example.com",
            "sender": "user@example.com",
            "password": "synthetic",
            "certificate": "cert",
        },
    )
    assert response.status_code == 422
    assert not bridge_store


def test_bundled_bridge_setup_requires_local_operator(mcp_client, bridge_store):
    mcp_client.app.dependency_overrides[get_operator_principal] = lambda: OperatorPrincipal(
        subject="remote",
        authentication_method="test",
        workspace_id=1,
        is_loopback=False,
        capabilities=ALL_OPERATOR_CAPABILITIES,
    )
    payload = {
        "name": "bridge",
        "username": "user@example.com",
        "sender": "user@example.com",
        "password": "synthetic",
        "certificate": "cert",
    }
    for route in ("/api/v1/mcp/proton-bridge", "/api/v1/mcp/proton-bridge/test"):
        assert mcp_client.post(route, json=payload).status_code == 403
    assert bridge_store == {}


@pytest.mark.parametrize("legacy", [False, True])
def test_bridge_edit_preserves_send_history(
    mcp_client, bridge_store, tmp_path, monkeypatch, legacy
):
    import json

    from app.models.database.mcp_server import update_mcp_server
    from app.services.mcp_client import McpConnection
    from app.services.mcp_tool_source import config_from_model
    from tests.fixtures.mail_bridge import bridge_fixture

    monkeypatch.setattr("app.runtime_config.default_data_dir", lambda: tmp_path)
    with bridge_fixture(tmp_path) as (settings, state):
        payload = {k: v for k, v in settings.items() if k != "send_ledger"}
        payload["name"] = "rotation"
        created = mcp_client.post("/api/v1/mcp/proton-bridge", json=payload)
        assert created.status_code == 201, created.text
        server_id = created.json()["mcp_server_id"]
        assert "ledger_id" not in created.json()["managed_email"]
        server = get_mcp_server(server_id)
        if legacy:
            managed = dict(server.managed_email)
            managed.pop("ledger_id")
            server = update_mcp_server(server_id, {"managed_email": managed})
        first = config_from_model(server)
        args = {"to": ["recipient@example.test"], "subject": "Test", "body": "Synthetic mail"}
        connection = McpConnection(first)
        try:
            connection.call_tool("send_message", args, idempotency_key="d" * 64)
        finally:
            connection.close()
        edited = mcp_client.put(
            f"/api/v1/mcp/proton-bridge/{server_id}", json={**payload, "password": None}
        )
        assert edited.status_code == 200, edited.text
        second = config_from_model(get_mcp_server(server_id))
        assert first.managed_email["credential_id"] != second.managed_email["credential_id"]
        assert first.managed_email["send_ledger"] == second.managed_email["send_ledger"]
        connection = McpConnection(second)
        try:
            result = json.loads(
                connection.call_tool("send_message", args, idempotency_key="d" * 64)
            )
            assert result["replayed"]
            assert len(state["messages"]) == 1
        finally:
            connection.close()
