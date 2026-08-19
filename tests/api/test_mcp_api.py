import importlib
from types import SimpleNamespace

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
from app.services.mcp_client import McpError


@pytest.fixture()
def mcp_client(tmp_path, monkeypatch):
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
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=1,
                username="ddworetzky",
                name="David Dworetzky",
                email="david@phantasmal.ai",
                password="",
            )
        )
        session.commit()
    from app.main import create_app

    with TestClient(create_app()) as client:
        yield client
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
    assert enabled.status_code == 409
    assert "security inspection" in enabled.json()["detail"]


def test_mcp_server_routes_hide_other_users_servers(mcp_client, monkeypatch):
    import app.api.v1.endpoints.mcp as mcp_endpoints

    created = mcp_client.post("/api/v1/mcp/servers", json=_stdio_payload())
    server_id = created.json()["mcp_server_id"]
    monkeypatch.setattr(
        mcp_endpoints,
        "get_current_user",
        lambda: SimpleNamespace(user_id=2),
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
