"""Tests for the /api/v1/plugins endpoints."""

import importlib
import json

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


def _write_plugin(root, name, skills=(), mcp_servers=None):
    plugin_root = root / name
    manifest_dir = plugin_root / ".claude-plugin"
    manifest_dir.mkdir(parents=True)
    manifest = {"name": name, "version": "2.0.0", "description": f"{name} plugin"}
    if mcp_servers is not None:
        manifest["mcpServers"] = mcp_servers
    (manifest_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    for skill_name in skills:
        skill_dir = plugin_root / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\ndescription: {skill_name} does things\n---\nBody", encoding="utf-8"
        )
    return plugin_root


@pytest.fixture()
def plugins_client(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    monkeypatch.setenv("GEIST_PLUGIN_DIR", str(plugin_dir))
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "false")
    monkeypatch.delenv("GEIST_ENABLED_PLUGINS", raising=False)

    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'plugins-api.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
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
        yield client, plugin_dir
    Session.remove()
    Base.metadata.drop_all(bind=engine)
    configure_database(original_config)


def test_list_plugins_empty(plugins_client):
    client, _ = plugins_client
    client.post("/api/v1/plugins/refresh")
    response = client.get("/api/v1/plugins")
    assert response.status_code == 200
    assert response.json()["plugins"] == []


def test_refresh_discovers_plugins(plugins_client, monkeypatch):
    client, plugin_dir = plugins_client
    _write_plugin(
        plugin_dir,
        "reviewer",
        skills=("code-review",),
        mcp_servers={"db": {"command": "run-server"}},
    )

    refreshed = client.post("/api/v1/plugins/refresh")
    assert refreshed.status_code == 200
    plugins = refreshed.json()["plugins"]
    assert len(plugins) == 1
    plugin = plugins[0]
    assert plugin["name"] == "reviewer"
    assert plugin["version"] == "2.0.0"
    assert plugin["enabled_for_mcp"] is False
    assert plugin["skills"][0]["qualified_name"] == "reviewer:code-review"
    assert plugin["mcp_servers"][0]["qualified_name"] == "reviewer.db"
    assert plugin["mcp_servers"][0]["enabled"] is False

    monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "reviewer")
    listed = client.get("/api/v1/plugins")
    assert listed.json()["plugins"][0]["enabled_for_mcp"] is True
    assert listed.json()["plugins"][0]["mcp_servers"][0]["enabled"] is True
