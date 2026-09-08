"""Tests for the built-in tool catalogue API."""

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


@pytest.fixture()
def tools_client(tmp_path, monkeypatch):
    operator_token = "t" * 43
    secret = "catalog-secret-that-must-not-leak"
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "false")
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", operator_token)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_IMAGE_BASE_URL", "https://images.example.test/v2/")
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "example-image-model")

    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'tools-api.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        session.add(GeistUser(user_id=1, workspace_key="default", name="Local Workspace"))
        session.commit()

    from app.main import create_app

    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
        headers={"Authorization": f"GeistOperator {operator_token}"},
    ) as client:
        yield client, secret
    Session.remove()
    Base.metadata.drop_all(bind=engine)
    configure_database(original_config)


def test_tool_catalogue_reports_redacted_image_configuration(tools_client):
    client, secret = tools_client

    response = client.get("/agent/tools")

    assert response.status_code == 200
    assert secret not in response.text
    tools = {tool["name"]: tool for tool in response.json()["tools"]}
    assert "workspace.write_markdown" not in tools
    assert "communication.email.send" not in tools
    assert "communication.sms.send" not in tools
    assert "adapter.JobStatusAdapter.check_async_tool" not in tools
    assert tools["workspace.list_markdown"]["enabled"] is True
    assert tools["workspace.read_markdown"]["enabled"] is True
    assert tools["workspace.read_markdown"]["allows_standing_grant"] is True
    assert tools["image.generate"]["configuration"] == {
        "kind": "environment",
        "provider": "OpenAI-compatible image API",
        "api_key_configured": True,
        "base_url": "https://images.example.test/v2",
        "model": "example-image-model",
        "environment_variables": {
            "api_key": "OPENAI_API_KEY",
            "base_url": "OPENAI_IMAGE_BASE_URL",
            "model": "OPENAI_IMAGE_MODEL",
        },
    }


def test_runtime_source_catalog_never_offers_standing_grants(tools_client, monkeypatch):
    from types import SimpleNamespace

    from pydantic import BaseModel

    from agents.models.tool_calling import ToolDefinition, ToolExecutionOutput
    from app import main
    from app.services.tool_registry import ToolRegistry

    class Arguments(BaseModel):
        query: str

    registry = ToolRegistry()
    definition = ToolDefinition(
        name="runtime.lookup",
        description="Runtime lookup",
        arguments_model=Arguments,
        handler=lambda context, arguments: ToolExecutionOutput(content="result"),
    )
    registry.add_source(
        SimpleNamespace(name="test-source", definitions=lambda context: [definition])
    )
    monkeypatch.setattr(main.chat_orchestrator, "registry", registry)
    client, _secret = tools_client
    response = client.get("/agent/tools")
    assert response.status_code == 200
    assert response.json()["tools"][0]["allows_standing_grant"] is False
