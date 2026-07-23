"""Tests for provider key management endpoints and stored-key resolution."""
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
def providers_client(tmp_path, monkeypatch):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'providers-api.sqlite3'}",
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


def test_list_providers_reports_env_and_stored_state(providers_client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_configured")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    response = providers_client.get("/api/v1/providers/")
    assert response.status_code == 200
    providers = {entry["id"]: entry for entry in response.json()}

    assert providers["groq"]["env_configured"] is True
    assert providers["groq"]["has_stored_key"] is False
    assert providers["moonshot"]["env_configured"] is False
    assert providers["huggingface"]["api_key_env"] == "HUGGING_FACE_HUB_TOKEN"
    assert providers["self-hosted"]["supports_base_url"] is True
    # Raw keys must never appear anywhere in the listing.
    assert "gsk_env_configured" not in response.text


def test_put_key_stores_masked_and_delete_removes(providers_client):
    put_response = providers_client.put(
        "/api/v1/providers/openai/key",
        json={"api_key": "sk-super-secret-value-1234"},
    )
    assert put_response.status_code == 200
    body = put_response.json()
    assert body["has_stored_key"] is True
    assert body["key_hint"] == "****1234"
    assert "sk-super-secret-value-1234" not in put_response.text

    delete_response = providers_client.delete("/api/v1/providers/openai/key")
    assert delete_response.status_code == 200
    assert delete_response.json()["has_stored_key"] is False

    missing_delete = providers_client.delete("/api/v1/providers/openai/key")
    assert missing_delete.status_code == 404


def test_put_key_validates_provider_and_payload(providers_client):
    unknown = providers_client.put(
        "/api/v1/providers/not-a-provider/key", json={"api_key": "abc12345"}
    )
    assert unknown.status_code == 400

    blank = providers_client.put("/api/v1/providers/openai/key", json={"api_key": "   "})
    assert blank.status_code == 400

    base_url_rejected = providers_client.put(
        "/api/v1/providers/openai/key",
        json={"api_key": "sk-abc12345", "base_url": "http://localhost:8000/v1"},
    )
    assert base_url_rejected.status_code == 400


def test_self_hosted_key_accepts_base_url(providers_client):
    response = providers_client.put(
        "/api/v1/providers/self-hosted/key",
        json={"api_key": "local-key-9876", "base_url": "http://localhost:8000/v1"},
    )
    assert response.status_code == 200
    assert response.json()["base_url"] == "http://localhost:8000/v1"

    from app.services.provider_keys import resolve_base_url

    assert resolve_base_url("self-hosted", user_id=1) == "http://localhost:8000/v1"


def test_resolve_api_key_prefers_stored_over_env(providers_client, monkeypatch):
    from app.services.provider_keys import resolve_api_key

    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert resolve_api_key("deepseek", user_id=1) == "env-key"

    providers_client.put(
        "/api/v1/providers/deepseek/key", json={"api_key": "stored-key-5678"}
    )
    assert resolve_api_key("deepseek", user_id=1) == "stored-key-5678"

    providers_client.delete("/api/v1/providers/deepseek/key")
    assert resolve_api_key("deepseek", user_id=1) == "env-key"


def test_online_agent_uses_stored_key(providers_client):
    providers_client.put(
        "/api/v1/providers/openai/key", json={"api_key": "sk-stored-agent-key"}
    )

    from agents.online_agent import OnlineAgent

    assert OnlineAgent._get_stored_api_key("openai") == "sk-stored-agent-key"


def test_huggingface_token_resolution(providers_client, monkeypatch):
    from app.services.provider_keys import resolve_huggingface_token

    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_env_fallback")
    assert resolve_huggingface_token(user_id=1) == "hf_env_fallback"

    providers_client.put(
        "/api/v1/providers/huggingface/key", json={"api_key": "hf_stored_token"}
    )
    assert resolve_huggingface_token(user_id=1) == "hf_stored_token"
