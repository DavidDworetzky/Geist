import importlib

import pytest
from fastapi.testclient import TestClient

from app.models.database.chat_session import update_chat_history
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
def memory_client(tmp_path, monkeypatch):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'memory-api.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "false")
    monkeypatch.setenv("GEIST_MEMORY_IDLE_SECONDS", "0")
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


def test_folder_and_chat_memory_settings_api(memory_client):
    folder_response = memory_client.post(
        "/api/v1/memory/folders",
        json={"name": "Private research", "color": "mint"},
    )
    assert folder_response.status_code == 201
    folder_id = folder_response.json()["folder_id"]

    update_chat_history(
        "Hello",
        "Hi",
        session_id=71,
        user_id=1,
        status="completed",
        run_id="api-run",
    )
    settings_response = memory_client.patch(
        "/api/v1/memory/chats/71",
        json={"folder_id": folder_id, "memory_mode": "private"},
    )

    assert settings_response.status_code == 200
    assert settings_response.json()["effective_scope"] == "folder"
    folders = memory_client.get("/api/v1/memory/folders").json()
    assert folders[0]["chat_count"] == 1

    renamed = memory_client.patch(
        f"/api/v1/memory/folders/{folder_id}",
        json={"name": "Renamed research"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Renamed research"

    deleted = memory_client.delete(f"/api/v1/memory/folders/{folder_id}")
    assert deleted.status_code == 204
    settings = memory_client.get("/api/v1/memory/chats/71").json()
    assert settings["folder_id"] is None
    assert settings["effective_scope"] == "private"


def test_memory_api_does_not_expose_another_users_chat(memory_client):
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=2,
                username="other",
                name="Other",
                email="other@example.com",
                password="",
            )
        )
        session.commit()
    update_chat_history(
        "Private",
        "Hidden",
        session_id=72,
        user_id=2,
        status="completed",
        run_id="other-run",
    )

    assert memory_client.get("/api/v1/memory/chats/72").status_code == 404


def test_profile_fact_can_be_corrected_and_deleted(memory_client):
    from app.services.memory_processor import process_chat_memory

    chat = update_chat_history(
        "Remember profile-copper.",
        "Saved",
        session_id=73,
        user_id=1,
        status="completed",
        run_id="profile-run",
    )
    process_chat_memory(
        user_id=1,
        chat_session_id=73,
        expected_revision=chat.memory_revision,
    )
    profile = memory_client.get("/api/v1/memory/profile").json()
    memory_id = profile["facts"][0]["memory_id"]

    corrected = memory_client.patch(
        f"/api/v1/memory/records/{memory_id}",
        json={"content": "The user prefers profile-bronze."},
    )
    assert corrected.status_code == 200
    assert "profile-bronze" in memory_client.get("/api/v1/memory/profile").json()["summary"]
    rejected_secret = memory_client.patch(
        f"/api/v1/memory/records/{memory_id}",
        json={"content": "My API key is never-store-this"},
    )
    assert rejected_secret.status_code == 422

    deleted = memory_client.delete(f"/api/v1/memory/records/{memory_id}")
    assert deleted.status_code == 204
    deleted_profile = memory_client.get("/api/v1/memory/profile").json()
    assert deleted_profile["facts"] == []
    assert deleted_profile["summary"] is None
