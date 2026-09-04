"""API tests for prompt schedule management."""

import importlib

import pytest
from fastapi.testclient import TestClient

from app.api.utils import get_current_workspace
from app.main import create_app
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser, WorkspaceModel
from app.schemas.prompt_schedule import PromptScheduleCreate
from app.services.prompt_scheduler import (
    create_prompt_schedule,
    enqueue_prompt_schedule_now,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN_FILE", raising=False)
    original_config = DATABASE_CONFIG
    config = DatabaseConfig(
        provider="sqlite",
        database_url=f"sqlite:///{tmp_path / 'geist.sqlite3'}",
    )
    engine = configure_database(config)
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=1,
                workspace_key="default",
                username="api-user",
                name="API User",
                email="api@example.com",
                password="",
            )
        )
        session.commit()

    monkeypatch.setattr("app.main.start_worker", lambda: None)
    monkeypatch.setattr("app.main.stop_worker", lambda: None)
    monkeypatch.setattr("app.main.start_scheduler", lambda: None)
    monkeypatch.setattr("app.main.stop_scheduler", lambda: None)
    app = create_app()
    app.dependency_overrides[get_current_workspace] = lambda: WorkspaceModel(
        1, "default", "API User"
    )
    try:
        with TestClient(
            app,
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        ) as test_client:
            yield test_client
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def test_schedule_crud_and_run_now(client):
    response = client.post(
        "/api/v1/prompt-schedules/",
        json={
            "name": "Daily note",
            "prompt": "Write a daily note.",
            "cron_expression": "0 9 * * *",
            "timezone": "UTC",
        },
    )
    assert response.status_code == 201
    schedule = response.json()
    schedule_id = schedule["prompt_schedule_id"]

    listed = client.get("/api/v1/prompt-schedules/")
    assert listed.status_code == 200
    assert [item["name"] for item in listed.json()] == ["Daily note"]

    updated = client.patch(
        f"/api/v1/prompt-schedules/{schedule_id}",
        json={"enabled": False},
    )
    assert updated.status_code == 200
    assert updated.json()["enabled"] is False
    assert updated.json()["next_run_at"] is None

    run = client.post(f"/api/v1/prompt-schedules/{schedule_id}/run")
    assert run.status_code == 202
    assert run.json()["status"] == "queued"

    history = client.get(f"/api/v1/prompt-schedules/{schedule_id}/runs")
    assert history.status_code == 200
    assert [item["job_id"] for item in history.json()] == [run.json()["job_id"]]

    deleted = client.delete(f"/api/v1/prompt-schedules/{schedule_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/prompt-schedules/{schedule_id}").status_code == 404


def test_invalid_cron_returns_validation_error(client):
    response = client.post(
        "/api/v1/prompt-schedules/",
        json={
            "name": "Broken",
            "prompt": "Never runs.",
            "cron_expression": "not cron",
            "timezone": "UTC",
        },
    )

    assert response.status_code == 422


def test_scheduled_job_status_is_scoped_to_its_owner(client):
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=2,
                workspace_key="other",
                username="other-user",
                name="Other User",
                email="other@example.com",
                password="",
            )
        )
        session.commit()
    schedule = create_prompt_schedule(
        2,
        PromptScheduleCreate(
            name="Private schedule",
            prompt="Keep this result private.",
            cron_expression="0 9 * * *",
            enabled=False,
        ),
    )
    job = enqueue_prompt_schedule_now(schedule)

    wrong_user = client.get(f"/api/v1/jobs/{job.job_id}")
    wrong_user_list = client.get("/api/v1/jobs/")
    client.app.dependency_overrides[get_current_workspace] = lambda: WorkspaceModel(
        2, "other", "Other User"
    )
    owner = client.get(f"/api/v1/jobs/{job.job_id}")

    assert wrong_user.status_code == 404
    assert owner.status_code == 200
    assert all(item["job_id"] != job.job_id for item in wrong_user_list.json())
