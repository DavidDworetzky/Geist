import datetime
import importlib

import pytest
from fastapi.testclient import TestClient

from app.models.database.database import Base, Session, SessionLocal, configure_database
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser
from app.services.tool_approvals import ToolApprovalRegistry


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import main
    from app.models.database import database

    original = database.DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(provider="sqlite", database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        session.add_all(
            [
                GeistUser(user_id=1, workspace_key="default"),
                GeistUser(user_id=2, workspace_key="other"),
            ]
        )
        session.commit()
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", "test-operator-token-not-secret-123456789")
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "0")
    monkeypatch.setattr(main, "tool_approval_registry", ToolApprovalRegistry())
    monkeypatch.setattr(main.routine_scheduler, "enabled", False)
    try:
        with TestClient(
            main.create_app(),
            client=("127.0.0.1", 50000),
            headers={"Authorization": "GeistOperator test-operator-token-not-secret-123456789"},
        ) as test_client:
            yield test_client
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original)


def test_approval_http_owner_deadline_and_decision_contract(client):
    from app import main

    approvals = main.tool_approval_registry
    for call, owner, timeout in [("own", 1, 30), ("foreign", 2, 30), ("expired", 1, 0)]:
        approvals.request(
            "run",
            call,
            "test.tool",
            workspace_id=owner,
            arguments_fingerprint="args",
            definition_fingerprint="definition",
            timeout_seconds=timeout,
        )
    url = "/agent/runs/run/tool_approval"
    assert client.post(url, json={"call_id": "own", "decision": "always"}).status_code == 422
    assert client.post(url, json={"call_id": "own", "decision": "invalid"}).status_code == 422
    assert client.post(url, json={"call_id": "foreign", "decision": "approve"}).status_code == 404
    assert client.post(url, json={"call_id": "expired", "decision": "approve"}).status_code == 404
    assert client.post(url, json={"call_id": "own", "decision": "approve"}).status_code == 200
    assert client.post(url, json={"call_id": "own", "decision": "approve"}).status_code == 404
    assert (
        client.post(
            url,
            headers={"Authorization": "invalid"},
            json={"call_id": "own", "decision": "approve"},
        ).status_code
        == 401
    )


def test_overdue_routine_outcome_is_unknown_without_replaying_or_rewriting_claim(client):
    from app.models.database.agent_routine import AgentRoutine, create_routine, get_routine

    routine = create_routine(1, "Interrupted", "Work", 60, enabled=False)
    with SessionLocal() as session:
        session.query(AgentRoutine).filter_by(routine_id=routine.routine_id).update(
            {
                "last_status": "running",
                "last_run_at": datetime.datetime(2000, 1, 1),
            }
        )
        session.commit()
    response = client.get("/api/v1/routines/")
    assert response.status_code == 200
    item = response.json()[0]
    assert item["last_status"] == "outcome_unknown"
    assert item["next_run_at"] is None
    assert get_routine(routine.routine_id).last_status == "running"


def test_routine_http_ownership_validation_and_disabled_run_now(client):
    from app.models.database.agent_routine import create_routine

    foreign = create_routine(2, "Other", "Private", 60)
    response = client.post(
        "/api/v1/routines/", json={"name": "Test", "prompt": "Work", "enabled": False}
    )
    assert response.status_code == 200
    routine = response.json()
    assert routine["user_id"] == 1
    path = f"/api/v1/routines/{routine['routine_id']}"
    assert client.put(path, json={"user_id": 2}).status_code == 422
    assert client.put(path, json={"prompt": None}).status_code == 422
    for method, suffix, payload in [
        ("put", "", {"name": "Steal"}),
        ("delete", "", None),
        ("post", "/run_now", None),
    ]:
        kwargs = {"json": payload} if payload else {}
        assert (
            client.request(
                method, f"/api/v1/routines/{foreign.routine_id}{suffix}", **kwargs
            ).status_code
            == 404
        )
    queued = client.post(path + "/run_now").json()
    assert queued["run_once_requested"] is True and queued["enabled"] is False
    listed = client.get("/api/v1/routines/").json()
    assert [entry["routine_id"] for entry in listed] == [routine["routine_id"]]
    assert listed[0]["last_status"] is None
    assert client.delete(path).status_code == 200
