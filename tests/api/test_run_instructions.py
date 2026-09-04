from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.chat_orchestrator import RunControlRegistry


@pytest.fixture
def instructions_client(monkeypatch):
    controls = RunControlRegistry()
    saved = []
    controls.start("owned", user_id=1, chat_id=7, on_instruction=saved.append)
    controls.start("other", user_id=2, chat_id=8)
    monkeypatch.setattr(main, "run_controls", controls)
    monkeypatch.setattr(main, "get_default_user", lambda: SimpleNamespace(user_id=1))
    return TestClient(main.create_app()), controls, saved


def test_instruction_api_acknowledges_persisted_idempotent_input(instructions_client):
    client, controls, saved = instructions_client
    payload = {"instruction_id": "one", "text": "  Use local only  "}
    response = client.post("/agent/runs/owned/instructions", json=payload)
    assert response.status_code == 200
    assert saved == [{"id": "one", "text": "Use local only", "status": "queued"}]
    assert client.post("/agent/runs/owned/instructions", json=payload).status_code == 200
    assert len(saved) == 1
    controls.drain("owned")
    assert client.post("/agent/runs/owned/instructions", json=payload).status_code == 200
    assert controls.drain("owned") == []
    assert (
        client.post(
            "/agent/runs/owned/instructions",
            json={**payload, "text": "Use hosted"},
        ).status_code
        == 422
    )
    controls.seal("owned")
    assert client.post("/agent/runs/owned/instructions", json=payload).status_code == 409


@pytest.mark.parametrize("run_id", ["other", "missing"])
def test_instruction_api_rejects_wrong_owner_or_missing_run(instructions_client, run_id):
    client, _, saved = instructions_client
    assert (
        client.post(
            f"/agent/runs/{run_id}/instructions",
            json={"instruction_id": "one", "text": "Use local"},
        ).status_code
        == 409
    )
    assert saved == []


@pytest.mark.parametrize("text", ["", " ", "x" * 20_001])
def test_instruction_api_validates_text(instructions_client, text):
    client, _, saved = instructions_client
    assert (
        client.post(
            "/agent/runs/owned/instructions",
            json={"instruction_id": "one", "text": text},
        ).status_code
        == 422
    )
    assert saved == []
