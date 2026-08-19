"""Focused API tests for local llama.cpp runtime controls."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints import models


def test_runtime_devices_refresh_query_reprobes_inventory(monkeypatch) -> None:
    refresh_requests = []

    class FakeManager:
        def device_inventory(self, *, refresh: bool = False):
            refresh_requests.append(refresh)
            return {"refresh": refresh}

    monkeypatch.setattr(models, "get_llama_server_manager", lambda: FakeManager())
    app = FastAPI()
    app.include_router(models.router)

    with TestClient(app) as client:
        response = client.get("/local/runtime/devices?refresh=true")

    assert response.status_code == 200
    assert response.json() == {"refresh": True}
    assert refresh_requests == [True]
