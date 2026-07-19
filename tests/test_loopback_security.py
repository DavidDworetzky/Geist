import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.loopback_security import install_loopback_security


@pytest.fixture
def client():
    app = FastAPI()
    install_loopback_security(app)

    @app.get("/status")
    def status():
        return {"status": "ok"}

    @app.post("/change")
    def change():
        return {"changed": True}

    @app.websocket("/stream")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("ready")
        await websocket.close()

    with TestClient(app, base_url="http://127.0.0.1:8123") as test_client:
        yield test_client


@pytest.mark.parametrize(
    "host",
    [
        "attacker.example",
        "attacker.example:8123",
        "127.0.0.1.attacker.example:8123",
        "127.0.0.1:invalid",
    ],
)
def test_non_loopback_or_malformed_host_is_rejected(client, host):
    response = client.get("/status", headers={"host": host})

    assert response.status_code == 400
    assert response.text == "Invalid Host header"


@pytest.mark.parametrize("host", ["127.0.0.1:8123", "localhost:8123", "[::1]:8123"])
def test_loopback_hosts_are_allowed(client, host):
    response = client.get("/status", headers={"host": host})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "https://attacker.example",
        "http://127.0.0.1:9999",
        "https://127.0.0.1:8123",
        "null",
    ],
)
def test_foreign_origin_is_rejected_for_mutating_http(client, origin):
    response = client.post("/change", headers={"origin": origin})

    assert response.status_code == 403
    assert response.text == "Invalid Origin header"


def test_same_origin_and_non_browser_mutations_are_allowed(client):
    same_origin = client.post(
        "/change", headers={"origin": "http://127.0.0.1:8123"}
    )
    without_origin = client.post("/change")

    assert same_origin.json() == {"changed": True}
    assert without_origin.json() == {"changed": True}


def test_safe_http_request_does_not_require_same_origin(client):
    response = client.get("/status", headers={"origin": "https://attacker.example"})

    assert response.status_code == 200


def test_websocket_requires_same_origin_when_browser_supplies_origin(client):
    with client.websocket_connect(
        "/stream",
        headers={
            "host": "127.0.0.1:8123",
            "origin": "http://127.0.0.1:8123",
        },
    ) as websocket:
        assert websocket.receive_text() == "ready"

    with (
        pytest.raises(WebSocketDisconnect) as rejected,
        client.websocket_connect(
            "/stream",
            headers={
                "host": "127.0.0.1:8123",
                "origin": "https://attacker.example",
            },
        ),
    ):
        pass

    assert rejected.value.code == 1008


def test_non_browser_websocket_without_origin_is_allowed(client):
    with client.websocket_connect(
        "/stream", headers={"host": "127.0.0.1:8123"}
    ) as websocket:
        assert websocket.receive_text() == "ready"
