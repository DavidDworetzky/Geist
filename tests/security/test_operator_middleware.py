import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.models.database.geist_user import WorkspaceModel
from app.security.middleware import OperatorAuthenticationMiddleware
from app.security.operator import OPERATOR_AUTHENTICATION_SCHEME


@pytest.fixture()
def protected_client(monkeypatch):
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", "m" * 43)
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN_FILE", raising=False)
    monkeypatch.setattr(
        "app.security.operator.get_default_workspace",
        lambda: WorkspaceModel(1, "default", "Local Workspace"),
    )
    app = FastAPI()
    app.add_middleware(OperatorAuthenticationMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/protected")
    def protected():
        return {"status": "authorized"}

    @app.websocket("/socket")
    async def socket(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("authorized")
        await websocket.close()

    return TestClient(app, base_url="http://127.0.0.1")


def operator_headers() -> dict[str, str]:
    return {"Authorization": f"{OPERATOR_AUTHENTICATION_SCHEME} {'m' * 43}"}


def test_health_is_the_narrow_unauthenticated_exception(protected_client):
    assert protected_client.get("/health").status_code == 200
    assert protected_client.get("/protected").status_code == 401


def test_operator_token_authenticates_protected_http_route(protected_client):
    response = protected_client.get("/protected", headers=operator_headers())

    assert response.status_code == 200
    assert response.json() == {"status": "authorized"}


def test_operator_token_authenticates_websocket_handshake(protected_client):
    with protected_client.websocket_connect("/socket", headers=operator_headers()) as websocket:
        assert websocket.receive_text() == "authorized"


def test_websocket_without_operator_token_is_rejected(protected_client):
    with (
        pytest.raises(WebSocketDisconnect) as raised,
        protected_client.websocket_connect("/socket"),
    ):
        pass

    assert raised.value.code == 4401
