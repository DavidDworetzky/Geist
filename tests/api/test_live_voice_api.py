from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints import voice
from app.services import live_voice


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(voice.router, prefix="/voice")
    with TestClient(app) as client:
        yield client


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-server-key")
    post = AsyncMock(
        return_value=httpx.Response(
            201,
            json={
                "session": {"id": "live_opaque"},
                "transport": {"type": "webrtc", "sdp": "answer"},
                "private_field": "do-not-forward",
            },
        )
    )
    connection = AsyncMock()
    connection.__aenter__.return_value.post = post
    monkeypatch.setattr(live_voice.httpx, "AsyncClient", Mock(return_value=connection))
    return post


def test_session_uses_live_contract_and_keeps_credentials_server_side(client, upstream):
    response = client.post("/voice/live/session", json={"sdp": "v=0\r\n", "voice": "quartz"})
    assert response.status_code == 201
    assert response.json() == {"session_id": "live_opaque", "sdp": "answer"}
    assert upstream.call_args.args == ("https://api.openai.com/v1/live/sessions",)
    payload = upstream.call_args.kwargs["json"]
    assert payload["session"]["model"] == "gpt-live-1"
    assert payload["session"]["delegation"] == {"type": "client"}
    assert payload["session"]["audio"] == {"output": {"voice": "quartz"}}
    assert payload["transport"] == {"type": "webrtc", "sdp": "v=0\r\n"}
    assert "test-server-key" not in response.text
    assert "no backend tools" in payload["session"]["instructions"]


def test_tool_delegation_is_opt_in_and_retains_application_approvals(client, upstream):
    response = client.post("/voice/live/session", json={"sdp": "v=0\n", "tools_enabled": True})
    assert response.status_code == 201
    session = upstream.call_args.kwargs["json"]["session"]
    assert session["delegation"] == {"type": "client"}
    assert "enabled tool catalog" in session["instructions"]
    assert "spoken consent does not replace" in session["instructions"]
    assert "after the backend confirms success" in session["instructions"]


@pytest.mark.parametrize(
    "body",
    [
        {"sdp": ""},
        {"sdp": "not-sdp"},
        {"sdp": "v=0\n", "model": "gpt-realtime"},
        {"sdp": "v=0\n", "voice": "unknown"},
        {"sdp": "v=0\n" + "x" * 65536},
    ],
)
def test_invalid_session_rejected_before_provider_call(client, upstream, body):
    assert client.post("/voice/live/session", json=body).status_code == 422
    upstream.assert_not_called()


def test_missing_key_is_actionable(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/voice/live/session", json={"sdp": "v=0\n"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_provider_errors_do_not_expose_upstream_payload(client, upstream, status):
    upstream.return_value = httpx.Response(status, text="secret upstream body")
    response = client.post("/voice/live/session", json={"sdp": "v=0\n"})
    assert response.status_code == 502
    assert "secret upstream body" not in response.text


def test_timeout_and_invalid_response(client, upstream):
    upstream.side_effect = httpx.ReadTimeout("secret upstream details")
    assert client.post("/voice/live/session", json={"sdp": "v=0\n"}).status_code == 504
    upstream.side_effect = None
    upstream.return_value = httpx.Response(201, json={"session": {}})
    assert client.post("/voice/live/session", json={"sdp": "v=0\n"}).status_code == 502


def test_catalog_includes_conversation_without_changing_default(client):
    response = client.get("/voice/models").json()
    assert response["default_provider"] == "sesame"
    provider = next(p for p in response["providers"] if p["provider"] == "openai_live")
    assert provider["mode"] == "conversation"
    assert provider["default_model"] == "gpt-live-1"
    assert provider["models"][0]["voices"][0]["id"] == "marin"
    moshi = next(p for p in response["providers"] if p["provider"] == "local_live")
    assert moshi["mode"] == "conversation"
    assert moshi["models"][0]["streaming_mode"] == "websocket"


def test_dictation_only_returns_text_and_never_creates_an_agent(client, monkeypatch):
    from tests.services.test_dictation import recording

    transcribe = Mock(return_value="Draft text")
    factory = Mock(side_effect=AssertionError("Dictation must never create a chat agent"))
    monkeypatch.setattr(voice, "transcribe_dictation", transcribe)
    monkeypatch.setattr(voice.UserSettingsService, "create_agent_from_default_workspace", factory)
    response = client.post(
        "/voice/transcribe?provider=whisper",
        files={"audio_file": ("dictation.wav", recording(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "Draft text"}
    factory.assert_not_called()


def test_dictation_rejects_invalid_provider_and_oversize_recording(client):
    response = client.post(
        "/voice/transcribe?provider=bad", files={"audio_file": ("dictation.wav", b"a", "audio/wav")}
    )
    assert response.status_code == 422
    response = client.post(
        "/voice/transcribe",
        files={
            "audio_file": ("dictation.wav", b"a" * (voice.MAX_DICTATION_BYTES + 1), "audio/wav")
        },
    )
    assert response.status_code == 413


def test_local_route_reports_transport_and_dispatches_moshi(client, monkeypatch):
    from app.services import local_live_voice

    monkeypatch.delenv("GEIST_LOCAL_LIVE_VOICE_BACKEND", raising=False)
    response = client.get("/voice/live/local")
    assert response.status_code == 200
    assert response.json() == {
        "endpoint": "/api/v1/voice/live/local",
        "protocol": "geist-pcm-v1",
        "model": "kyutai/moshiko-mlx-q4",
        "sample_rate": 24000,
        "frame_samples": 1920,
    }

    async def adapter(socket):
        await socket.accept()
        await socket.send_json({"type": "ready"})
        await socket.close()

    worker = AsyncMock(side_effect=adapter)
    monkeypatch.setattr(local_live_voice, "serve_moshi", worker)
    with client.websocket_connect("/voice/live/local") as socket:
        assert socket.receive_json() == {"type": "ready"}
    worker.assert_awaited_once()


def test_invalid_local_configuration_does_not_hide_online_models(client, monkeypatch):
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_BACKEND", "unknown")
    assert client.get("/voice/live/local").status_code == 503
    response = client.get("/voice/models")
    assert response.status_code == 200
    providers = response.json()["providers"]
    assert any(provider["provider"] == "openai_live" for provider in providers)
    local = next(provider for provider in providers if provider["provider"] == "local_live")
    assert "invalid" in local["description"]
