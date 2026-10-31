import asyncio
import json
import threading
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.testclient import TestClient
from websockets.sync.server import serve

from app.services import local_live_voice as local


@pytest.fixture(autouse=True)
def clean_configuration(monkeypatch):
    for name in ("BACKEND", "URL", "MODEL", "SAMPLE_RATE", "FRAME_SAMPLES"):
        monkeypatch.delenv(f"GEIST_LOCAL_LIVE_VOICE_{name}", raising=False)


@pytest.fixture
def client():
    app = FastAPI()

    @app.websocket("/live/local")
    async def voice(socket: WebSocket):
        await local.serve_local_voice(socket)

    with TestClient(app) as connection:
        yield connection


@contextmanager
def upstream(monkeypatch, handler):
    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_BACKEND", "websocket")
        monkeypatch.setenv(
            "GEIST_LOCAL_LIVE_VOICE_URL", f"ws://127.0.0.1:{server.socket.getsockname()[1]}/voice"
        )
        try:
            yield
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_default_uses_private_moshi_adapter(client, monkeypatch):
    config = local.local_voice_backend()
    assert config.backend == "moshi"
    assert config.transport.sample_rate == 24000
    assert config.transport.frame_samples == 1920
    assert local.local_voice_provider()["provider"] == "local_live"

    async def adapter(socket):
        await socket.accept()
        await socket.send_json({"type": "ready"})
        await socket.close()

    worker = AsyncMock(side_effect=adapter)
    monkeypatch.setattr(local, "serve_moshi", worker)
    with client.websocket_connect("/live/local") as socket:
        assert socket.receive_json() == {"type": "ready"}
    worker.assert_awaited_once()


@pytest.mark.parametrize("rate, samples", [(16000, 320), (24000, 1920), (48000, 4800)])
def test_configurable_audio_and_catalog(monkeypatch, rate, samples):
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_BACKEND", "websocket")
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_URL", "ws://localhost:8998/voice")
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_SAMPLE_RATE", str(rate))
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_FRAME_SAMPLES", str(samples))
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_MODEL", "another-engine")
    config = local.local_voice_backend().transport
    assert config.sample_rate == rate
    assert config.frame_samples == samples
    assert "localhost" not in config.model_dump_json()
    catalog = local.local_voice_provider()
    assert catalog["default_model"] == "another-engine"
    assert catalog["models"][0]["voices"] == []


@pytest.mark.parametrize(
    "name, value",
    [
        ("BACKEND", "unknown"),
        ("URL", ""),
        ("URL", "http://localhost:8000"),
        ("URL", "ws://user:secret@localhost:8000"),
        ("URL", "ws://localhost:99999"),
        ("URL", "ws://localhost/#secret"),
        ("SAMPLE_RATE", "44100"),
        ("SAMPLE_RATE", "invalid"),
        ("FRAME_SAMPLES", "1"),
        ("FRAME_SAMPLES", "4800"),
    ],
)
def test_invalid_server_configuration_fails_closed(client, monkeypatch, name, value):
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_BACKEND", "websocket")
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_URL", "ws://localhost:8998/voice")
    monkeypatch.setenv(f"GEIST_LOCAL_LIVE_VOICE_{name}", value)
    with pytest.raises(HTTPException) as error:
        local.local_voice_backend()
    assert error.value.status_code == 503
    with client.websocket_connect("/live/local") as socket:
        assert "GEIST_LOCAL_LIVE_VOICE" in socket.receive_json()["message"]


def test_real_upstream_relays_audio_captions_and_closes_on_disconnect(client, monkeypatch):
    closed = threading.Event()
    auth = []

    def handler(socket):
        auth.append(socket.request.headers.get("Authorization"))
        try:
            socket.send(
                json.dumps(
                    {
                        "type": "ready",
                        "protocol": "geist-pcm-v1",
                        "sample_rate": 16000,
                        "frame_samples": 320,
                    }
                )
            )
            for audio in socket:
                socket.send(audio)
                socket.send(json.dumps({"type": "transcript", "role": "user", "text": "Hello"}))
                socket.send(json.dumps({"type": "transcript", "text": "Hi"}))
        finally:
            closed.set()

    with upstream(monkeypatch, handler):
        monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_SAMPLE_RATE", "16000")
        monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_FRAME_SAMPLES", "320")
        with client.websocket_connect(
            "/live/local?url=ws://ignored.invalid",
            headers={"Authorization": "GeistOperator private"},
        ) as socket:
            assert socket.receive_json() == {"type": "ready"}
            audio = b"\0" * 1280
            socket.send_bytes(audio)
            assert socket.receive_bytes() == audio
            assert socket.receive_json() == {"type": "transcript", "role": "user", "text": "Hello"}
            assert socket.receive_json() == {
                "type": "transcript",
                "role": "assistant",
                "text": "Hi",
            }
        assert closed.wait(3)
    assert auth == [None]


@pytest.mark.parametrize(
    "message",
    [
        b"bad",
        "not-json",
        '{"type":"error","message":"private upstream details"}',
        '{"type":"transcript","role":"tool","text":"bad"}',
    ],
)
def test_malformed_upstream_is_generic_to_client(client, monkeypatch, message):
    def handler(socket):
        socket.send(
            json.dumps(
                {
                    "type": "ready",
                    "protocol": "geist-pcm-v1",
                    "sample_rate": 24000,
                    "frame_samples": 1920,
                }
            )
        )
        socket.send(message)
        list(socket)

    with upstream(monkeypatch, handler), client.websocket_connect("/live/local") as socket:
        assert socket.receive_json()["type"] == "ready"
        error = socket.receive_json()
        assert error["type"] == "error"
        assert "private" not in error["message"]
        assert "stopped responding" in error["message"]


def test_invalid_browser_audio_is_not_forwarded(client, monkeypatch):
    received = []

    def handler(socket):
        socket.send(
            json.dumps(
                {
                    "type": "ready",
                    "protocol": "geist-pcm-v1",
                    "sample_rate": 24000,
                    "frame_samples": 1920,
                }
            )
        )
        received.extend(socket)

    with upstream(monkeypatch, handler), client.websocket_connect("/live/local") as socket:
        socket.receive_json()
        socket.send_bytes(b"wrong size")
        assert socket.receive_json()["type"] == "error"
        assert socket.receive()["type"] == "websocket.close"
    assert received == []


def test_upstream_readiness_timeout_closes_connection(client, monkeypatch):
    closed = threading.Event()

    def handler(socket):
        list(socket)
        closed.set()

    monkeypatch.setattr(local, "READY_TIMEOUT", 0.05)
    with upstream(monkeypatch, handler), client.websocket_connect("/live/local") as socket:
        assert socket.receive_json()["type"] == "error"
        assert socket.receive()["type"] == "websocket.close"
        assert closed.wait(3)


def test_disconnect_cancels_pending_upstream_connection(client, monkeypatch):
    started, cancelled = threading.Event(), threading.Event()

    @asynccontextmanager
    async def stalled(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
            yield
        finally:
            cancelled.set()

    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_BACKEND", "websocket")
    monkeypatch.setenv("GEIST_LOCAL_LIVE_VOICE_URL", "ws://localhost:8998/voice")
    monkeypatch.setattr(local, "connect", stalled)
    with client.websocket_connect("/live/local"):
        assert started.wait(3)
    assert cancelled.wait(3)


def test_upstream_audio_format_must_match_configuration(client, monkeypatch):
    def handler(socket):
        socket.send(
            json.dumps(
                {
                    "type": "ready",
                    "protocol": "geist-pcm-v1",
                    "sample_rate": 48000,
                    "frame_samples": 1920,
                }
            )
        )
        list(socket)

    with upstream(monkeypatch, handler), client.websocket_connect("/live/local") as socket:
        assert socket.receive_json()["type"] == "error"


@pytest.mark.parametrize("timeout_name", ["IDLE_TIMEOUT", "CALL_TIMEOUT"])
def test_idle_and_total_call_limits_close_upstream(client, monkeypatch, timeout_name):
    def handler(socket):
        socket.send(
            json.dumps(
                {
                    "type": "ready",
                    "protocol": "geist-pcm-v1",
                    "sample_rate": 24000,
                    "frame_samples": 1920,
                }
            )
        )
        list(socket)

    monkeypatch.setattr(local, timeout_name, 0.05)
    with upstream(monkeypatch, handler), client.websocket_connect("/live/local") as socket:
        assert socket.receive_json()["type"] == "ready"
        assert socket.receive_json()["type"] == "error"
        assert socket.receive()["type"] == "websocket.close"


def test_unicode_caption_fits_bounded_message_limit(client, monkeypatch):
    text = "語" * 4000

    def handler(socket):
        socket.send(
            json.dumps(
                {
                    "type": "ready",
                    "protocol": "geist-pcm-v1",
                    "sample_rate": 24000,
                    "frame_samples": 1920,
                }
            )
        )
        socket.send(json.dumps({"type": "transcript", "text": text}))
        list(socket)

    with upstream(monkeypatch, handler), client.websocket_connect("/live/local") as socket:
        assert socket.receive_json()["type"] == "ready"
        assert socket.receive_json()["text"] == text
