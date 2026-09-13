import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.services import moshi_voice


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(moshi_voice.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(moshi_voice.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(moshi_voice, "worker_python", lambda: Path(sys.executable))
    app = FastAPI()

    @app.websocket("/moshi")
    async def route(socket: WebSocket):
        await moshi_voice.serve_moshi(socket)

    with TestClient(app) as connection:
        yield connection
    assert not moshi_voice._active


@pytest.fixture
def worker(monkeypatch):
    launch = asyncio.create_subprocess_exec
    processes = []

    def configure(body):
        async def start(*args, **kwargs):
            process = await launch(sys.executable, "-u", "-c", body, **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(moshi_voice.asyncio, "create_subprocess_exec", start)

    yield configure
    assert all(process.returncode is not None for process in processes)


def test_local_platform_and_runtime_guards(client, monkeypatch):
    monkeypatch.setattr(moshi_voice.platform, "system", lambda: "Linux")
    with client.websocket_connect("/moshi") as socket:
        assert "native Apple Silicon" in socket.receive_json()["message"]
    monkeypatch.setattr(moshi_voice.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(moshi_voice, "worker_python", lambda: Path("/missing/moshi/python"))
    with client.websocket_connect("/moshi") as socket:
        assert "local runtime" in socket.receive_json()["message"]


def test_audio_and_captions_use_worker_and_disconnect_releases_it(client, worker):
    worker("""
import sys, struct
def emit(kind, data=b''):
    sys.stdout.buffer.write(struct.pack('<BI', kind, len(data)) + data)
    sys.stdout.buffer.flush()
emit(0)
while True:
    audio = sys.stdin.buffer.read(7680)
    if len(audio) != 7680: break
    emit(1, audio)
    emit(2, b' Hello')
    emit(4)
""")
    with client.websocket_connect("/moshi") as socket:
        assert socket.receive_json() == {"type": "ready"}
        with client.websocket_connect("/moshi") as busy:
            assert "already in a voice call" in busy.receive_json()["message"]
        audio = b"\0" * 7680
        socket.send_bytes(audio)
        assert socket.receive_bytes() == audio
        assert socket.receive_json() == {"type": "transcript", "text": " Hello"}


@pytest.mark.parametrize(
    "frames, expected", [([b"bad"], "Invalid"), ([b"\0" * 7680] * 11, "keep up")]
)
def test_invalid_audio_and_slow_inference_end_call(client, worker, frames, expected):
    worker(
        "import sys, time; sys.stdout.buffer.write(b'\\0' * 5); sys.stdout.buffer.flush(); time.sleep(60)"
    )
    with client.websocket_connect("/moshi") as socket:
        assert socket.receive_json() == {"type": "ready"}
        for frame in frames:
            socket.send_bytes(frame)
        assert expected in socket.receive_json()["message"]


def test_disconnect_during_model_loading_terminates_worker(client, worker):
    worker("import time; time.sleep(60)")
    with client.websocket_connect("/moshi"):
        pass
