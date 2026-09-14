import asyncio
import logging
import struct
import subprocess
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


@pytest.mark.parametrize("corrupt_frame", [False, True])
def test_worker_failures_are_logged_without_exposing_details(client, worker, caplog, corrupt_frame):
    body = "import os; os.write(2, b'codec initialization failed\\n')"
    if corrupt_frame:
        body += "; os.write(1, b'broken protocol')"
    worker(body)
    with (
        caplog.at_level(logging.WARNING, logger=moshi_voice.__name__),
        client.websocket_connect("/moshi") as socket,
    ):
        assert socket.receive_json() == {
            "type": "error",
            "message": "Moshi stopped responding. End the call and reconnect.",
        }
        # Wait for shutdown so the stderr consumer finishes before checking logs.
        assert socket.receive()["type"] == "websocket.close"
    assert "codec initialization failed" in caplog.text
    assert "Moshi voice session failed" in caplog.text
    assert any(record.exc_info for record in caplog.records)


def test_worker_protocol_survives_native_stdout_writes():
    script = Path(__file__).resolve().parents[2] / "scripts/runtimes/moshi/worker.py"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy, sys; worker = runpy.run_path(sys.argv[1]); "
            "worker['isolate_protocol_output'](); "
            "os.write(1, b'native device warning\\n'); "
            "print('Python diagnostic', flush=True); "
            "worker['emit'](0); worker['emit'](1, b'audio')",
            str(script),
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert result.stdout == struct.pack("<BI", 0, 0) + struct.pack("<BI", 1, 5) + b"audio"
    assert b"native device warning" in result.stderr
    assert b"Python diagnostic" in result.stderr
