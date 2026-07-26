import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from agents.architectures import llama_server_process
from app import main
from app.services import local_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_app_lifespan_starts_and_stops_the_job_worker(monkeypatch):
    events = []
    monkeypatch.setattr(main, "start_worker", lambda: events.append("start"))
    monkeypatch.setattr(main, "stop_worker", lambda: events.append("stop"))
    monkeypatch.setattr(
        local_models, "shutdown_local_model_manager", lambda: events.append("models")
    )
    monkeypatch.setattr(
        llama_server_process,
        "shutdown_llama_server_manager",
        lambda: events.append("llama"),
    )
    app = main.create_app()

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert events == ["start"]
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/health/live").json()["status"] == "live"
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "version": main.application_version(),
            "checks": {"lifespan": "ok", "database": "ok"},
        }
        assert client.get("/api/v1/system").json()["spa"] is False
        assert client.get("/").json() == {"Version": "1.0"}

    assert events == ["start", "stop", "models", "llama"]


def test_core_app_import_does_not_load_optional_inference_or_voice_stacks():
    script = """
import importlib.abc
import sys

blocked = {"mlx", "sounddevice", "sphn", "torch", "torchaudio", "transformers"}

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise RuntimeError(f"optional dependency imported during startup: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptionalImports())
import app.main
"""
    environment = os.environ.copy()
    environment["GEIST_JOB_WORKER_ENABLED"] = "false"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
