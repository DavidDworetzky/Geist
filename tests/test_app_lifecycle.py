import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from agents.architectures import llama_server_process
from app import main
from app.models.database.geist_user import WorkspaceModel
from app.services import local_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _use_test_workspace(monkeypatch):
    monkeypatch.setattr(
        "app.security.operator.get_default_workspace",
        lambda: WorkspaceModel(1, "default", "Local Workspace"),
    )


def test_app_lifespan_starts_and_stops_the_job_worker(monkeypatch):
    _use_test_workspace(monkeypatch)
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN_FILE", raising=False)
    events = []
    monkeypatch.setattr(main, "start_worker", lambda: events.append("start"))
    monkeypatch.setattr(main, "start_scheduler", lambda: events.append("schedule-start"))
    monkeypatch.setattr(main, "stop_worker", lambda: events.append("stop"))
    monkeypatch.setattr(main, "stop_scheduler", lambda: events.append("schedule-stop"))
    monkeypatch.setattr(
        local_models, "shutdown_local_model_manager", lambda: events.append("models")
    )
    monkeypatch.setattr(
        llama_server_process,
        "shutdown_llama_server_manager",
        lambda: events.append("llama"),
    )
    app = main.create_app()

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        assert events == ["start", "schedule-start"]
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/health/live").json()["status"] == "live"
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "version": main.application_version(),
            "checks": {"lifespan": "ok", "database": "ok"},
        }
        system = client.get("/api/v1/system").json()
        assert system["spa"] is False
        assert system["version"] == main.application_version()
        assert system["platform"]["system"]
        assert system["platform"]["release"]
        assert system["platform"]["machine"]
        assert system["python"]["version"]
        assert system["inference"]["mode"] in {"local", "online"}
        assert system["inference"]["engine"]
        assert system["inference"]["model"]
        assert client.get("/").json() == {"Version": "1.0"}

    assert events == [
        "start",
        "schedule-start",
        "schedule-stop",
        "stop",
        "models",
        "llama",
    ]


def test_operator_authentication_wraps_the_application_surface(monkeypatch):
    _use_test_workspace(monkeypatch)
    token = "g" * 43
    monkeypatch.setenv("GEIST_OPERATOR_TOKEN", token)
    monkeypatch.delenv("GEIST_OPERATOR_TOKEN_FILE", raising=False)
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "false")
    app = main.create_app()

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/system").status_code == 401
        authorized = client.get(
            "/api/v1/system",
            headers={"Authorization": f"GeistOperator {token}"},
        )

    assert authorized.status_code == 200
    assert authorized.json()["apiVersion"] == str(main.api_version)


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
