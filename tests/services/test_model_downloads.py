"""Tests for local model weight downloads and packed-folder detection."""
import importlib
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser
from app.services import model_downloads
from app.services.job_queue import JobWorker


@pytest.fixture()
def downloads_env(tmp_path, monkeypatch):
    """SQLite database plus an isolated packed weights folder."""
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'downloads.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setenv("GEIST_JOB_WORKER_ENABLED", "false")
    weights_root = tmp_path / "model_weights"
    weights_root.mkdir()
    monkeypatch.setenv("LOCAL_WEIGHTS_DIR", str(weights_root))
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=1,
                username="ddworetzky",
                name="David Dworetzky",
                email="david@phantasmal.ai",
                password="",
            )
        )
        session.commit()
    yield weights_root
    Session.remove()
    Base.metadata.drop_all(bind=engine)
    configure_database(original_config)


@pytest.fixture()
def downloads_client(downloads_env):
    # Mount just the models router so tests run without torch-heavy app.main.
    from app.api.v1.endpoints.models import router as models_router

    app = FastAPI()
    app.include_router(models_router, prefix="/api/v1/models", tags=["models"])
    with TestClient(app) as client:
        yield client


def _fake_weights(weights_root, directory: str, size: int = 16) -> None:
    model_dir = weights_root / directory
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"0" * size)


def test_model_dir_name_conventions():
    assert model_downloads.model_dir_name("Qwen/Qwen3-4B") == "Qwen_Qwen3-4B"
    # Both Llama 3.1 IDs must land in the legacy packed folder runners load.
    assert model_downloads.model_dir_name("Meta-Llama-3.1-8B-Instruct") == "llama_3_1"
    assert (
        model_downloads.model_dir_name("meta-llama/Meta-Llama-3.1-8B-Instruct") == "llama_3_1"
    )
    assert (
        model_downloads.hf_repo_id("Meta-Llama-3.1-8B-Instruct")
        == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )


def test_local_statuses_detect_downloaded_and_extra_dirs(downloads_env):
    weights_root = downloads_env
    _fake_weights(weights_root, "Qwen_Qwen3-4B")
    _fake_weights(weights_root, "some_custom_finetune")
    # Directories without weight files are ignored.
    (weights_root / "empty_dir").mkdir()

    statuses = model_downloads.local_model_statuses(user_id=1)
    by_id = {entry["id"]: entry for entry in statuses["models"]}

    assert by_id["Qwen/Qwen3-4B"]["downloaded"] is True
    assert by_id["Qwen/Qwen3-4B"]["size_bytes"] > 0
    assert by_id["Qwen/Qwen3-8B"]["downloaded"] is False
    # Server-backed heavyweights never appear as downloadable local models.
    assert "meta-llama/Llama-3.3-70B-Instruct" not in by_id

    detected = [entry["directory"] for entry in statuses["detected_directories"]]
    assert detected == ["some_custom_finetune"]


def test_enqueue_validates_model_and_dedupes(downloads_env):
    with pytest.raises(ValueError, match="not a downloadable"):
        model_downloads.enqueue_model_download("glm-5.2", user_id=1)
    with pytest.raises(ValueError, match="not a downloadable"):
        model_downloads.enqueue_model_download("no-such-model", user_id=1)

    first = model_downloads.enqueue_model_download("Qwen/Qwen3-4B", user_id=1)
    assert first["status"] == "queued"
    second = model_downloads.enqueue_model_download("Qwen/Qwen3-4B", user_id=1)
    assert second["job_id"] == first["job_id"]


def test_enqueue_rejects_downloaded_and_gated_without_token(downloads_env, monkeypatch):
    weights_root = downloads_env
    _fake_weights(weights_root, "Qwen_Qwen3-4B")
    with pytest.raises(ValueError, match="already downloaded"):
        model_downloads.enqueue_model_download("Qwen/Qwen3-4B", user_id=1)

    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="gated"):
        model_downloads.enqueue_model_download("google/gemma-3-1b-it", user_id=1)

    monkeypatch.setenv("HF_TOKEN", "hf_token_present")
    gated_job = model_downloads.enqueue_model_download("google/gemma-3-1b-it", user_id=1)
    assert gated_job["status"] == "queued"


def test_download_job_handler_runs_snapshot_download(downloads_env, monkeypatch):
    weights_root = downloads_env
    calls = {}

    def fake_snapshot_download(repo_id, local_dir, token=None, revision=None):
        calls["repo_id"] = repo_id
        calls["local_dir"] = local_dir
        calls["token"] = token
        calls["revision"] = revision
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "config.json"), "w") as f:
            f.write("{}")

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("FakeHub", (), {"snapshot_download": staticmethod(fake_snapshot_download)}),
    )
    monkeypatch.setenv("HF_TOKEN", "hf_worker_token")

    model_downloads.enqueue_model_download("Qwen/Qwen3-1.7B", user_id=1, revision="abc123")
    worker = JobWorker(poll_interval=0.01)
    assert worker.run_once() is True

    assert calls["repo_id"] == "Qwen/Qwen3-1.7B"
    assert calls["local_dir"] == os.path.join(str(weights_root), "Qwen_Qwen3-1.7B")
    assert calls["token"] == "hf_worker_token"
    assert calls["revision"] == "abc123"
    assert model_downloads.is_model_downloaded("Qwen/Qwen3-1.7B")

    statuses = model_downloads.local_model_statuses(user_id=1)
    entry = next(e for e in statuses["models"] if e["id"] == "Qwen/Qwen3-1.7B")
    assert entry["downloaded"] is True
    assert entry["download_status"] == "succeeded"


def test_download_api_roundtrip(downloads_client, downloads_env):
    response = downloads_client.post(
        "/api/v1/models/download", json={"model_id": "Qwen/Qwen3-4B"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["model_id"] == "Qwen/Qwen3-4B"
    assert body["status"] == "queued"

    invalid = downloads_client.post(
        "/api/v1/models/download", json={"model_id": "glm-5.2"}
    )
    assert invalid.status_code == 400

    local = downloads_client.get("/api/v1/models/local")
    assert local.status_code == 200
    local_body = local.json()
    entry = next(e for e in local_body["models"] if e["id"] == "Qwen/Qwen3-4B")
    assert entry["download_status"] == "queued"
    assert entry["downloaded"] is False
    assert local_body["weights_root"] == str(downloads_env)
