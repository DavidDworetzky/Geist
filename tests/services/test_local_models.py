"""Focused tests for Geist's managed local-model store."""

from __future__ import annotations

import hashlib
import io
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.local_models import (
    CURATED_LOCAL_ARTIFACTS,
    LocalModelArtifact,
    LocalModelManager,
    default_model_home,
    local_artifact_supported,
)


MODEL_BYTES = b"GGUF" + b"test-model"


def _artifact(**overrides) -> LocalModelArtifact:
    values = {
        "id": "test-q4",
        "model_id": "test/model",
        "display_name": "Test Q4",
        "format": "gguf",
        "backend": "llama_server",
        "filename": "test-q4.gguf",
        "repo_id": "test/model-gguf",
        "revision": "immutable-revision",
        "sha256": hashlib.sha256(MODEL_BYTES).hexdigest(),
        "size_bytes": len(MODEL_BYTES),
        "quantization": "Q4_K_M",
    }
    values.update(overrides)
    return LocalModelArtifact(**values)


def _mlx_artifact(**overrides) -> LocalModelArtifact:
    values = {
        "id": "test-mlx",
        "model_id": "meta-llama/Test-MLX",
        "display_name": "Test MLX",
        "format": "snapshot",
        "backend": "mlx_llama",
        "filename": "snapshot",
        "repo_id": "meta-llama/Test-MLX",
        "revision": "a" * 40,
        "allow_patterns": ("*.json", "*.safetensors"),
        "requires_auth": True,
    }
    values.update(overrides)
    return LocalModelArtifact(**values)


@pytest.fixture
def managers():
    active = []
    yield active
    for manager in active:
        manager.shutdown()


def test_default_model_home_uses_geist_data_directory(tmp_path):
    data_directory = tmp_path / "Geist"

    with patch(
        "app.services.local_models.default_data_dir",
        return_value=data_directory,
    ) as resolve_data_directory:
        assert default_model_home({}) == data_directory / "models"

    resolve_data_directory.assert_called_once_with(environ={})


def test_download_is_verified_and_atomically_installed(tmp_path, managers):
    progress = []

    def downloader(_artifact, destination, callback):
        destination.write_bytes(MODEL_BYTES)
        callback(len(MODEL_BYTES), len(MODEL_BYTES))
        progress.append(destination)

    manager = LocalModelManager(
        tmp_path,
        artifacts=(_artifact(),),
        downloader=downloader,
    )
    managers.append(manager)

    artifact, installed = manager.ensure_artifact("test/model")

    assert artifact.id == "test-q4"
    assert installed.read_bytes() == MODEL_BYTES
    assert progress[0].name.endswith(".partial.gguf")
    assert manager.status("test-q4")["status"] == "installed"
    assert manager.status("test-q4")["sha256"] == hashlib.sha256(MODEL_BYTES).hexdigest()


def test_bad_checksum_is_rejected_without_installing(tmp_path, managers):
    def downloader(_artifact, destination, callback):
        destination.write_bytes(MODEL_BYTES)
        callback(len(MODEL_BYTES), len(MODEL_BYTES))

    manager = LocalModelManager(
        tmp_path,
        artifacts=(_artifact(sha256="0" * 64),),
        downloader=downloader,
    )
    managers.append(manager)

    with pytest.raises(ValueError, match="SHA-256"):
        manager.download_artifact("test-q4")

    status = manager.status("test-q4")
    assert status["status"] == "failed"
    assert status["path"] is None
    assert not list((tmp_path / ".downloads").glob("*.partial.gguf"))


def test_installed_artifact_is_reverified_before_inference(tmp_path, managers):
    manager = LocalModelManager(tmp_path, artifacts=(_artifact(),))
    managers.append(manager)
    target = tmp_path / "artifacts" / "test-q4" / "test-q4.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"GGUF" + b"tampered!!")

    with pytest.raises(RuntimeError, match="failed verification"):
        manager.require_installed("test-q4")

    status = manager.status("test-q4")
    assert status["status"] == "failed"
    assert status["path"] is None


def test_require_installed_does_not_queue_or_start_download(tmp_path, managers):
    download_attempts = []

    def downloader(_artifact, _destination, _callback):
        download_attempts.append(True)

    artifact = _artifact()
    manager = LocalModelManager(
        tmp_path,
        artifacts=(artifact,),
        downloader=downloader,
    )
    managers.append(manager)

    with pytest.raises(RuntimeError, match="explicitly from the Models page"):
        manager.require_installed(artifact.id)

    assert download_attempts == []
    assert artifact.id not in manager._futures
    assert manager.status(artifact.id)["status"] == "not_installed"
    assert not manager.downloads_dir.exists()


def test_transient_failure_keeps_partial_for_resume(tmp_path, managers):
    def downloader(_artifact, destination, _callback):
        destination.write_bytes(b"GGUFpartial")
        raise OSError("connection reset")

    manager = LocalModelManager(
        tmp_path,
        artifacts=(_artifact(),),
        downloader=downloader,
    )
    managers.append(manager)

    with pytest.raises(OSError, match="connection reset"):
        manager.download_artifact("test-q4")

    assert list((tmp_path / ".downloads").glob("*.partial.gguf"))


def test_retry_reports_and_reuses_existing_partial(tmp_path, managers):
    observed = []

    def downloader(_artifact, destination, callback):
        observed.append(destination.read_bytes())
        with destination.open("ab") as target:
            target.write(MODEL_BYTES[4:])
        callback(len(MODEL_BYTES), len(MODEL_BYTES))

    manager = LocalModelManager(
        tmp_path,
        artifacts=(_artifact(),),
        downloader=downloader,
    )
    managers.append(manager)
    partial = tmp_path / ".downloads" / "test-q4.partial.gguf"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(MODEL_BYTES[:4])

    installed = manager.download_artifact("test-q4")

    assert observed == [MODEL_BYTES[:4]]
    assert installed.read_bytes() == MODEL_BYTES


def test_remove_rejects_while_cancelled_worker_is_still_running(tmp_path, managers):
    started = threading.Event()
    release = threading.Event()

    def downloader(_artifact, destination, _callback):
        destination.write_bytes(b"GGUF")
        started.set()
        release.wait(timeout=2)

    manager = LocalModelManager(tmp_path, artifacts=(_artifact(),), downloader=downloader)
    managers.append(manager)
    manager.request_download("test-q4")
    assert started.wait(timeout=2)

    status = manager.cancel_download("test-q4")
    assert status["status"] == "cancelling"
    with pytest.raises(RuntimeError, match="finish cancelling"):
        manager.remove_artifact("test-q4")

    release.set()


def test_import_copies_gguf_into_managed_store_and_persists(tmp_path, managers):
    manager = LocalModelManager(tmp_path, artifacts=())
    managers.append(manager)

    imported = manager.import_stream(
        io.BytesIO(MODEL_BYTES),
        "my-model.gguf",
        model_id="custom/model",
    )

    assert imported["status"] == "installed"
    assert imported["model_id"] == "custom/model"
    assert Path(imported["path"]).read_bytes() == MODEL_BYTES

    reloaded = LocalModelManager(tmp_path, artifacts=())
    managers.append(reloaded)
    assert reloaded.find_artifact("custom/model").id == imported["id"]

    removed = reloaded.remove_artifact(imported["id"])
    assert removed["status"] == "removed"
    assert not Path(imported["path"]).exists()
    with pytest.raises(KeyError, match="Unknown local model artifact"):
        reloaded.status(imported["id"])


@pytest.mark.parametrize("filename", ["../escape.gguf", r"..\escape.gguf", "model.bin"])
def test_import_rejects_unsafe_or_non_gguf_names(tmp_path, managers, filename):
    manager = LocalModelManager(tmp_path, artifacts=())
    managers.append(manager)

    with pytest.raises(ValueError, match="single .gguf"):
        manager.import_stream(io.BytesIO(MODEL_BYTES), filename)


def test_legacy_mlx_snapshot_uses_shared_manager_hook(tmp_path, managers):
    manager = LocalModelManager(tmp_path, artifacts=())
    managers.append(manager)
    destination = tmp_path / "mlx"

    with patch("huggingface_hub.snapshot_download") as download:
        manager.ensure_hugging_face_snapshot(
            "meta-llama/model",
            destination,
            token="secret",
            allow_patterns=["*.json", "*.safetensors"],
        )

    download.assert_called_once_with(
        repo_id="meta-llama/model",
        token="secret",
        local_dir=str(destination),
        allow_patterns=["*.json", "*.safetensors"],
    )


def test_llama_artifacts_are_not_offered_as_runnable_on_macos_arm64(tmp_path, managers):
    artifact = _artifact()
    assert local_artifact_supported(artifact, system="Darwin", machine="arm64") is False

    manager = LocalModelManager(
        tmp_path,
        artifacts=(artifact,),
        artifact_support=lambda _artifact: False,
    )
    managers.append(manager)

    assert manager.status(artifact.id)["supported"] is False
    with pytest.raises(ValueError, match="Select an MLX model"):
        manager.request_download(artifact.id)
    with pytest.raises(ValueError, match="Select an MLX model"):
        manager.import_stream(io.BytesIO(MODEL_BYTES), "model.gguf")


def test_curated_mlx_snapshot_is_pinned_and_gated():
    artifact = next(item for item in CURATED_LOCAL_ARTIFACTS if item.backend == "mlx_llama")

    assert artifact.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert artifact.revision and len(artifact.revision) == 40
    assert artifact.revision != "main"
    assert artifact.requires_auth is True
    assert "*.safetensors" in (artifact.allow_patterns or ())


def test_mlx_snapshot_is_managed_selected_and_removed(tmp_path, managers):
    progress = []

    def downloader(_artifact, destination, callback):
        (destination / "config.json").write_text("{}", encoding="utf-8")
        callback(1, 3)
        (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
        callback(2, 3)
        (destination / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        callback(3, 3)
        progress.append(destination)

    artifact = _mlx_artifact()
    manager = LocalModelManager(
        tmp_path,
        artifacts=(artifact,),
        downloader=downloader,
        artifact_support=lambda _artifact: True,
    )
    managers.append(manager)

    installed = manager.download_artifact(artifact.id)

    assert progress[0].name.endswith(".partial.snapshot")
    assert (installed / ".geist-artifact.json").is_file()
    assert manager.status(artifact.id)["progress_unit"] == "files"
    assert manager.status(artifact.id)["progress_completed"] == 3
    selected, selected_path = manager.require_installed("Test-MLX")
    assert selected == artifact
    assert selected_path == installed

    removed = manager.remove_artifact(artifact.id)
    assert removed["status"] == "not_installed"
    assert not installed.exists()


def test_mlx_snapshot_requires_completion_manifest(tmp_path, managers):
    artifact = _mlx_artifact()
    manager = LocalModelManager(
        tmp_path,
        artifacts=(artifact,),
        artifact_support=lambda _artifact: True,
    )
    managers.append(manager)
    target = tmp_path / "artifacts" / artifact.id / "snapshot"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="completion manifest"):
        manager.require_installed(artifact.id)


def test_mlx_snapshot_background_download_can_be_cancelled(tmp_path, managers):
    started = threading.Event()
    release = threading.Event()

    def downloader(_artifact, destination, callback):
        (destination / "config.json").write_text("{}", encoding="utf-8")
        callback(1, 3)
        started.set()
        release.wait(timeout=2)
        (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
        callback(2, 3)

    artifact = _mlx_artifact()
    manager = LocalModelManager(
        tmp_path,
        artifacts=(artifact,),
        downloader=downloader,
        artifact_support=lambda _artifact: True,
    )
    managers.append(manager)

    manager.request_download(artifact.id)
    assert started.wait(timeout=2)
    assert manager.cancel_download(artifact.id)["status"] == "cancelling"
    release.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        manager._futures[artifact.id].result(timeout=2)

    assert manager.status(artifact.id)["status"] == "cancelled"
    assert manager._partial_path(artifact).is_dir()


def test_mlx_snapshot_support_is_apple_silicon_only():
    artifact = _mlx_artifact()

    assert local_artifact_supported(artifact, system="Darwin", machine="arm64") is True
    assert local_artifact_supported(artifact, system="Windows", machine="AMD64") is False
    assert local_artifact_supported(artifact, system="Linux", machine="x86_64") is False


def test_snapshot_downloader_pins_revision_and_reports_file_progress(
    tmp_path, managers, monkeypatch
):
    artifact = _mlx_artifact()
    manager = LocalModelManager(
        tmp_path,
        artifacts=(artifact,),
        artifact_support=lambda _artifact: True,
    )
    managers.append(manager)
    observed = []
    captured = {}

    def snapshot_download(**kwargs):
        captured.update(kwargs)
        progress_type = kwargs["tqdm_class"]
        progress = progress_type(total=2)
        progress.update()
        progress.update()

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    manager._download_hugging_face_snapshot(
        artifact, tmp_path / "snapshot", lambda n, total: observed.append((n, total))
    )

    assert observed[-1] == (2, 2)
    assert captured["revision"] == artifact.revision
    assert captured["token"] == "secret"
    assert captured["allow_patterns"] == list(artifact.allow_patterns or ())
    assert captured["max_workers"] == 1
