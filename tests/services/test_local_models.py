"""Focused tests for Geist's managed local-model store."""

from __future__ import annotations

import errno
import hashlib
import io
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.local_models import (
    CURATED_LOCAL_ARTIFACTS,
    DOWNLOAD_DISK_RESERVE_BYTES,
    InsufficientStorageError,
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


def test_qwen3_8_uses_pinned_platform_artifacts(tmp_path: Path):
    artifacts = {
        item.backend: item
        for item in CURATED_LOCAL_ARTIFACTS
        if item.model_id == "Qwen/Qwen3.8-27B"
    }

    mlx = artifacts["mlx_llama"]
    assert mlx.id == "qwen3.8-27b-4bit-mlx"
    assert mlx.repo_id == "mlx-community/Qwen3.8-27B-4bit"
    assert mlx.revision == "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
    assert mlx.quantization == "4-bit"
    assert local_artifact_supported(mlx, system="Darwin", machine="arm64") is True
    assert local_artifact_supported(mlx, system="Linux", machine="x86_64") is False

    gguf = artifacts["llama_server"]
    assert gguf.id == "qwen3.8-27b-q4-k-m-gguf"
    assert gguf.repo_id == "ggml-org/Qwen3.8-27B-GGUF"
    assert gguf.revision == "0669b98607d47046c7c2b3f801011d54a08cfccf"
    assert gguf.filename == "Qwen3.8-27B-Q4_K_M.gguf"
    assert gguf.size_bytes == 18_973_870_432
    assert gguf.sha256 == "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
    assert gguf.quantization == "Q4_K_M"
    runtime = tmp_path / "runtime"
    server = runtime / "cpu" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"binary")
    server.chmod(0o755)
    assert (
        local_artifact_supported(
            gguf,
            system="Linux",
            machine="x86_64",
            environment={
                "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
                "GEIST_LLAMA_ACCELERATION": "cpu",
            },
        )
        is True
    )
    assert local_artifact_supported(gguf, system="Darwin", machine="arm64") is False


def test_linux_gguf_is_unsupported_without_a_llama_runtime(tmp_path: Path):
    gguf = next(
        artifact for artifact in CURATED_LOCAL_ARTIFACTS if artifact.backend == "llama_server"
    )

    assert (
        local_artifact_supported(
            gguf,
            system="Linux",
            machine="x86_64",
            environment={"GEIST_LLAMA_RUNTIME_ROOT": str(tmp_path / "missing")},
        )
        is False
    )


def test_model_lookup_prefers_the_platform_supported_artifact(tmp_path, managers):
    mlx = _mlx_artifact(model_id="Qwen/Qwen3.8-27B")
    gguf = _artifact(
        id="test-qwen-gguf",
        model_id="Qwen/Qwen3.8-27B",
        repo_id="test/qwen-gguf",
    )
    manager = LocalModelManager(
        tmp_path,
        artifacts=(mlx, gguf),
        artifact_support=lambda artifact: artifact.backend == "llama_server",
    )
    managers.append(manager)

    assert manager.find_artifact("Qwen/Qwen3.8-27B") == gguf


@pytest.fixture
def managers():
    active = []
    yield active
    for manager in active:
        manager.shutdown()


@pytest.fixture(autouse=True)
def make_artifact_manager_tests_platform_independent(monkeypatch):
    """Exercise manager behavior without inheriting the current host policy."""

    monkeypatch.setattr(
        "app.services.local_models.local_artifact_supported",
        lambda _artifact: True,
    )


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


def test_download_fails_before_queueing_when_model_store_is_too_small(tmp_path, managers):
    artifact = _artifact(size_bytes=16 * 1024**3)
    manager = LocalModelManager(tmp_path, artifacts=(artifact,))
    managers.append(manager)

    with (
        patch(
            "app.services.local_models.shutil.disk_usage",
            return_value=SimpleNamespace(free=512 * 1024**2),
        ),
        pytest.raises(InsufficientStorageError, match="needed; 512.0 MB available"),
    ):
        manager.request_download(artifact.id)

    status = manager.status(artifact.id)
    assert status["status"] == "not_installed"
    assert status["progress_unit"] == "bytes"
    assert status["progress_total"] == artifact.size_bytes
    assert artifact.id not in manager._futures


def test_capacity_check_accounts_for_resumable_partial_download(tmp_path, managers):
    artifact = _artifact(size_bytes=1024)
    manager = LocalModelManager(tmp_path, artifacts=(artifact,))
    managers.append(manager)
    partial = manager._partial_path(artifact)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * 768)

    with patch(
        "app.services.local_models.shutil.disk_usage",
        return_value=SimpleNamespace(free=DOWNLOAD_DISK_RESERVE_BYTES + 256),
    ):
        manager._require_download_capacity(artifact)


def test_stream_capacity_recheck_uses_model_store_filesystem(tmp_path, managers):
    artifact = _artifact(size_bytes=len(MODEL_BYTES))
    manager = LocalModelManager(tmp_path, artifacts=(artifact,))
    managers.append(manager)
    destination = manager._partial_path(artifact)
    destination.parent.mkdir(parents=True)
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-length": str(len(MODEL_BYTES))}
    response.iter_bytes.return_value = [MODEL_BYTES]
    stream = MagicMock()
    stream.__enter__.return_value = response
    client = MagicMock()
    client.__enter__.return_value = client
    client.stream.return_value = stream
    checked_paths = []

    def disk_usage(path):
        checked_paths.append(Path(path))
        return SimpleNamespace(free=1024**3)

    with (
        patch("app.services.local_models.httpx.Client", return_value=client),
        patch("app.services.local_models.shutil.disk_usage", side_effect=disk_usage),
    ):
        manager._download_hugging_face_artifact(
            artifact,
            destination,
            lambda _downloaded, _total: None,
        )

    assert checked_paths == [tmp_path]


def test_only_one_model_download_can_be_active(tmp_path, managers):
    started = threading.Event()
    release = threading.Event()

    def downloader(_artifact, destination, callback):
        started.set()
        release.wait(timeout=2)
        destination.write_bytes(MODEL_BYTES)
        callback(len(MODEL_BYTES), len(MODEL_BYTES))

    first = _artifact(id="first", model_id="test/first", filename="first.gguf")
    second = _artifact(id="second", model_id="test/second", filename="second.gguf")
    manager = LocalModelManager(
        tmp_path,
        artifacts=(first, second),
        downloader=downloader,
    )
    managers.append(manager)

    manager.request_download(first.id)
    assert started.wait(timeout=2)
    try:
        with pytest.raises(RuntimeError, match="Another model is already installing"):
            manager.request_download(second.id)
    finally:
        release.set()

    manager._futures[first.id].result(timeout=2)
    assert manager.status(second.id)["status"] == "not_installed"
    assert second.id not in manager._futures


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


def test_invalid_managed_artifact_is_not_installed_and_can_be_repaired(tmp_path, managers):
    invalid_target_seen = []

    def downloader(_artifact, destination, callback):
        invalid_target_seen.append(target.exists())
        destination.write_bytes(MODEL_BYTES)
        callback(len(MODEL_BYTES), len(MODEL_BYTES))

    manager = LocalModelManager(
        tmp_path,
        artifacts=(_artifact(),),
        downloader=downloader,
    )
    managers.append(manager)
    target = tmp_path / "artifacts" / "test-q4" / "test-q4.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"GGUF" + b"tampered!!")

    with pytest.raises(RuntimeError, match="not installed"):
        manager.require_installed("test-q4")

    status = manager.status("test-q4")
    assert status["status"] == "not_installed"
    assert status["path"] is None
    assert status["error"] is None

    manager.request_download("test-q4")
    manager._futures["test-q4"].result(timeout=5)

    assert invalid_target_seen == [False]
    assert manager.status("test-q4")["status"] == "installed"
    assert target.read_bytes() == MODEL_BYTES


def test_status_reconciles_installed_state_when_files_are_missing(tmp_path, managers):
    artifact = _artifact()
    manager = LocalModelManager(tmp_path, artifacts=(artifact,))
    managers.append(manager)
    manager._states[artifact.id] = {
        "status": "installed",
        "bytes_downloaded": len(MODEL_BYTES),
        "total_bytes": len(MODEL_BYTES),
        "path": str(tmp_path / "artifacts" / artifact.id / artifact.filename),
        "error": None,
    }
    manager._save_index_locked()

    status = manager.status(artifact.id)

    assert status["status"] == "not_installed"
    assert status["bytes_downloaded"] == 0
    assert status["progress_completed"] == 0
    assert status["path"] is None
    assert status["error"] is None
    assert manager.status(artifact.id) == status

    reloaded = LocalModelManager(tmp_path, artifacts=(artifact,))
    managers.append(reloaded)
    assert reloaded.status(artifact.id)["status"] == "not_installed"


@pytest.mark.parametrize(
    ("persisted_status", "expected_status", "expected_error"),
    [
        ("queued", "failed", "Install was interrupted."),
        ("downloading", "failed", "Install was interrupted."),
        ("cancelling", "cancelled", None),
    ],
)
def test_restart_reconciles_incomplete_operations(
    tmp_path,
    managers,
    persisted_status,
    expected_status,
    expected_error,
):
    artifact = _artifact()
    manager = LocalModelManager(tmp_path, artifacts=(artifact,))
    managers.append(manager)
    manager._states[artifact.id] = {
        "status": persisted_status,
        "bytes_downloaded": 4,
        "total_bytes": len(MODEL_BYTES),
        "path": None,
        "error": None,
    }
    manager._save_index_locked()

    reloaded = LocalModelManager(tmp_path, artifacts=(artifact,))
    managers.append(reloaded)
    status = reloaded.status(artifact.id)

    assert status["status"] == expected_status
    assert status["error"] == expected_error


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


def test_out_of_space_download_failure_has_a_clear_message(tmp_path, managers):
    def downloader(_artifact, _destination, _callback):
        raise OSError(errno.ENOSPC, "No space left on device")

    manager = LocalModelManager(
        tmp_path,
        artifacts=(_artifact(),),
        downloader=downloader,
    )
    managers.append(manager)

    with pytest.raises(OSError):
        manager.download_artifact("test-q4")

    status = manager.status("test-q4")
    assert status["status"] == "failed"
    assert status["error"] == "Not enough space to finish installing this model."


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


def test_import_reports_out_of_space_clearly(tmp_path, managers):
    manager = LocalModelManager(tmp_path, artifacts=())
    managers.append(manager)

    with (
        patch("pathlib.Path.open", side_effect=OSError(errno.ENOSPC, "No space left")),
        pytest.raises(InsufficientStorageError, match="Not enough space to import"),
    ):
        manager.import_stream(io.BytesIO(MODEL_BYTES), "model.gguf")


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
    with pytest.raises(ValueError, match="not supported by an available local runtime"):
        manager.request_download(artifact.id)
    with pytest.raises(ValueError, match="not supported by an available local runtime"):
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

    with pytest.raises(RuntimeError, match="not installed"):
        manager.require_installed(artifact.id)

    status = manager.status(artifact.id)
    assert status["status"] == "not_installed"
    assert status["error"] is None


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
        progress = progress_type(total=2, name="huggingface_hub.snapshot_download")
        progress.update()
        progress.update()

    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
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
