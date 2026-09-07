"""Managed local-model artifacts shared by MLX and llama.cpp runners.

The model store deliberately owns downloads and imports instead of allowing an
inference runtime to fetch arbitrary content.  Curated artifacts are pinned to
an immutable upstream revision and, when available, an expected SHA-256.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import httpx
from huggingface_hub import hf_hub_url

from agents.architectures.llama_server_process import llama_server_runtime_available
from app.runtime_config import default_data_dir


logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
COPY_CHUNK_SIZE = 4 * 1024 * 1024
DOWNLOAD_DISK_RESERVE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class LocalModelArtifact:
    """One concrete, runnable model artifact."""

    id: str
    model_id: str
    display_name: str
    format: str
    backend: str
    filename: str
    repo_id: str | None = None
    revision: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    quantization: str | None = None
    license: str | None = None
    source: str = "curated"
    supports_tool_calling: bool = False
    allow_patterns: tuple[str, ...] | None = None
    requires_auth: bool = False


# Official Qwen GGUF metadata verified for the pinned artifact.  Keeping this
# here makes the download visible to both the API and the llama-server runner.
CURATED_LOCAL_ARTIFACTS: tuple[LocalModelArtifact, ...] = (
    LocalModelArtifact(
        id="qwen3-4b-q4-k-m",
        model_id="Qwen/Qwen3-4B",
        display_name="Qwen3 4B Q4_K_M (GGUF)",
        format="gguf",
        backend="llama_server",
        repo_id="Qwen/Qwen3-4B-GGUF",
        revision="a9a60d009fa7ff9606305047c2bf77ac25dbec49",
        filename="Qwen3-4B-Q4_K_M.gguf",
        size_bytes=2_497_280_256,
        sha256="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        quantization="Q4_K_M",
        license="Apache-2.0",
        supports_tool_calling=True,
    ),
    LocalModelArtifact(
        id="meta-llama-3.1-8b-instruct-mlx",
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        display_name="Meta Llama 3.1 8B Instruct (MLX)",
        format="snapshot",
        backend="mlx_llama",
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # Current immutable commit advertised by the official gated repository.
        revision="0e9e39f249a16976918f6564b8830bc894c89659",
        filename="snapshot",
        license="llama3.1",
        requires_auth=True,
        allow_patterns=(
            "*.json",
            "*.model",
            "*.safetensors",
            "*.txt",
            "LICENSE",
            "USE_POLICY.md",
            "README.md",
        ),
    ),
    LocalModelArtifact(
        id="qwen3.8-27b-4bit-mlx",
        model_id="Qwen/Qwen3.8-27B",
        display_name="Qwen 3.8 27B 4-bit (MLX)",
        format="snapshot",
        backend="mlx_llama",
        repo_id="mlx-community/Qwen3.8-27B-4bit",
        revision="3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        filename="snapshot",
        size_bytes=16_074_530_674,
        quantization="4-bit",
        license="Apache-2.0",
        supports_tool_calling=True,
        allow_patterns=(
            "*.json",
            "*.jinja",
            "*.safetensors",
        ),
    ),
    LocalModelArtifact(
        id="qwen3.8-27b-q4-k-m-gguf",
        model_id="Qwen/Qwen3.8-27B",
        display_name="Qwen 3.8 27B Q4_K_M (GGUF)",
        format="gguf",
        backend="llama_server",
        repo_id="ggml-org/Qwen3.8-27B-GGUF",
        revision="0669b98607d47046c7c2b3f801011d54a08cfccf",
        filename="Qwen3.8-27B-Q4_K_M.gguf",
        size_bytes=18_973_870_432,
        sha256="31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34",
        quantization="Q4_K_M",
        license="Apache-2.0",
        supports_tool_calling=True,
    ),
)


DownloadFunction = Callable[[LocalModelArtifact, Path, Callable[[int, int | None], None]], None]
ArtifactSupportFunction = Callable[[LocalModelArtifact], bool]


class ModelDownloadCancelledError(RuntimeError):
    """Raised inside a download worker after a user cancellation request."""


class InsufficientStorageError(RuntimeError):
    """Raised before a model download when its storage target is too small."""


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            digits = 0 if unit == "B" else 1
            return f"{value:.{digits}f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def default_model_home(environment: dict[str, str] | None = None) -> Path:
    """Return a user-writable model root with an explicit override for hosts."""

    env = environment if environment is not None else os.environ
    configured = env.get("GEIST_MODEL_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    data_dir = env.get("GEIST_DATA_DIR")
    if data_dir:
        return (Path(data_dir).expanduser() / "models").resolve()

    return default_data_dir(environ=env) / "models"


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not normalized or normalized in {".", ".."}:
        raise ValueError("Artifact identifier does not contain a safe path component")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(COPY_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_gguf(path: Path) -> None:
    if path.suffix.lower() != ".gguf":
        raise ValueError("Local model artifact must use the .gguf extension")
    with path.open("rb") as model_file:
        if model_file.read(4) != GGUF_MAGIC:
            raise ValueError("Local model artifact does not contain a GGUF header")


def local_artifact_supported(
    artifact: LocalModelArtifact,
    *,
    system: str | None = None,
    machine: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether this distribution has a runnable backend for an artifact."""

    platform_name = (system or platform.system()).lower()
    architecture = (machine or platform.machine()).lower()
    if artifact.backend == "llama_server":
        platform_supported = (platform_name, architecture) in {
            ("windows", "amd64"),
            ("windows", "x86_64"),
            ("linux", "amd64"),
            ("linux", "x86_64"),
        }
        return platform_supported and llama_server_runtime_available(
            environment,
            system=platform_name,
        )
    if artifact.backend == "mlx_llama":
        return platform_name == "darwin" and architecture in {"arm64", "aarch64"}
    return True


class LocalModelManager:
    """Own local artifacts, installation state, downloads, and imports."""

    def __init__(
        self,
        model_home: str | Path | None = None,
        *,
        artifacts: tuple[LocalModelArtifact, ...] = CURATED_LOCAL_ARTIFACTS,
        downloader: DownloadFunction | None = None,
        artifact_support: ArtifactSupportFunction | None = None,
    ) -> None:
        self.model_home = Path(model_home) if model_home is not None else default_model_home()
        self.artifacts_dir = self.model_home / "artifacts"
        self.downloads_dir = self.model_home / ".downloads"
        self.index_path = self.model_home / "index.json"
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geist-model")
        self._futures: dict[str, Future[Path]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._verified_files: dict[str, tuple[int, int, str]] = {}
        self._downloader = downloader
        self._artifact_support = artifact_support or local_artifact_supported
        self._artifacts = {artifact.id: artifact for artifact in artifacts}
        self._states: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError):
            logger.warning("Ignoring invalid local-model index at %s", self.index_path)
            return

        for raw in payload.get("custom_artifacts", []):
            try:
                artifact = LocalModelArtifact(**raw)
            except (TypeError, ValueError):
                logger.warning("Ignoring malformed custom local-model artifact")
                continue
            self._artifacts[artifact.id] = artifact
        states = payload.get("states")
        if isinstance(states, dict):
            self._states = {str(key): dict(value) for key, value in states.items()}
            for state in self._states.values():
                if state.get("status") in {"queued", "downloading", "importing"}:
                    state["status"] = "failed"
                    state["error"] = "Install was interrupted."
                elif state.get("status") == "cancelling":
                    state["status"] = "cancelled"
                    state["error"] = None

    def _save_index_locked(self) -> None:
        self.model_home.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "custom_artifacts": [
                asdict(artifact)
                for artifact in self._artifacts.values()
                if artifact.source == "imported"
            ],
            "states": self._states,
        }
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.index_path)

    def _artifact_path(self, artifact: LocalModelArtifact) -> Path:
        filename = Path(artifact.filename).name
        if artifact.format == "snapshot":
            if filename != "snapshot":
                raise ValueError(f"Unsafe snapshot directory for artifact {artifact.id}")
            return self.artifacts_dir / _safe_component(artifact.id) / filename
        if (
            filename != artifact.filename
            or "/" in filename
            or "\\" in filename
            or not filename.lower().endswith(".gguf")
        ):
            raise ValueError(f"Unsafe GGUF filename for artifact {artifact.id}")
        return self.artifacts_dir / _safe_component(artifact.id) / filename

    def _partial_path(self, artifact: LocalModelArtifact) -> Path:
        suffix = "partial.snapshot" if artifact.format == "snapshot" else "partial.gguf"
        return self.downloads_dir / f"{_safe_component(artifact.id)}.{suffix}"

    @staticmethod
    def _existing_parent(path: Path) -> Path:
        candidate = path
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    def _require_download_capacity(self, artifact: LocalModelArtifact) -> None:
        if artifact.size_bytes is None:
            return
        partial = self._partial_path(artifact)
        downloaded = self._payload_size(partial) if partial.exists() else 0
        remaining = max(0, artifact.size_bytes - downloaded)
        required = remaining + DOWNLOAD_DISK_RESERVE_BYTES
        storage_root = self._existing_parent(self.model_home)
        available = shutil.disk_usage(storage_root).free
        if available < required:
            raise InsufficientStorageError(
                f"Not enough space to install {artifact.display_name}. "
                f"{_format_size(required)} needed; {_format_size(available)} available."
            )

    @staticmethod
    def _artifact_exists(artifact: LocalModelArtifact, path: Path) -> bool:
        return path.is_dir() if artifact.format == "snapshot" else path.is_file()

    def _remove_artifact_target(self, artifact: LocalModelArtifact) -> None:
        target = self._artifact_path(artifact)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        self._verified_files.pop(artifact.id, None)
        with suppress(OSError):
            target.parent.rmdir()

    def _installed_artifact_error(
        self,
        artifact: LocalModelArtifact,
        target: Path,
    ) -> str | None:
        """Quickly detect persisted installation state that no longer matches disk."""

        if not self._artifact_exists(artifact, target):
            return f"Installed model files are missing from {target}."
        try:
            if artifact.format == "snapshot":
                self._verify_snapshot(artifact, target)
            else:
                _validate_gguf(target)
                if artifact.size_bytes is not None and target.stat().st_size != artifact.size_bytes:
                    return (
                        f"Local GGUF size {target.stat().st_size} does not match expected "
                        f"{artifact.size_bytes}"
                    )
        except (OSError, ValueError) as error:
            return str(error)
        return None

    @staticmethod
    def _mark_not_installed(
        state: dict[str, Any],
        artifact: LocalModelArtifact,
    ) -> None:
        progress_uses_bytes = artifact.size_bytes is not None
        state.update(
            status="not_installed",
            bytes_downloaded=0,
            total_bytes=artifact.size_bytes,
            progress_unit="bytes" if progress_uses_bytes else "files",
            progress_completed=0,
            progress_total=artifact.size_bytes if progress_uses_bytes else None,
            sha256=None,
            path=None,
            error=None,
        )

    def _state_for_locked(self, artifact: LocalModelArtifact) -> dict[str, Any]:
        state = self._states.setdefault(
            artifact.id,
            {
                "status": "not_installed",
                "bytes_downloaded": 0,
                "total_bytes": artifact.size_bytes,
                "sha256": None,
                "error": None,
                "path": None,
            },
        )
        progress_uses_bytes = artifact.format != "snapshot" or artifact.size_bytes is not None
        state.setdefault("progress_unit", "bytes" if progress_uses_bytes else "files")
        state.setdefault("progress_completed", state.get("bytes_downloaded", 0))
        state.setdefault("progress_total", state.get("total_bytes"))
        target = self._artifact_path(artifact)
        if state.get("status") == "installed":
            mismatch = self._installed_artifact_error(artifact, target)
            if mismatch:
                self._mark_not_installed(state, artifact)
            else:
                state.update(path=str(target), error=None)
            return state
        if self._artifact_exists(artifact, target) and state.get("status") != "installed":
            try:
                checksum = self._verify_artifact_file(artifact, target)
            except (OSError, ValueError):
                self._mark_not_installed(state, artifact)
            else:
                installed_size = self._payload_size(target)
                state.update(
                    status="installed",
                    bytes_downloaded=installed_size,
                    total_bytes=installed_size,
                    progress_completed=(1 if artifact.format == "snapshot" else installed_size),
                    progress_total=(1 if artifact.format == "snapshot" else installed_size),
                    sha256=(None if artifact.format == "snapshot" else checksum),
                    path=str(target),
                    error=None,
                )
        return state

    @staticmethod
    def _payload_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _verify_artifact_file(self, artifact: LocalModelArtifact, path: Path) -> str:
        """Validate the exact managed bytes before exposing them to inference."""

        if artifact.format == "snapshot":
            return self._verify_snapshot(artifact, path)

        _validate_gguf(path)
        stats = path.stat()
        if artifact.size_bytes is not None and stats.st_size != artifact.size_bytes:
            raise ValueError(
                f"Local GGUF size {stats.st_size} does not match expected {artifact.size_bytes}"
            )

        cached = self._verified_files.get(artifact.id)
        fingerprint = (stats.st_mtime_ns, stats.st_size)
        if cached is not None and cached[:2] == fingerprint:
            checksum = cached[2]
        else:
            checksum = _sha256_file(path)
            self._verified_files[artifact.id] = (*fingerprint, checksum)
        if artifact.sha256 is not None and checksum.casefold() != artifact.sha256.casefold():
            self._verified_files.pop(artifact.id, None)
            raise ValueError("Local GGUF SHA-256 does not match the managed artifact")
        return checksum

    @staticmethod
    def _verify_snapshot(artifact: LocalModelArtifact, path: Path) -> str:
        if not path.is_dir():
            raise ValueError("Managed MLX snapshot is not a directory")
        manifest_path = path / ".geist-artifact.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise ValueError("Managed MLX snapshot is missing its completion manifest") from error
        if manifest.get("artifact_id") != artifact.id:
            raise ValueError("Managed MLX snapshot belongs to a different artifact")
        if artifact.revision and manifest.get("revision") != artifact.revision:
            raise ValueError("Managed MLX snapshot revision does not match the curated artifact")
        if not (path / "config.json").is_file():
            raise ValueError("Managed MLX snapshot is missing config.json")
        if not ((path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()):
            raise ValueError("Managed MLX snapshot is missing tokenizer files")
        if not any(path.glob("*.safetensors")):
            raise ValueError("Managed MLX snapshot is missing safetensors weights")
        return str(manifest.get("revision") or artifact.revision or "snapshot")

    def list_artifacts(self, model_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            state_changed = False
            for artifact in self._artifacts.values():
                if model_id is not None and artifact.model_id != model_id:
                    continue
                previous_state = dict(self._states.get(artifact.id, {}))
                state = self._state_for_locked(artifact)
                state_changed = state_changed or (
                    state != previous_state
                    and (bool(previous_state) or state.get("status") != "not_installed")
                )
                result.append(
                    {
                        **asdict(artifact),
                        **dict(state),
                        "supported": self._artifact_support(artifact),
                    }
                )
            if state_changed:
                try:
                    self._save_index_locked()
                except OSError:
                    logger.warning("Could not persist reconciled local-model state", exc_info=True)
            return result

    def get_artifact(self, artifact_id: str) -> LocalModelArtifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as error:
            raise KeyError(f"Unknown local model artifact: {artifact_id}") from error

    def status(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self.get_artifact(artifact_id)
            previous_state = dict(self._states.get(artifact.id, {}))
            state = self._state_for_locked(artifact)
            if state != previous_state and (
                previous_state or state.get("status") != "not_installed"
            ):
                try:
                    self._save_index_locked()
                except OSError:
                    logger.warning("Could not persist reconciled local-model state", exc_info=True)
            return {
                **asdict(artifact),
                **dict(state),
                "supported": self._artifact_support(artifact),
            }

    def find_artifact(self, model_or_artifact_id: str) -> LocalModelArtifact:
        with self._lock:
            direct = self._artifacts.get(model_or_artifact_id)
            if direct is not None:
                return direct
            matches = [
                artifact
                for artifact in self._artifacts.values()
                if artifact.model_id.casefold() == model_or_artifact_id.casefold()
                or artifact.model_id.rsplit("/", 1)[-1].casefold()
                == model_or_artifact_id.casefold()
            ]
            if not matches:
                raise KeyError(
                    f"No managed local artifact is available for {model_or_artifact_id}. "
                    "Download a supported curated model first."
                )
            supported = [artifact for artifact in matches if self._artifact_support(artifact)]
            installed = [
                artifact
                for artifact in matches
                if self._state_for_locked(artifact).get("status") == "installed"
            ]
            installed_supported = [artifact for artifact in installed if artifact in supported]
            if installed_supported:
                return installed_supported[0]
            if supported:
                return supported[0]
            return (installed or matches)[0]

    def request_download(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self.get_artifact(artifact_id)
            self._require_supported(artifact)
            state = self._state_for_locked(artifact)
            if state.get("status") == "installed":
                return {**asdict(artifact), **dict(state)}
            another_download = next(
                (
                    active_artifact_id
                    for active_artifact_id, future in self._futures.items()
                    if active_artifact_id != artifact_id and not future.done()
                ),
                None,
            )
            if another_download is not None:
                raise RuntimeError("Another model is already installing.")
            target = self._artifact_path(artifact)
            if self._artifact_exists(artifact, target):
                try:
                    self._remove_artifact_target(artifact)
                except OSError as error:
                    raise ValueError(f"Could not clear invalid model files: {error}") from error
            self._require_download_capacity(artifact)
            future = self._futures.get(artifact_id)
            if future is None or future.done():
                self._cancel_events.setdefault(artifact_id, threading.Event()).clear()
                progress_uses_bytes = artifact.size_bytes is not None
                state.update(
                    status="queued",
                    progress_unit="bytes" if progress_uses_bytes else "files",
                    progress_completed=(
                        state.get("bytes_downloaded", 0) if progress_uses_bytes else 0
                    ),
                    progress_total=artifact.size_bytes if progress_uses_bytes else None,
                    error=None,
                )
                self._save_index_locked()
                self._futures[artifact_id] = self._executor.submit(
                    self.download_artifact, artifact_id
                )
            return {**asdict(artifact), **dict(state)}

    def cancel_download(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self.get_artifact(artifact_id)
            state = self._state_for_locked(artifact)
            if state.get("status") not in {"queued", "downloading"}:
                return {**asdict(artifact), **dict(state)}
            self._cancel_events.setdefault(artifact_id, threading.Event()).set()
            future = self._futures.get(artifact_id)
            if future is not None:
                future.cancel()
            state.update(
                status=("cancelling" if future is not None and not future.done() else "cancelled"),
                error=None,
            )
            self._save_index_locked()
            return {**asdict(artifact), **dict(state)}

    def download_artifact(self, artifact_id: str) -> Path:
        artifact = self.get_artifact(artifact_id)
        self._require_supported(artifact)
        if artifact.format == "snapshot":
            return self._download_snapshot_artifact(artifact)
        target = self._artifact_path(artifact)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._partial_path(artifact)
        if (
            temporary.is_file()
            and artifact.size_bytes is not None
            and temporary.stat().st_size > artifact.size_bytes
        ):
            temporary.unlink()
        partial_size = temporary.stat().st_size if temporary.is_file() else 0
        with self._lock:
            state = self._state_for_locked(artifact)
            if state.get("status") == "installed" and target.is_file():
                return target
            state.update(
                status="downloading",
                bytes_downloaded=partial_size,
                total_bytes=artifact.size_bytes,
                progress_unit="bytes",
                progress_completed=partial_size,
                progress_total=artifact.size_bytes,
                error=None,
                path=None,
            )
            self._save_index_locked()

        download_complete = False

        def progress(downloaded: int, total: int | None) -> None:
            with self._lock:
                cancel_event = self._cancel_events.setdefault(artifact.id, threading.Event())
                if cancel_event.is_set():
                    raise ModelDownloadCancelledError("Model download was cancelled")
                state = self._state_for_locked(artifact)
                state["bytes_downloaded"] = downloaded
                state["progress_completed"] = downloaded
                if total is not None:
                    state["total_bytes"] = total
                    state["progress_total"] = total
                # Persist progress at useful checkpoints without rewriting the
                # index for every small network chunk.
                if downloaded == total or downloaded % (16 * 1024 * 1024) < COPY_CHUNK_SIZE:
                    self._save_index_locked()

        try:
            self._require_download_capacity(artifact)
            progress(partial_size, artifact.size_bytes)
            if artifact.size_bytes is None or partial_size != artifact.size_bytes:
                self._run_downloader(artifact, temporary, progress)
            if self._cancel_events.setdefault(artifact.id, threading.Event()).is_set():
                raise ModelDownloadCancelledError("Model download was cancelled")
            download_complete = True
            _validate_gguf(temporary)
            actual_size = temporary.stat().st_size
            if artifact.size_bytes is not None and actual_size != artifact.size_bytes:
                raise ValueError(
                    f"Downloaded size {actual_size} does not match expected {artifact.size_bytes}"
                )
            actual_sha256 = _sha256_file(temporary)
            if (
                artifact.sha256 is not None
                and actual_sha256.casefold() != artifact.sha256.casefold()
            ):
                raise ValueError("Downloaded GGUF SHA-256 does not match the curated artifact")

            with self._lock:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, target)
                target_stats = target.stat()
                self._verified_files[artifact.id] = (
                    target_stats.st_mtime_ns,
                    target_stats.st_size,
                    actual_sha256,
                )
                state = self._states[artifact.id]
                state.update(
                    status="installed",
                    bytes_downloaded=actual_size,
                    total_bytes=actual_size,
                    progress_unit="bytes",
                    progress_completed=actual_size,
                    progress_total=actual_size,
                    sha256=actual_sha256,
                    path=str(target),
                    error=None,
                )
                self._save_index_locked()
            return target
        except Exception as error:
            with self._lock:
                state = self._state_for_locked(artifact)
                cancelled = isinstance(error, ModelDownloadCancelledError)
                state.update(
                    status="cancelled" if cancelled else "failed",
                    bytes_downloaded=(temporary.stat().st_size if temporary.is_file() else 0),
                    error=None if cancelled else self._download_error_message(error),
                    path=None,
                )
                self._save_index_locked()
            # Network/interruption failures intentionally retain the partial so
            # the next curated download can resume with an HTTP Range request.
            # Completed-but-invalid artifacts are never retained.
            if download_complete:
                temporary.unlink(missing_ok=True)
            raise

    def _download_snapshot_artifact(self, artifact: LocalModelArtifact) -> Path:
        target = self._artifact_path(artifact)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._partial_path(artifact)
        temporary.mkdir(parents=True, exist_ok=True)
        partial_size = self._payload_size(temporary)
        with self._lock:
            state = self._state_for_locked(artifact)
            if state.get("status") == "installed" and target.is_dir():
                return target
            state.update(
                status="downloading",
                bytes_downloaded=partial_size,
                total_bytes=artifact.size_bytes,
                progress_unit="bytes" if artifact.size_bytes is not None else "files",
                progress_completed=partial_size if artifact.size_bytes is not None else 0,
                progress_total=artifact.size_bytes,
                error=None,
                path=None,
            )
            self._save_index_locked()

        def progress(completed: int, total: int | None) -> None:
            with self._lock:
                cancel_event = self._cancel_events.setdefault(artifact.id, threading.Event())
                if cancel_event.is_set():
                    raise ModelDownloadCancelledError("Model download was cancelled")
                state = self._state_for_locked(artifact)
                downloaded = self._payload_size(temporary)
                if artifact.size_bytes is not None:
                    state.update(
                        bytes_downloaded=downloaded,
                        total_bytes=artifact.size_bytes,
                        progress_unit="bytes",
                        progress_completed=downloaded,
                        progress_total=artifact.size_bytes,
                    )
                else:
                    state.update(
                        bytes_downloaded=downloaded,
                        progress_unit="files",
                        progress_completed=completed,
                        progress_total=total,
                    )
                self._save_index_locked()

        completed = False
        try:
            self._require_download_capacity(artifact)
            progress(0, None)
            self._run_downloader(artifact, temporary, progress)
            if self._cancel_events.setdefault(artifact.id, threading.Event()).is_set():
                raise ModelDownloadCancelledError("Model download was cancelled")
            completed = True
            self._write_snapshot_manifest(artifact, temporary)
            self._verify_snapshot(artifact, temporary)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
            installed_size = self._payload_size(target)
            with self._lock:
                state = self._states[artifact.id]
                final_total = state.get("progress_total") or state.get("progress_completed") or 1
                progress_uses_bytes = artifact.size_bytes is not None
                state.update(
                    status="installed",
                    bytes_downloaded=installed_size,
                    total_bytes=installed_size,
                    progress_unit="bytes" if progress_uses_bytes else "files",
                    progress_completed=installed_size if progress_uses_bytes else final_total,
                    progress_total=installed_size if progress_uses_bytes else final_total,
                    sha256=None,
                    path=str(target),
                    error=None,
                )
                self._save_index_locked()
            return target
        except Exception as error:
            with self._lock:
                state = self._state_for_locked(artifact)
                cancelled = isinstance(error, ModelDownloadCancelledError)
                state.update(
                    status="cancelled" if cancelled else "failed",
                    bytes_downloaded=(self._payload_size(temporary) if temporary.exists() else 0),
                    error=None if cancelled else self._snapshot_error_message(artifact, error),
                    path=None,
                )
                self._save_index_locked()
            if completed:
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    @staticmethod
    def _write_snapshot_manifest(artifact: LocalModelArtifact, target: Path) -> None:
        manifest = {
            "version": 1,
            "artifact_id": artifact.id,
            "repo_id": artifact.repo_id,
            "revision": artifact.revision,
        }
        manifest_path = target / ".geist-artifact.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, manifest_path)

    @staticmethod
    def _snapshot_error_message(artifact: LocalModelArtifact, error: Exception) -> str:
        message = LocalModelManager._download_error_message(error)
        if artifact.requires_auth and any(
            marker in message.casefold()
            for marker in ("401", "403", "gated", "unauthorized", "forbidden")
        ):
            return (
                "This gated model requires accepted Hugging Face access and an HF_TOKEN or "
                "HUGGING_FACE_HUB_TOKEN. " + message
            )
        return message

    @staticmethod
    def _download_error_message(error: Exception) -> str:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            return "Not enough space to finish installing this model."
        return str(error)

    def remove_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Remove managed weights without following paths outside the model store."""

        with self._lock:
            artifact = self.get_artifact(artifact_id)
            state = self._state_for_locked(artifact)
            if state.get("status") in {"queued", "downloading"}:
                raise RuntimeError("Cancel the model download before removing the artifact")
            future = self._futures.get(artifact_id)
            if future is not None and not future.done():
                raise RuntimeError("Wait for the model download to finish cancelling")

            self._remove_artifact_target(artifact)

            partial = self._partial_path(artifact)
            if partial.is_dir():
                shutil.rmtree(partial)
            else:
                partial.unlink(missing_ok=True)
            self._cancel_events.pop(artifact.id, None)
            self._futures.pop(artifact.id, None)

            if artifact.source == "imported":
                self._artifacts.pop(artifact.id, None)
                self._states.pop(artifact.id, None)
                result = {**asdict(artifact), "status": "removed"}
            else:
                self._mark_not_installed(state, artifact)
                result = {**asdict(artifact), **dict(state)}
            self._save_index_locked()
            return result

    def ensure_artifact(self, model_or_artifact_id: str) -> tuple[LocalModelArtifact, Path]:
        artifact = self.find_artifact(model_or_artifact_id)
        return artifact, self.download_artifact(artifact.id)

    def require_installed(self, model_or_artifact_id: str) -> tuple[LocalModelArtifact, Path]:
        """Resolve an installed artifact without silently starting a large download."""

        artifact = self.find_artifact(model_or_artifact_id)
        self._require_supported(artifact)
        status = self.status(artifact.id)
        path_value = status.get("path")
        if status.get("status") != "installed" or not path_value:
            if status.get("error"):
                raise RuntimeError(
                    f"Installed local model {artifact.display_name} failed verification: "
                    f"{status['error']}"
                )
            raise RuntimeError(
                f"Local model {artifact.display_name} is not installed. "
                f"Download artifact {artifact.id} explicitly from the Models page or "
                "local-model download API before starting inference."
            )
        path = Path(str(path_value))
        try:
            checksum = self._verify_artifact_file(artifact, path)
        except (OSError, ValueError) as error:
            with self._lock:
                state = self._state_for_locked(artifact)
                self._mark_not_installed(state, artifact)
                self._save_index_locked()
            raise RuntimeError(
                f"Installed local model {artifact.display_name} failed verification: {error}"
            ) from error
        with self._lock:
            state = self._state_for_locked(artifact)
            if artifact.format != "snapshot" and state.get("sha256") != checksum:
                state["sha256"] = checksum
                self._save_index_locked()
        return artifact, path

    def import_file(
        self,
        source_path: str | Path,
        *,
        model_id: str | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Imported GGUF path must be a regular file")
        with source.open("rb") as model_file:
            return self.import_stream(
                model_file,
                source.name,
                model_id=model_id,
                display_name=display_name,
            )

    def import_stream(
        self,
        source: BinaryIO,
        filename: str,
        *,
        model_id: str | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        safe_filename = Path(filename).name
        if (
            safe_filename != filename
            or "/" in filename
            or "\\" in filename
            or not safe_filename.lower().endswith(".gguf")
        ):
            raise ValueError("Imported model must be a single .gguf file")
        imported_backend = LocalModelArtifact(
            id="import-check",
            model_id=model_id or "imported",
            display_name=display_name or "Imported GGUF",
            format="gguf",
            backend="llama_server",
            filename=safe_filename,
            source="imported",
        )
        self._require_supported(imported_backend)

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.downloads_dir / f"import-{time.time_ns()}.partial.gguf"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as destination:
                while True:
                    chunk = source.read(COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            _validate_gguf(temporary)
            checksum = digest.hexdigest()
            artifact_id = f"imported-{checksum[:16]}"
            normalized_model_id = model_id or artifact_id
            artifact = LocalModelArtifact(
                id=artifact_id,
                model_id=normalized_model_id,
                display_name=display_name or Path(safe_filename).stem,
                format="gguf",
                backend="llama_server",
                filename=safe_filename,
                sha256=checksum,
                size_bytes=size,
                source="imported",
            )
            target = self._artifact_path(artifact)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                temporary.unlink()
            else:
                os.replace(temporary, target)
            target_stats = target.stat()
            self._verified_files[artifact.id] = (
                target_stats.st_mtime_ns,
                target_stats.st_size,
                checksum,
            )
            with self._lock:
                self._artifacts[artifact.id] = artifact
                self._states[artifact.id] = {
                    "status": "installed",
                    "bytes_downloaded": size,
                    "total_bytes": size,
                    "sha256": checksum,
                    "path": str(target),
                    "error": None,
                }
                self._save_index_locked()
            return self.status(artifact.id)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                raise InsufficientStorageError("Not enough space to import this model.") from error
            raise

    def _require_supported(self, artifact: LocalModelArtifact) -> None:
        if not self._artifact_support(artifact):
            raise ValueError(
                f"{artifact.display_name} is not supported by an available local runtime."
            )

    def ensure_hugging_face_snapshot(
        self,
        model_id: str,
        destination: str | Path,
        *,
        allow_patterns: list[str] | None = None,
        token: str | None = None,
    ) -> Path:
        """Shared blocking snapshot path used by the legacy MLX first load."""

        from huggingface_hub import snapshot_download

        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            token=token,
            local_dir=str(target),
            allow_patterns=allow_patterns,
        )
        return target

    def _run_downloader(
        self,
        artifact: LocalModelArtifact,
        destination: Path,
        progress: Callable[[int, int | None], None],
    ) -> None:
        if self._downloader is not None:
            self._downloader(artifact, destination, progress)
        elif artifact.format == "snapshot":
            self._download_hugging_face_snapshot(artifact, destination, progress)
        else:
            self._download_hugging_face_artifact(artifact, destination, progress)

    def _download_hugging_face_snapshot(
        self,
        artifact: LocalModelArtifact,
        destination: Path,
        progress: Callable[[int, int | None], None],
    ) -> None:
        if not artifact.repo_id:
            raise ValueError(f"Artifact {artifact.id} does not have a snapshot repository")

        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm

        token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

        class SnapshotProgress(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.pop("name", None)
                kwargs.setdefault("disable", True)
                super().__init__(*args, **kwargs)
                progress(int(self.n), int(self.total) if self.total is not None else None)

            def update(self, n=1):
                if self.disable:
                    self.n += n
                    result = None
                else:
                    result = super().update(n)
                progress(int(self.n), int(self.total) if self.total is not None else None)
                return result

        snapshot_download(
            repo_id=artifact.repo_id,
            revision=artifact.revision or "main",
            token=token,
            local_dir=str(destination),
            allow_patterns=list(artifact.allow_patterns) if artifact.allow_patterns else None,
            max_workers=1,
            tqdm_class=SnapshotProgress,
        )

    def _download_hugging_face_artifact(
        self,
        artifact: LocalModelArtifact,
        destination: Path,
        progress: Callable[[int, int | None], None],
    ) -> None:
        if not artifact.repo_id:
            raise ValueError(f"Artifact {artifact.id} does not have a download repository")
        url = hf_hub_url(
            artifact.repo_id,
            artifact.filename,
            revision=artifact.revision or "main",
        )
        token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        offset = destination.stat().st_size if destination.exists() else 0
        if offset:
            headers["Range"] = f"bytes={offset}-"

        with (
            httpx.Client(follow_redirects=True, timeout=60.0) as client,
            client.stream("GET", url, headers=headers) as response,
        ):
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            if not append:
                offset = 0
            content_length = response.headers.get("content-length")
            response_bytes = int(content_length) if content_length else None
            total = offset + response_bytes if response_bytes is not None else artifact.size_bytes
            if total is not None:
                storage_root = self._existing_parent(self.model_home)
                free = shutil.disk_usage(storage_root).free
                remaining = max(0, total - offset)
                if free < remaining + 64 * 1024 * 1024:
                    raise OSError("Not enough free disk space for the selected local model")
            mode = "ab" if append else "wb"
            downloaded = offset
            with destination.open(mode) as target:
                for chunk in response.iter_bytes(COPY_CHUNK_SIZE):
                    target.write(chunk)
                    downloaded += len(chunk)
                    progress(downloaded, total)

    def shutdown(self) -> None:
        with self._lock:
            for cancel_event in self._cancel_events.values():
                cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)


_default_manager: LocalModelManager | None = None
_default_manager_lock = threading.Lock()


def get_local_model_manager() -> LocalModelManager:
    global _default_manager
    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = LocalModelManager()
        return _default_manager


def shutdown_local_model_manager() -> None:
    """Shut down the singleton only when it was created during this process."""

    if _default_manager is not None:
        _default_manager.shutdown()


atexit.register(shutdown_local_model_manager)
