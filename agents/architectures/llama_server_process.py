"""Lifecycle owner for a private, loopback-only llama-server process."""

from __future__ import annotations

import atexit
import logging
import os
import secrets
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agents.architectures.llama_devices import (
    LlamaDeviceService,
    get_llama_device_service,
    llama_server_filename,
)
from agents.architectures.llama_server_process_platform import (
    attach_process_lifetime,
    close_process_lifetime,
    process_options,
    terminate_process_tree,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlamaServerConnection:
    base_url: str
    api_key: str
    backend: str
    model_id: str
    model_path: str
    device_ids: tuple[str, ...] = ()
    selection_backend: str = "auto"
    selection_device_ids: tuple[str, ...] = ()


@dataclass
class LlamaServerState:
    status: str = "stopped"
    backend: str | None = None
    model_id: str | None = None
    device_ids: list[str] = field(default_factory=list)
    detail: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


ProcessFactory = Callable[..., subprocess.Popen[str]]
HealthProbe = Callable[[str, str, subprocess.Popen[str], float], None]


@dataclass(frozen=True)
class LlamaServerCandidate:
    backend: str
    executable: Path
    runtime_device_ids: tuple[str, ...] = ()
    public_device_ids: tuple[str, ...] = ()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class LlamaServerManager:
    """Run one model server at a time and retain its exact process handle."""

    def __init__(
        self,
        *,
        environment: dict[str, str] | None = None,
        process_factory: ProcessFactory = subprocess.Popen,
        health_probe: HealthProbe | None = None,
        port_factory: Callable[[], int] = _free_loopback_port,
        device_service: LlamaDeviceService | None = None,
    ) -> None:
        uses_process_environment = environment is None
        self.environment = environment if environment is not None else os.environ
        self._process_factory = process_factory
        self._health_probe = health_probe or self._default_health_probe
        self._port_factory = port_factory
        self._device_service = device_service or (
            get_llama_device_service()
            if uses_process_environment
            else LlamaDeviceService(environment=self.environment)
        )
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._stop_epoch = 0
        self._process: subprocess.Popen[str] | None = None
        self._process_lifetime_handle: Any | None = None
        self._connection: LlamaServerConnection | None = None
        self._state = LlamaServerState()

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                self._state.status = "error"
                self._state.detail = f"llama-server exited with code {self._process.returncode}"
            return self._state.public_dict()

    def device_inventory(self, *, refresh: bool = False) -> dict[str, Any]:
        return self._device_service.inventory(refresh=refresh).public_dict()

    def _selection_identity(
        self,
        backend: str,
        device_ids: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...]]:
        """Remove user choices that cannot affect an operator-managed runtime."""

        if self.environment.get("GEIST_LLAMA_SERVER_PATH"):
            return "environment", ()
        acceleration = self.environment.get("GEIST_LLAMA_ACCELERATION", "auto").strip().lower()
        if acceleration == "cpu":
            return "cpu", ()
        if acceleration == "vulkan":
            return "gpu", ()
        if backend == "cpu":
            return "cpu", ()
        return backend, device_ids

    def _candidate_executables(
        self,
        backend: str,
        device_ids: tuple[str, ...],
    ) -> list[LlamaServerCandidate]:
        explicit = self.environment.get("GEIST_LLAMA_SERVER_PATH")
        if explicit:
            return [LlamaServerCandidate("explicit", Path(explicit).expanduser().resolve())]

        root_value = self.environment.get("GEIST_LLAMA_RUNTIME_ROOT")
        if not root_value:
            raise FileNotFoundError(
                "Set GEIST_LLAMA_SERVER_PATH or GEIST_LLAMA_RUNTIME_ROOT to a verified "
                "llama.cpp runtime directory."
            )
        root = Path(root_value).expanduser().resolve()
        acceleration = self.environment.get("GEIST_LLAMA_ACCELERATION", "auto").strip().lower()
        if acceleration not in {"auto", "cpu", "vulkan"}:
            raise ValueError("GEIST_LLAMA_ACCELERATION must be auto, cpu, or vulkan")

        requested_backend = backend.strip().lower()
        if requested_backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("llama.cpp backend must be auto, cpu, or gpu")
        forced_backend = {"cpu": "cpu", "vulkan": "gpu"}.get(acceleration)
        effective_backend = forced_backend or requested_backend
        filename = llama_server_filename()

        candidates: list[LlamaServerCandidate]
        if effective_backend == "cpu":
            candidates = [LlamaServerCandidate("cpu", root / "cpu" / filename)]
        elif effective_backend == "gpu":
            if not device_ids and forced_backend is None:
                raise ValueError("Select at least one llama.cpp GPU device")
            runtime_ids = self._device_service.resolve_runtime_ids(device_ids) if device_ids else ()
            candidates = [
                LlamaServerCandidate(
                    "vulkan",
                    root / "vulkan" / filename,
                    runtime_ids,
                    device_ids,
                )
            ]
        else:
            inventory = self._device_service.inventory()
            if inventory.recommended_backend == "gpu" and inventory.recommended_device_ids:
                recommended_ids = inventory.recommended_device_ids
                candidates = [
                    LlamaServerCandidate(
                        "vulkan",
                        root / "vulkan" / filename,
                        self._device_service.resolve_runtime_ids(recommended_ids),
                        recommended_ids,
                    ),
                    LlamaServerCandidate("cpu", root / "cpu" / filename),
                ]
            else:
                candidates = [LlamaServerCandidate("cpu", root / "cpu" / filename)]

        existing = [candidate for candidate in candidates if candidate.executable.is_file()]
        if not existing:
            expected = ", ".join(str(candidate.executable) for candidate in candidates)
            raise FileNotFoundError(f"No llama-server executable found; checked {expected}")
        return existing

    def start(
        self,
        model_path: str | Path,
        model_id: str,
        *,
        backend: str = "auto",
        device_ids: list[str] | tuple[str, ...] | None = None,
    ) -> LlamaServerConnection:
        resolved_model = Path(model_path).expanduser().resolve(strict=True)
        requested_backend = backend.strip().lower()
        requested_device_ids = tuple(device_ids or ())
        selection_backend, selection_device_ids = self._selection_identity(
            requested_backend,
            requested_device_ids,
        )
        with self._lock:
            requested_epoch = self._stop_epoch

        # Serialize competing model starts without preventing stop() from
        # acquiring the lifecycle lock and terminating a process during health.
        with self._start_lock:
            with self._lock:
                if requested_epoch != self._stop_epoch:
                    raise RuntimeError("llama-server startup was cancelled")
                if (
                    self._process is not None
                    and self._process.poll() is None
                    and self._connection is not None
                    and self._connection.model_id == model_id
                    and self._connection.model_path == str(resolved_model)
                    and self._connection.selection_backend == selection_backend
                    and self._connection.selection_device_ids == selection_device_ids
                ):
                    return self._connection

                self._stop_locked()
                self._state = LlamaServerState(status="starting", model_id=model_id)

            try:
                candidates = self._candidate_executables(
                    selection_backend,
                    selection_device_ids,
                )
            except Exception as error:
                with self._lock:
                    if requested_epoch != self._stop_epoch:
                        raise RuntimeError("llama-server startup was cancelled") from error
                    self._state = LlamaServerState(
                        status="error",
                        model_id=model_id,
                        detail=str(error),
                    )
                raise

            errors: list[str] = []
            for candidate in candidates:
                try:
                    return self._start_candidate(
                        candidate,
                        resolved_model,
                        model_id,
                        requested_epoch,
                        selection_backend,
                        selection_device_ids,
                    )
                except Exception as error:
                    with self._lock:
                        if requested_epoch != self._stop_epoch:
                            raise RuntimeError("llama-server startup was cancelled") from error
                        self._stop_locked()
                    errors.append(f"{candidate.backend}: {error}")
                    logger.warning("llama-server %s startup failed: %s", candidate.backend, error)

            detail = "; ".join(errors) or "No llama-server runtime candidate was available"
            with self._lock:
                self._state = LlamaServerState(status="error", model_id=model_id, detail=detail)
            raise RuntimeError(f"Unable to start llama-server ({detail})")

    def _start_candidate(
        self,
        candidate: LlamaServerCandidate,
        model_path: Path,
        model_id: str,
        start_epoch: int,
        selection_backend: str,
        selection_device_ids: tuple[str, ...],
    ) -> LlamaServerConnection:
        executable = candidate.executable
        backend = candidate.backend
        if not executable.is_file():
            raise FileNotFoundError(f"llama-server executable does not exist: {executable}")
        port = self._port_factory()
        api_key = secrets.token_urlsafe(32)
        args = [
            str(executable),
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            api_key,
            "--no-webui",
            "--jinja",
        ]
        context_size = self.environment.get("GEIST_LLAMA_CONTEXT_SIZE", "32768").strip()
        if not context_size.isdigit() or int(context_size) <= 0:
            raise ValueError("GEIST_LLAMA_CONTEXT_SIZE must be a positive integer")
        args.extend(["--ctx-size", context_size])
        if backend == "vulkan":
            gpu_layers = self.environment.get("GEIST_LLAMA_GPU_LAYERS", "999").strip()
            if not gpu_layers.isdigit():
                raise ValueError("GEIST_LLAMA_GPU_LAYERS must be a non-negative integer")
            args.extend(["--n-gpu-layers", gpu_layers])
            if candidate.runtime_device_ids:
                args.extend(["--device", ",".join(candidate.runtime_device_ids)])
        popen_options: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "shell": False,
            "cwd": str(executable.parent),
            "env": dict(self.environment),
        }
        popen_options.update(process_options_for_platform())

        process = self._process_factory(args, **popen_options)
        process_lifetime_handle = attach_process_lifetime(process)
        with self._lock:
            if start_epoch != self._stop_epoch:
                terminate_process_tree(process, process_lifetime_handle)
                raise RuntimeError("llama-server startup was cancelled")
            self._process = process
            self._process_lifetime_handle = process_lifetime_handle
            self._state.backend = backend
            self._state.device_ids = list(candidate.public_device_ids)
        self._drain_output(process, api_key)
        timeout = float(self.environment.get("GEIST_LLAMA_STARTUP_TIMEOUT_SECONDS", "180"))
        base_url = f"http://127.0.0.1:{port}"
        self._health_probe(base_url, api_key, process, timeout)
        with self._lock:
            if (
                start_epoch != self._stop_epoch
                or self._process is not process
                or process.poll() is not None
            ):
                raise RuntimeError("llama-server startup was cancelled")
            connection = LlamaServerConnection(
                base_url,
                api_key,
                backend,
                model_id,
                str(model_path),
                candidate.public_device_ids,
                selection_backend,
                selection_device_ids,
            )
            self._connection = connection
            self._state = LlamaServerState(
                status="ready",
                backend=backend,
                model_id=model_id,
                device_ids=list(candidate.public_device_ids),
                detail=None,
            )
            return connection

    @staticmethod
    def _drain_output(process: subprocess.Popen[str], api_key: str) -> None:
        if process.stdout is None:
            return

        def consume() -> None:
            for line in process.stdout or ():
                logger.info("[llama-server] %s", line.rstrip().replace(api_key, "[redacted]"))

        threading.Thread(
            target=consume,
            name="geist-llama-server-log",
            daemon=True,
        ).start()

    @staticmethod
    def _default_health_probe(
        base_url: str,
        api_key: str,
        process: subprocess.Popen[str],
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        headers = {"Authorization": f"Bearer {api_key}"}
        last_error: Exception | None = None
        with httpx.Client(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {process.returncode}")
                try:
                    response = client.get(f"{base_url}/health", headers=headers)
                    if response.status_code == 200:
                        return
                    last_error = RuntimeError(f"health returned HTTP {response.status_code}")
                except httpx.HTTPError as error:
                    last_error = error
                time.sleep(0.2)
        raise TimeoutError(f"llama-server did not become healthy: {last_error}")

    def stop(self) -> None:
        with self._lock:
            self._stop_epoch += 1
            self._stop_locked()
            self._state = LlamaServerState()

    def _stop_locked(self) -> None:
        process = self._process
        process_lifetime_handle = self._process_lifetime_handle
        self._process = None
        self._process_lifetime_handle = None
        self._connection = None
        if process is None:
            close_process_lifetime(process_lifetime_handle)
            return
        terminate_process_tree(process, process_lifetime_handle)


_default_manager: LlamaServerManager | None = None
_default_manager_lock = threading.Lock()


def get_llama_server_manager() -> LlamaServerManager:
    global _default_manager
    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = LlamaServerManager()
        return _default_manager


def shutdown_llama_server_manager() -> None:
    """Stop the singleton only when a runner created it."""

    if _default_manager is not None:
        _default_manager.stop()


atexit.register(shutdown_llama_server_manager)


def process_options_for_platform() -> dict[str, Any]:
    """Small indirection retained as a test seam for platform process options."""

    return process_options()
