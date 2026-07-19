"""Lifecycle owner for a private, loopback-only llama-server process."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import secrets
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlamaServerConnection:
    base_url: str
    api_key: str
    backend: str
    model_id: str
    model_path: str


@dataclass
class LlamaServerState:
    status: str = "stopped"
    backend: str | None = None
    model_id: str | None = None
    detail: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


ProcessFactory = Callable[..., subprocess.Popen[str]]
HealthProbe = Callable[[str, str, subprocess.Popen[str], float], None]


def _server_filename() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


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
    ) -> None:
        self.environment = environment if environment is not None else os.environ
        self._process_factory = process_factory
        self._health_probe = health_probe or self._default_health_probe
        self._port_factory = port_factory
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._stop_epoch = 0
        self._process: subprocess.Popen[str] | None = None
        self._windows_job_handle: Any | None = None
        self._connection: LlamaServerConnection | None = None
        self._state = LlamaServerState()

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                self._state.status = "error"
                self._state.detail = f"llama-server exited with code {self._process.returncode}"
            return self._state.public_dict()

    def _candidate_executables(self) -> list[tuple[str, Path]]:
        explicit = self.environment.get("GEIST_LLAMA_SERVER_PATH")
        if explicit:
            return [("explicit", Path(explicit).expanduser().resolve())]

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
        order = ["vulkan", "cpu"] if acceleration == "auto" else [acceleration]
        candidates = [(backend, root / backend / _server_filename()) for backend in order]
        existing = [(backend, path) for backend, path in candidates if path.is_file()]
        if not existing:
            expected = ", ".join(str(path) for _backend, path in candidates)
            raise FileNotFoundError(f"No llama-server executable found; checked {expected}")
        return existing

    def start(self, model_path: str | Path, model_id: str) -> LlamaServerConnection:
        resolved_model = Path(model_path).expanduser().resolve(strict=True)
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
                ):
                    return self._connection

                self._stop_locked()
                self._state = LlamaServerState(status="starting", model_id=model_id)

            errors: list[str] = []
            for backend, executable in self._candidate_executables():
                try:
                    return self._start_candidate(
                        executable,
                        backend,
                        resolved_model,
                        model_id,
                        requested_epoch,
                    )
                except Exception as error:
                    with self._lock:
                        if requested_epoch != self._stop_epoch:
                            raise RuntimeError("llama-server startup was cancelled") from error
                        self._stop_locked()
                    errors.append(f"{backend}: {error}")
                    logger.warning("llama-server %s startup failed: %s", backend, error)

            detail = "; ".join(errors) or "No llama-server runtime candidate was available"
            with self._lock:
                self._state = LlamaServerState(
                    status="error", model_id=model_id, detail=detail
                )
            raise RuntimeError(f"Unable to start llama-server ({detail})")

    def _start_candidate(
        self,
        executable: Path,
        backend: str,
        model_path: Path,
        model_id: str,
        start_epoch: int,
    ) -> LlamaServerConnection:
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
        process_options: dict[str, Any] = {
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
        if os.name == "nt":
            process_options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            process_options["start_new_session"] = True

        process = self._process_factory(args, **process_options)
        windows_job_handle = _assign_windows_kill_on_close_job(process)
        with self._lock:
            if start_epoch != self._stop_epoch:
                _terminate_process_tree(process, windows_job_handle)
                raise RuntimeError("llama-server startup was cancelled")
            self._process = process
            self._windows_job_handle = windows_job_handle
            self._state.backend = backend
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
            )
            self._connection = connection
            self._state = LlamaServerState(
                status="ready", backend=backend, model_id=model_id, detail=None
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
        windows_job_handle = self._windows_job_handle
        self._process = None
        self._windows_job_handle = None
        self._connection = None
        if process is None:
            _close_windows_handle(windows_job_handle)
            return
        _terminate_process_tree(process, windows_job_handle)


def _terminate_process_tree(
    process: subprocess.Popen[str],
    windows_job_handle: Any | None,
) -> None:
    """Terminate a retained child and every descendant without lifecycle locks."""

    if process.poll() is not None:
        _close_windows_handle(windows_job_handle)
        return
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    for child in descendants:
        with contextlib.suppress(psutil.Error):
            child.terminate()
    try:
        process.terminate()
        process.wait(timeout=3.0)
    except (OSError, subprocess.TimeoutExpired):
        for child in descendants:
            with contextlib.suppress(psutil.Error):
                child.kill()
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("Unable to confirm llama-server process termination")
    _gone, alive = psutil.wait_procs(descendants, timeout=0.5)
    for child in alive:
        with contextlib.suppress(psutil.Error):
            child.kill()
    _close_windows_handle(windows_job_handle)


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


def _assign_windows_kill_on_close_job(process: subprocess.Popen[str]) -> Any | None:
    """Attach a child to a kill-on-close Job Object when Windows permits it."""

    if os.name != "nt" or not hasattr(process, "_handle"):
        return None

    import ctypes
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    job_handle = kernel32.CreateJobObjectW(None, None)
    if not job_handle:
        logger.warning("Unable to create a Windows Job Object for llama-server")
        return None

    information = _JobObjectExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    configured = kernel32.SetInformationJobObject(
        job_handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        job_handle,
        wintypes.HANDLE(process._handle),
    )
    if not assigned:
        error_code = ctypes.get_last_error()
        _close_windows_handle(job_handle)
        logger.warning(
            "Unable to assign llama-server to a kill-on-close Job Object (Win32 error %s)",
            error_code,
        )
        return None
    return job_handle


def _close_windows_handle(handle: Any | None) -> None:
    if os.name != "nt" or handle is None:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(handle)
