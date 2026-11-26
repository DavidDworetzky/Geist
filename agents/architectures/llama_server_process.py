"""Lifecycle owner for a private, loopback-only llama-server process."""

from __future__ import annotations

import atexit
import logging
import os
import platform
import secrets
import socket
import subprocess  # nosec B404 - argv-only lifecycle of the configured local inference server
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from agents.architectures.llama_devices import (
    LlamaDevice,
    LlamaDeviceInventory,
    LlamaDeviceService,
    get_llama_device_service,
    llama_server_filename,
    recommend_device,
)
from agents.architectures.llama_server_process_platform import (
    attach_process_lifetime,
    close_process_lifetime,
    process_options,
    terminate_process_tree,
)
from agents.model_load_errors import ModelLoadErrorCode, ModelMemoryError


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
    detection_error: str | None = None
    allow_system_ram: bool = False
    model_specific_placement: bool = False


@dataclass
class LlamaServerState:
    status: str = "stopped"
    backend: str | None = None
    model_id: str | None = None
    device_ids: list[str] = field(default_factory=list)
    detail: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StartupDiagnostics:
    memory_error: ModelLoadErrorCode | None = None
    finished: threading.Event = field(default_factory=threading.Event)

    def observe(self, line: str) -> None:
        text = line[:4096].casefold()
        if "erroroutofdevicememory" in text or "cuda out of memory" in text:
            self.memory_error = "gpu_memory"
        elif any(
            marker in text
            for marker in (
                "erroroutofhostmemory",
                "std::bad_alloc",
                "cannot allocate memory",
                "failed to allocate cpu",
                "out of host memory",
            )
        ):
            self.memory_error = "system_memory"


ProcessFactory = Callable[..., subprocess.Popen[str]]
HealthProbe = Callable[[str, str, subprocess.Popen[str], float], None]


@dataclass(frozen=True)
class LlamaServerCandidate:
    backend: str
    executable: Path
    runtime_device_ids: tuple[str, ...] = ()
    public_device_ids: tuple[str, ...] = ()
    detection_error: str | None = None

    full_offload: bool = False
    requires_gpu_oom: bool = False


def _fit_estimator_path(executable: Path) -> Path:
    return executable.with_name("llama-fit-params" + executable.suffix)


def _auto_gpu_pair(inventory: LlamaDeviceInventory) -> tuple[LlamaDevice, LlamaDevice] | None:
    singles = [
        device
        for device in inventory.devices
        if device.kind == "discrete" and not device.additional_runtime_ids
    ]
    first = next(
        (device for device in singles if (device.id,) == inventory.recommended_device_ids),
        None,
    )
    if first is None:
        return None
    second = recommend_device([device for device in singles if device != first])
    return (first, second) if second is not None else None


def _fits_gpu_memory(
    executable: Path, model: Path, devices: tuple[LlamaDevice, ...], environment: Mapping[str, str]
) -> bool | None:
    """Ask the sibling llama.cpp estimator for model, KV and compute allocations per GPU."""
    estimator = _fit_estimator_path(executable)
    if not estimator.is_file() or any(device.free_memory_mib is None for device in devices):
        return None
    ids = [device.runtime_id for device in devices]
    args = [
        str(estimator),
        "--model",
        str(model),
        "--device",
        ",".join(ids),
        "--ctx-size",
        environment.get("GEIST_LLAMA_CONTEXT_SIZE", "32768").strip(),
        "--n-gpu-layers",
        "999",
        "--fit-print",
        "on",
    ]
    try:
        result = subprocess.run(  # nosec B603 -- fixed sibling of the managed executable
            args,
            cwd=executable.parent,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    allocations = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] in ids:
            if (
                len(fields) != 4
                or fields[0] in allocations
                or not all(
                    field.isascii() and field.isdecimal() and len(field) <= 12
                    for field in fields[1:]
                )
            ):
                return None
            allocations[fields[0]] = sum(map(int, fields[1:]))
    if set(allocations) != set(ids) or not all(allocations.values()):
        return None
    # Reserve one GiB on each adapter for the driver and concurrent desktop use.
    return all(
        allocations[device.runtime_id] + 1024 <= device.free_memory_mib
        for device in devices
        if device.free_memory_mib is not None
    )


def _server_filename(system: str | None = None) -> str:
    platform_name = (system or platform.system()).lower()
    return "llama-server.exe" if platform_name == "windows" else "llama-server"


def default_llama_runtime_root(
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return the packaged llama.cpp root when no operator override is set."""

    env = environment if environment is not None else os.environ
    configured = env.get("GEIST_LLAMA_RUNTIME_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "runtimes" / "llama.cpp"

    geist_home = env.get("GEIST_HOME")
    if geist_home:
        return Path(geist_home).expanduser().resolve().parent / "geist-runtime" / "llama.cpp"

    return Path(__file__).resolve().parents[2] / "runtime" / "llama.cpp"


def llama_server_runtime_available(
    environment: Mapping[str, str] | None = None,
    *,
    system: str | None = None,
) -> bool:
    """Return whether the configured or packaged runtime has a usable server."""

    env = environment if environment is not None else os.environ
    platform_name = (system or platform.system()).lower()
    explicit = env.get("GEIST_LLAMA_SERVER_PATH")
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        root = default_llama_runtime_root(env)
        acceleration = env.get("GEIST_LLAMA_ACCELERATION", "auto").strip().lower()
        if acceleration not in {"auto", "cpu", "vulkan"}:
            return False
        order = ["vulkan", "cpu"] if acceleration == "auto" else [acceleration]
        candidates = [root / backend / _server_filename(platform_name) for backend in order]

    return any(
        candidate.is_file() and (platform_name == "windows" or os.access(candidate, os.X_OK))
        for candidate in candidates
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class LlamaServerManager:
    """Run one model server at a time and retain its exact process handle."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
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
        return self._device_service.inventory(
            refresh=refresh,
            allow_in_progress=True,
        ).public_dict()

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

    def _auto_gpu_candidates(
        self,
        inventory: LlamaDeviceInventory,
        executable: Path,
        model: Path,
        allow_system_ram: bool,
        placement_started_at: float,
    ) -> list[LlamaServerCandidate]:
        ids = inventory.recommended_device_ids
        primary = LlamaServerCandidate(
            "vulkan", executable, inventory.resolve_runtime_ids(ids), ids
        )
        gpu_pair = _auto_gpu_pair(inventory)
        if gpu_pair is None or "GEIST_LLAMA_GPU_LAYERS" in self.environment:
            return [primary]
        first, second = gpu_pair

        pair = LlamaServerCandidate(
            "vulkan", executable, (first.runtime_id, second.runtime_id), (first.id, second.id)
        )
        # A rate-limited refresh can return a probe begun before the old model stopped.
        first_fits = None
        if inventory.probe_started_at >= placement_started_at:
            first_fits = _fits_gpu_memory(executable, model, (first,), self.environment)
        pair_fits = (
            _fits_gpu_memory(executable, model, (first, second), self.environment)
            if first_fits is False
            else None
        )
        pair_first = first_fits is False and pair_fits is True
        logger.debug(
            "Auto Vulkan placement for %s: first=%s (%s), pair=%s,%s (%s), pair_first=%s",
            model.name,
            first.runtime_id,
            first_fits,
            first.runtime_id,
            second.runtime_id,
            pair_fits,
            pair_first,
        )
        candidates = [] if pair_first else [replace(primary, full_offload=True)]
        candidates.append(replace(pair, full_offload=True, requires_gpu_oom=not pair_first))
        if allow_system_ram:
            candidates.append(replace(pair, requires_gpu_oom=True))
        if pair_first or allow_system_ram:
            # Mixed adapters can fail together even when both memory estimates fit.
            candidates.append(replace(primary, full_offload=not allow_system_ram))
        return candidates

    def _candidate_executables(
        self,
        backend: str,
        device_ids: tuple[str, ...],
        model: Path,
        allow_system_ram: bool,
    ) -> list[LlamaServerCandidate]:
        explicit = self.environment.get("GEIST_LLAMA_SERVER_PATH")
        if explicit:
            return [LlamaServerCandidate("explicit", Path(explicit).expanduser().resolve())]

        root = default_llama_runtime_root(self.environment)
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
            placement_started_at = time.perf_counter()
            executable = root / "vulkan" / filename
            inventory = self._device_service.inventory()
            if (
                inventory.recommended_backend == "gpu"
                and inventory.recommended_device_ids
                and "GEIST_LLAMA_GPU_LAYERS" not in self.environment
                and _fit_estimator_path(executable).is_file()
                and _auto_gpu_pair(inventory) is not None
                and inventory.probe_started_at < placement_started_at
            ):
                inventory = self._device_service.inventory(refresh=True)
            if inventory.recommended_backend == "gpu" and inventory.recommended_device_ids:
                candidates = self._auto_gpu_candidates(
                    inventory, executable, model, allow_system_ram, placement_started_at
                )
                candidates.append(LlamaServerCandidate("cpu", root / "cpu" / filename))
            else:
                candidates = [
                    LlamaServerCandidate(
                        "cpu",
                        root / "cpu" / filename,
                        detection_error=inventory.selection_detection_error,
                    )
                ]

        if (
            effective_backend == "auto"
            and len(candidates) > 1
            and candidates[-1].executable.is_file()
        ):
            # Let the startup loop observe a preferred runtime that vanished
            # after discovery so its exact error is attached to the available
            # fallback. Otherwise retain the fail-closed executable filtering.
            return candidates

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
        allow_system_ram: bool = False,
    ) -> LlamaServerConnection:
        if not isinstance(allow_system_ram, bool):
            raise ValueError("allow_system_ram must be a boolean")
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
                    and self._connection.allow_system_ram == allow_system_ram
                ):
                    return self._connection

                self._stop_locked()
                self._state = LlamaServerState(status="starting", model_id=model_id)

            try:
                candidates = self._candidate_executables(
                    selection_backend,
                    selection_device_ids,
                    resolved_model,
                    allow_system_ram,
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
            prior_startup_error: str | None = None
            last_error: Exception | None = None
            gpu_memory_failure: ModelMemoryError | None = None
            adaptive = any(candidate.full_offload for candidate in candidates)
            for candidate in candidates:
                with self._lock:
                    if requested_epoch != self._stop_epoch:
                        raise RuntimeError("llama-server startup was cancelled")
                if candidate.backend == "cpu" and gpu_memory_failure and not allow_system_ram:
                    last_error = gpu_memory_failure
                    break
                gpu_oom = (
                    isinstance(last_error, ModelMemoryError) and last_error.code == "gpu_memory"
                )
                if candidate.requires_gpu_oom and not gpu_oom:
                    continue
                if (
                    isinstance(last_error, ModelMemoryError)
                    and not allow_system_ram
                    and not (gpu_oom and candidate.full_offload)
                ):
                    break
                startup_candidate = (
                    replace(candidate, detection_error=prior_startup_error)
                    if prior_startup_error is not None and candidate.detection_error is None
                    else candidate
                )
                try:
                    return self._start_candidate(
                        startup_candidate,
                        resolved_model,
                        model_id,
                        requested_epoch,
                        selection_backend,
                        selection_device_ids,
                        allow_system_ram,
                        adaptive,
                    )
                except Exception as error:
                    with self._lock:
                        if requested_epoch != self._stop_epoch:
                            raise RuntimeError("llama-server startup was cancelled") from error
                        self._stop_locked()
                    prior_startup_error = str(error)
                    last_error = error
                    if isinstance(error, ModelMemoryError) and error.code == "gpu_memory":
                        gpu_memory_failure = error
                    errors.append(f"{candidate.backend}: {error}")
                    logger.warning("llama-server %s startup failed: %s", candidate.backend, error)

            detail = "; ".join(errors) or "No llama-server runtime candidate was available"
            with self._lock:
                self._state = LlamaServerState(status="error", model_id=model_id, detail=detail)
            if isinstance(last_error, ModelMemoryError):
                raise last_error
            raise RuntimeError(f"Unable to start llama-server ({detail})")

    def _start_candidate(
        self,
        candidate: LlamaServerCandidate,
        model_path: Path,
        model_id: str,
        start_epoch: int,
        selection_backend: str,
        selection_device_ids: tuple[str, ...],
        allow_system_ram: bool,
        adaptive: bool,
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
            gpu_layers = self.environment.get(
                "GEIST_LLAMA_GPU_LAYERS",
                "auto" if allow_system_ram and not candidate.full_offload else "999",
            ).strip()
            if gpu_layers != "auto" and not gpu_layers.isdigit():
                raise ValueError("GEIST_LLAMA_GPU_LAYERS must be auto or a non-negative integer")
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
        diagnostics = self._drain_output(process, api_key)
        timeout = float(self.environment.get("GEIST_LLAMA_STARTUP_TIMEOUT_SECONDS", "180"))
        base_url = f"http://127.0.0.1:{port}"
        try:
            self._health_probe(base_url, api_key, process, timeout)
        except Exception as error:
            if process.poll() is not None:
                diagnostics.finished.wait(timeout=1.0)
            if diagnostics.memory_error:
                raise ModelMemoryError(
                    diagnostics.memory_error,
                    runtime="llama_server",
                    model_id=model_id,
                    allow_system_ram=allow_system_ram,
                    offload_setting_available=(
                        backend == "vulkan"
                        and "GEIST_LLAMA_GPU_LAYERS" not in self.environment
                        and not self.environment.get("GEIST_LLAMA_SERVER_PATH")
                    ),
                ) from error
            raise
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
                candidate.detection_error,
                allow_system_ram,
                adaptive,
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
    def _drain_output(process: subprocess.Popen[str], api_key: str) -> StartupDiagnostics:
        diagnostics = StartupDiagnostics()
        if process.stdout is None:
            diagnostics.finished.set()
            return diagnostics

        def consume() -> None:
            try:
                for line in process.stdout or ():
                    diagnostics.observe(line)
                    logger.info("[llama-server] %s", line.rstrip().replace(api_key, "[redacted]"))
            finally:
                diagnostics.finished.set()

        threading.Thread(
            target=consume,
            name="geist-llama-server-log",
            daemon=True,
        ).start()
        return diagnostics

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
