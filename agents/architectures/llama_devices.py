"""Inventory Vulkan devices exposed by the managed llama.cpp runtime."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess  # nosec B404 -- runs a fixed flag against a verified managed binary
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


DEVICE_LINE = re.compile(
    r"^\s*(?P<runtime_id>Vulkan\d+):\s+(?P<name>.+?)"
    r"(?:\s+\((?P<total>\d+)\s+MiB,\s+(?P<free>\d+)\s+MiB free\))?\s*$"
)
SOFTWARE_PATTERNS = (
    "llvmpipe",
    "lavapipe",
    "swiftshader",
    "microsoft basic render",
    "software rasterizer",
    "virgl",
)
INTEGRATED_PATTERNS = (
    re.compile(r"\b(?:intel\(r\)\s+)?(?:uhd|hd) graphics\b", re.IGNORECASE),
    re.compile(r"\b(?:intel\(r\)\s+)?iris(?: xe)? graphics\b", re.IGNORECASE),
    re.compile(r"\bintel graphics\b", re.IGNORECASE),
    re.compile(r"\bintel(?:\(r\))?\s+arc(?:\(tm\))?\s+graphics\b", re.IGNORECASE),
    re.compile(r"\bradeon(?:\(tm\))? graphics\b", re.IGNORECASE),
    re.compile(r"\bradeon\s+\d{3,4}m\b", re.IGNORECASE),
    re.compile(r"\bvega\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bapu\b", re.IGNORECASE),
)
DISCRETE_PATTERNS = (
    re.compile(r"\b(?:nvidia|geforce|quadro|tesla|rtx)\b", re.IGNORECASE),
    re.compile(r"\bradeon\s+(?:rx|pro)\b", re.IGNORECASE),
    re.compile(r"\bintel(?:\(r\))?\s+arc\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class LlamaDevice:
    id: str
    runtime_id: str
    name: str
    total_memory_mib: int | None
    free_memory_mib: int | None
    kind: str
    recommended: bool = False

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("runtime_id")
        return value


@dataclass(frozen=True)
class LlamaDeviceInventory:
    available: bool
    managed_by_environment: bool
    forced_backend: str | None
    devices: tuple[LlamaDevice, ...]
    recommended_backend: str
    recommended_device_ids: tuple[str, ...]
    reason: str
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "managed_by_environment": self.managed_by_environment,
            "forced_backend": self.forced_backend,
            "devices": [device.public_dict() for device in self.devices],
            "recommended_backend": self.recommended_backend,
            "recommended_device_ids": list(self.recommended_device_ids),
            "reason": self.reason,
            "error": self.error,
        }


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def llama_server_filename() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def classify_device(name: str) -> str:
    normalized = name.casefold()
    if any(pattern in normalized for pattern in SOFTWARE_PATTERNS):
        return "software"
    if any(pattern.search(name) for pattern in INTEGRATED_PATTERNS):
        return "integrated"
    if any(pattern.search(name) for pattern in DISCRETE_PATTERNS):
        return "discrete"
    return "unknown"


def parse_device_inventory(output: str) -> tuple[LlamaDevice, ...]:
    devices: list[LlamaDevice] = []
    name_counts: dict[str, int] = {}
    for line in output.splitlines():
        match = DEVICE_LINE.match(line)
        if match is None:
            continue
        name = match.group("name").strip()
        normalized_name = " ".join(name.casefold().split())
        ordinal = name_counts.get(normalized_name, 0)
        name_counts[normalized_name] = ordinal + 1
        digest = hashlib.sha256(f"{normalized_name}\0{ordinal}".encode()).hexdigest()[:16]
        devices.append(
            LlamaDevice(
                id=f"gpu-{digest}",
                runtime_id=match.group("runtime_id"),
                name=name,
                total_memory_mib=(int(match.group("total")) if match.group("total") else None),
                free_memory_mib=(int(match.group("free")) if match.group("free") else None),
                kind=classify_device(name),
            )
        )
    return tuple(devices)


def recommend_device(devices: Sequence[LlamaDevice]) -> LlamaDevice | None:
    suitable = [device for device in devices if device.kind not in {"integrated", "software"}]
    if not suitable:
        return None
    return max(
        suitable,
        key=lambda device: (
            device.free_memory_mib if device.free_memory_mib is not None else -1,
            device.total_memory_mib if device.total_memory_mib is not None else -1,
            -devices.index(device),
        ),
    )


class LlamaDeviceService:
    """Discover and resolve devices from a verified managed runtime tree."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        command_runner: CommandRunner = subprocess.run,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.environment = environment if environment is not None else os.environ
        self.command_runner = command_runner
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._cached_inventory: LlamaDeviceInventory | None = None

    def inventory(self) -> LlamaDeviceInventory:
        with self._lock:
            if self._cached_inventory is None:
                self._cached_inventory = self._discover_inventory()
            return self._cached_inventory

    def _discover_inventory(self) -> LlamaDeviceInventory:
        explicit = self.environment.get("GEIST_LLAMA_SERVER_PATH")
        if explicit:
            return LlamaDeviceInventory(
                available=False,
                managed_by_environment=True,
                forced_backend=None,
                devices=(),
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason="An explicit llama-server binary is configured by the environment.",
            )

        acceleration = self.environment.get("GEIST_LLAMA_ACCELERATION", "auto").strip().lower()
        if acceleration not in {"auto", "cpu", "vulkan"}:
            return LlamaDeviceInventory(
                available=False,
                managed_by_environment=True,
                forced_backend=None,
                devices=(),
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason="The llama.cpp acceleration environment override is invalid.",
                error="GEIST_LLAMA_ACCELERATION must be auto, cpu, or vulkan",
            )
        forced_backend = {"cpu": "cpu", "vulkan": "gpu"}.get(acceleration)

        root_value = self.environment.get("GEIST_LLAMA_RUNTIME_ROOT")
        if not root_value:
            return LlamaDeviceInventory(
                available=False,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                devices=(),
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason="A managed llama.cpp runtime is not installed on this platform.",
            )

        root = Path(root_value).expanduser().resolve()
        cpu_executable = root / "cpu" / llama_server_filename()
        vulkan_executable = root / "vulkan" / llama_server_filename()
        if not cpu_executable.is_file():
            return LlamaDeviceInventory(
                available=False,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                devices=(),
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason="The managed CPU llama.cpp runtime is unavailable.",
                error="The CPU llama-server executable is required",
            )
        if not vulkan_executable.is_file():
            return LlamaDeviceInventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                devices=(),
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason="The Vulkan llama.cpp runtime is unavailable, so CPU is recommended.",
                error="The Vulkan llama-server executable is unavailable",
            )

        try:
            result = self.command_runner(
                [str(vulkan_executable), "--list-devices"],
                cwd=str(vulkan_executable.parent),
                env=dict(self.environment),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return LlamaDeviceInventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                devices=(),
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason="GPU discovery failed, so CPU is recommended.",
                error=str(error),
            )

        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode != 0:
            return LlamaDeviceInventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                devices=(),
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason="The Vulkan runtime did not report usable GPUs, so CPU is recommended.",
                error=f"llama-server --list-devices exited with code {result.returncode}",
            )

        devices = parse_device_inventory(output)
        recommended = recommend_device(devices)
        if recommended is None:
            return LlamaDeviceInventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                devices=devices,
                recommended_backend="cpu",
                recommended_device_ids=(),
                reason=(
                    "Only integrated or software-rendered GPUs were detected, so CPU is recommended."
                    if devices
                    else "No Vulkan devices were detected, so CPU is recommended."
                ),
            )

        marked_devices = tuple(
            replace(device, recommended=device.id == recommended.id) for device in devices
        )
        return LlamaDeviceInventory(
            available=True,
            managed_by_environment=forced_backend is not None,
            forced_backend=forced_backend,
            devices=marked_devices,
            recommended_backend="gpu",
            recommended_device_ids=(recommended.id,),
            reason=f"{recommended.name} is the recommended discrete GPU.",
        )

    def resolve_runtime_ids(self, device_ids: Sequence[str]) -> tuple[str, ...]:
        inventory = self.inventory()
        by_id = {device.id: device.runtime_id for device in inventory.devices}
        missing = [device_id for device_id in device_ids if device_id not in by_id]
        if missing:
            raise ValueError(
                "Selected llama.cpp GPU devices are unavailable: " + ", ".join(missing)
            )
        return tuple(by_id[device_id] for device_id in device_ids)


_default_service: LlamaDeviceService | None = None
_default_service_lock = threading.Lock()


def get_llama_device_service() -> LlamaDeviceService:
    global _default_service
    with _default_service_lock:
        if _default_service is None:
            _default_service = LlamaDeviceService()
        return _default_service
