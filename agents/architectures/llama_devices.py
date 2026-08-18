"""Inventory Vulkan devices exposed by the managed llama.cpp runtime."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess  # nosec B404 -- runs a fixed flag against a verified managed binary
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


DEVICE_LINE = re.compile(
    r"^\s*(?P<runtime_id>Vulkan\d+):\s+(?P<name>.+?)"
    r"(?:\s+\((?P<total>\d+)\s+MiB,\s+(?P<free>\d+)\s+MiB free\))?\s*$"
)
DEVICE_UUID = re.compile(
    r"\s*(?:[\[(]\s*)?(?:device[\s_-]*)?uuid\s*[:=]\s*"
    r"(?P<identity>[0-9a-f]{8}(?:-?[0-9a-f]{4}){3}-?[0-9a-f]{12})"
    r"(?:\s*[\])])?",
    re.IGNORECASE,
)
DEVICE_PCI_ADDRESS = re.compile(
    r"\s*(?:[\[(]\s*)?(?:pci(?:e)?(?:\s+(?:bus(?:\s+id)?|address))?|bdf)"
    r"\s*[:=]?\s*(?P<identity>(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])"
    r"(?:\s*[\])])?",
    re.IGNORECASE,
)
BARE_DEVICE_PCI_ADDRESS = re.compile(
    r"\s*[\[(]\s*(?P<identity>(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])" r"\s*[\])]",
    re.IGNORECASE,
)
DEFAULT_INVENTORY_CACHE_TTL_SECONDS = 5.0
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
Clock = Callable[[], float]


class AmbiguousLlamaDeviceError(ValueError):
    """Raised when runtime enumeration cannot identify duplicate GPUs safely."""


@dataclass(frozen=True)
class _ParsedDevice:
    runtime_id: str
    name: str
    normalized_name: str
    stable_identity: str | None
    total_memory_mib: int | None
    free_memory_mib: int | None


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


def _normalized_uuid(value: str) -> str:
    return value.replace("-", "").casefold()


def _normalized_pci_address(value: str) -> str:
    normalized = value.casefold()
    return normalized if normalized.count(":") == 2 else f"0000:{normalized}"


def _extract_stable_identity(name: str) -> tuple[str, str | None]:
    uuid_match = DEVICE_UUID.search(name)
    pci_match = DEVICE_PCI_ADDRESS.search(name) or BARE_DEVICE_PCI_ADDRESS.search(name)
    stable_identity: str | None = None
    if uuid_match is not None:
        stable_identity = f"uuid:{_normalized_uuid(uuid_match.group('identity'))}"
    elif pci_match is not None:
        stable_identity = f"pci:{_normalized_pci_address(pci_match.group('identity'))}"

    display_name = DEVICE_UUID.sub(" ", name)
    display_name = DEVICE_PCI_ADDRESS.sub(" ", display_name)
    display_name = BARE_DEVICE_PCI_ADDRESS.sub(" ", display_name)
    display_name = " ".join(display_name.split()).strip(" -|,;")
    return display_name or name, stable_identity


def parse_device_inventory(output: str) -> tuple[LlamaDevice, ...]:
    parsed: list[_ParsedDevice] = []
    for line in output.splitlines():
        match = DEVICE_LINE.match(line)
        if match is None:
            continue
        name, stable_identity = _extract_stable_identity(match.group("name").strip())
        normalized_name = " ".join(name.casefold().split())
        parsed.append(
            _ParsedDevice(
                runtime_id=match.group("runtime_id"),
                name=name,
                normalized_name=normalized_name,
                stable_identity=stable_identity,
                total_memory_mib=(int(match.group("total")) if match.group("total") else None),
                free_memory_mib=(int(match.group("free")) if match.group("free") else None),
            )
        )

    stable_identities = [
        device.stable_identity for device in parsed if device.stable_identity is not None
    ]
    if len(stable_identities) != len(set(stable_identities)):
        raise AmbiguousLlamaDeviceError("llama-server reported duplicate stable GPU identities")

    devices_by_name: dict[str, list[_ParsedDevice]] = {}
    for device in parsed:
        devices_by_name.setdefault(device.normalized_name, []).append(device)
    ambiguous_names = sorted(
        devices[0].name
        for devices in devices_by_name.values()
        if len(devices) > 1 and any(device.stable_identity is None for device in devices)
    )
    if ambiguous_names:
        raise AmbiguousLlamaDeviceError(
            "llama-server reported duplicate GPU names without stable UUID or PCI identities: "
            + ", ".join(ambiguous_names)
        )

    devices: list[LlamaDevice] = []
    for device in parsed:
        identity = (
            f"stable\0{device.stable_identity}"
            if device.stable_identity is not None
            else f"{device.normalized_name}\0{0}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        devices.append(
            LlamaDevice(
                id=f"gpu-{digest}",
                runtime_id=device.runtime_id,
                name=device.name,
                total_memory_mib=device.total_memory_mib,
                free_memory_mib=device.free_memory_mib,
                kind=classify_device(device.name),
            )
        )
    return tuple(devices)


def recommend_device(devices: Sequence[LlamaDevice]) -> LlamaDevice | None:
    suitable = [
        (index, device)
        for index, device in enumerate(devices)
        if device.kind not in {"integrated", "software"}
    ]
    if not suitable:
        return None
    return max(
        suitable,
        key=lambda item: (
            item[1].free_memory_mib if item[1].free_memory_mib is not None else -1,
            item[1].total_memory_mib if item[1].total_memory_mib is not None else -1,
            -item[0],
        ),
    )[1]


def _cpu_inventory(
    *,
    available: bool,
    reason: str,
    managed_by_environment: bool = False,
    forced_backend: str | None = None,
    devices: tuple[LlamaDevice, ...] = (),
    error: str | None = None,
) -> LlamaDeviceInventory:
    return LlamaDeviceInventory(
        available=available,
        managed_by_environment=managed_by_environment,
        forced_backend=forced_backend,
        devices=devices,
        recommended_backend="cpu",
        recommended_device_ids=(),
        reason=reason,
        error=error,
    )


class LlamaDeviceService:
    """Discover and resolve devices from a verified managed runtime tree."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        command_runner: CommandRunner = subprocess.run,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: float = DEFAULT_INVENTORY_CACHE_TTL_SECONDS,
        clock: Clock = time.monotonic,
    ) -> None:
        self.environment = environment if environment is not None else os.environ
        self.command_runner = command_runner
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self.clock = clock
        self._lock = threading.Lock()
        self._cached_inventory: LlamaDeviceInventory | None = None
        self._cached_inventory_expires_at = 0.0

    def inventory(self, *, refresh: bool = False) -> LlamaDeviceInventory:
        with self._lock:
            if (
                not refresh
                and self._cached_inventory is not None
                and self.clock() < self._cached_inventory_expires_at
            ):
                return self._cached_inventory

            inventory = self._discover_inventory()
            if inventory.error is None and inventory.devices:
                self._cached_inventory = inventory
                self._cached_inventory_expires_at = self.clock() + self.cache_ttl_seconds
            else:
                self._cached_inventory = None
                self._cached_inventory_expires_at = 0.0
            return inventory

    def _discover_inventory(self) -> LlamaDeviceInventory:
        configured_runner = self.environment.get("GEIST_LOCAL_RUNNER", "").strip()
        if configured_runner and configured_runner != "llama_server":
            return _cpu_inventory(
                available=False,
                managed_by_environment=True,
                reason=(
                    f"GEIST_LOCAL_RUNNER selects {configured_runner}, so llama.cpp compute "
                    "settings are unavailable."
                ),
            )

        explicit = self.environment.get("GEIST_LLAMA_SERVER_PATH")
        if explicit:
            return _cpu_inventory(
                available=False,
                managed_by_environment=True,
                reason="An explicit llama-server binary is configured by the environment.",
            )

        acceleration = self.environment.get("GEIST_LLAMA_ACCELERATION", "auto").strip().lower()
        if acceleration not in {"auto", "cpu", "vulkan"}:
            return _cpu_inventory(
                available=False,
                managed_by_environment=True,
                reason="The llama.cpp acceleration environment override is invalid.",
                error="GEIST_LLAMA_ACCELERATION must be auto, cpu, or vulkan",
            )
        forced_backend = {"cpu": "cpu", "vulkan": "gpu"}.get(acceleration)

        root_value = self.environment.get("GEIST_LLAMA_RUNTIME_ROOT")
        if not root_value:
            return _cpu_inventory(
                available=False,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                reason="A managed llama.cpp runtime is not installed on this platform.",
            )

        root = Path(root_value).expanduser().resolve()
        cpu_executable = root / "cpu" / llama_server_filename()
        vulkan_executable = root / "vulkan" / llama_server_filename()
        if not cpu_executable.is_file():
            return _cpu_inventory(
                available=False,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                reason="The managed CPU llama.cpp runtime is unavailable.",
                error="The CPU llama-server executable is required",
            )
        if not vulkan_executable.is_file():
            return _cpu_inventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
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
            return _cpu_inventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                reason="GPU discovery failed, so CPU is recommended.",
                error=str(error),
            )

        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode != 0:
            return _cpu_inventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                reason="The Vulkan runtime did not report usable GPUs, so CPU is recommended.",
                error=f"llama-server --list-devices exited with code {result.returncode}",
            )

        try:
            devices = parse_device_inventory(output)
        except AmbiguousLlamaDeviceError as error:
            return _cpu_inventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                reason="GPU discovery returned ambiguous device identities, so CPU is recommended.",
                error=str(error),
            )
        recommended = recommend_device(devices)
        if recommended is None:
            return _cpu_inventory(
                available=True,
                managed_by_environment=forced_backend is not None,
                forced_backend=forced_backend,
                devices=devices,
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
