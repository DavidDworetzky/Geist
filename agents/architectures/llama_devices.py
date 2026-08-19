"""Inventory Vulkan devices exposed by the managed llama.cpp runtime."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess  # nosec B404 -- runs a fixed flag against a verified managed binary
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal


DEVICE_LINE = re.compile(r"^\s*(?P<runtime_id>Vulkan\d+):\s+(?P<description>.+?)\s*$")
DEVICE_MEMORY = re.compile(
    r"\s*\((?P<total>\d+)\s+MiB,\s+(?P<free>\d+)\s+MiB free\)\s*",
    re.IGNORECASE,
)
LEGACY_TRAILING_DEVICE_MEMORY = re.compile(r"\s+\(\d+\s+MiB,\s+\d+\s+MiB free\)\s*$")
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
DEFAULT_INVENTORY_CACHE_TTL_SECONDS = 45.0
DEFAULT_NEGATIVE_INVENTORY_CACHE_TTL_SECONDS = 3.0
DEFAULT_MINIMUM_REFRESH_INTERVAL_SECONDS = 2.0
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


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlamaDevice:
    id: str
    runtime_id: str
    name: str
    total_memory_mib: int | None
    free_memory_mib: int | None
    kind: str
    recommended: bool = False
    compatibility_ids: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("runtime_id")
        value["compatibility_ids"] = list(self.compatibility_ids)
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
    cache_policy: Literal["positive", "negative"] = "positive"

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

    def resolve_runtime_ids(self, device_ids: Sequence[str]) -> tuple[str, ...]:
        """Resolve public IDs against this exact inventory snapshot.

        Stable UUID/PCI IDs remain canonical. A stable device may also accept
        former name-hash IDs that uniquely identify it in this inventory. A
        stripped name shared by duplicate devices is never accepted as an alias.
        """

        by_id = {device.id: device.runtime_id for device in self.devices}
        alias_candidates: dict[str, set[str]] = {}
        for device in self.devices:
            for compatibility_id in device.compatibility_ids:
                alias_candidates.setdefault(compatibility_id, set()).add(device.runtime_id)
        by_id.update(
            (compatibility_id, next(iter(runtime_ids)))
            for compatibility_id, runtime_ids in alias_candidates.items()
            if len(runtime_ids) == 1 and compatibility_id not in by_id
        )

        missing = [device_id for device_id in device_ids if device_id not in by_id]
        if missing:
            raise ValueError(
                "Selected llama.cpp GPU devices are unavailable: " + ", ".join(missing)
            )
        runtime_ids = tuple(by_id[device_id] for device_id in device_ids)
        if len(runtime_ids) != len(set(runtime_ids)):
            raise ValueError("Selected llama.cpp GPU device IDs must resolve to unique devices")
        return runtime_ids


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
Clock = Callable[[], float]


class AmbiguousLlamaDeviceError(ValueError):
    """Raised when runtime enumeration cannot identify duplicate GPUs safely."""


@dataclass(frozen=True)
class _ParsedDevice:
    runtime_id: str
    name: str
    normalized_name: str
    legacy_normalized_name: str
    stable_identity: str | None
    total_memory_mib: int | None
    free_memory_mib: int | None


def llama_server_filename() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def llama_compute_managed_by_environment(environment: Mapping[str, str]) -> bool:
    """Return whether operator environment overrides own llama compute selection."""

    configured_runner = environment.get("GEIST_LOCAL_RUNNER", "").strip()
    if configured_runner and configured_runner != "llama_server":
        return True
    if environment.get("GEIST_LLAMA_SERVER_PATH", "").strip():
        return True
    acceleration = environment.get("GEIST_LLAMA_ACCELERATION", "auto").strip().lower()
    return acceleration != "auto"


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


def _public_device_id(identity: str) -> str:
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"gpu-{digest}"


def _legacy_name_device_id(normalized_name: str) -> str:
    return _public_device_id(f"{normalized_name}\0{0}")


def _legacy_normalized_name(description: str) -> str:
    """Reproduce the normalized name hashed by the pre-stable-ID parser."""

    legacy_name = LEGACY_TRAILING_DEVICE_MEMORY.sub("", description).strip()
    return " ".join(legacy_name.casefold().split())


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
        description = match.group("description").strip()
        memory_match = DEVICE_MEMORY.search(description)
        description_without_memory = DEVICE_MEMORY.sub(" ", description)
        name, stable_identity = _extract_stable_identity(description_without_memory)
        normalized_name = " ".join(name.casefold().split())
        parsed.append(
            _ParsedDevice(
                runtime_id=match.group("runtime_id"),
                name=name,
                normalized_name=normalized_name,
                legacy_normalized_name=_legacy_normalized_name(description),
                stable_identity=stable_identity,
                total_memory_mib=(
                    int(memory_match.group("total")) if memory_match is not None else None
                ),
                free_memory_mib=(
                    int(memory_match.group("free")) if memory_match is not None else None
                ),
            )
        )

    stable_identities = [
        device.stable_identity for device in parsed if device.stable_identity is not None
    ]
    if len(stable_identities) != len(set(stable_identities)):
        raise AmbiguousLlamaDeviceError("llama-server reported duplicate stable GPU identities")

    devices_by_name: dict[str, list[_ParsedDevice]] = {}
    devices_by_legacy_name: dict[str, list[_ParsedDevice]] = {}
    for device in parsed:
        devices_by_name.setdefault(device.normalized_name, []).append(device)
        devices_by_legacy_name.setdefault(device.legacy_normalized_name, []).append(device)
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
        legacy_id = _legacy_name_device_id(device.normalized_name)
        if device.stable_identity is not None:
            device_id = _public_device_id(f"stable\0{device.stable_identity}")
            compatibility_id_candidates = []
            if len(devices_by_name[device.normalized_name]) == 1:
                compatibility_id_candidates.append(legacy_id)
            if len(devices_by_legacy_name[device.legacy_normalized_name]) == 1:
                compatibility_id_candidates.append(
                    _legacy_name_device_id(device.legacy_normalized_name)
                )
            compatibility_ids = tuple(dict.fromkeys(compatibility_id_candidates))
        else:
            device_id = legacy_id
            compatibility_ids = ()
        devices.append(
            LlamaDevice(
                id=device_id,
                runtime_id=device.runtime_id,
                name=device.name,
                total_memory_mib=device.total_memory_mib,
                free_memory_mib=device.free_memory_mib,
                kind=classify_device(device.name),
                compatibility_ids=compatibility_ids,
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
    cache_policy: Literal["positive", "negative"] = "positive",
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
        cache_policy=cache_policy,
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
        negative_cache_ttl_seconds: float = DEFAULT_NEGATIVE_INVENTORY_CACHE_TTL_SECONDS,
        minimum_refresh_interval_seconds: float = DEFAULT_MINIMUM_REFRESH_INTERVAL_SECONDS,
        clock: Clock = time.monotonic,
    ) -> None:
        self.environment = environment if environment is not None else os.environ
        self.command_runner = command_runner
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self.negative_cache_ttl_seconds = max(0.0, negative_cache_ttl_seconds)
        self.minimum_refresh_interval_seconds = max(0.0, minimum_refresh_interval_seconds)
        self.clock = clock
        self._lock = threading.Lock()
        self._cached_inventory: LlamaDeviceInventory | None = None
        self._cached_inventory_expires_at = 0.0
        self._cached_inventory_was_refresh = False
        self._next_refresh_allowed_at = 0.0

    def inventory(self, *, refresh: bool = False) -> LlamaDeviceInventory:
        with self._lock:
            now = self.clock()
            if self._cached_inventory is not None:
                # A normal cached result can be bypassed once. The result of
                # that forced probe then gates queued/repeated refreshes.
                if (
                    refresh
                    and self._cached_inventory_was_refresh
                    and now < self._next_refresh_allowed_at
                ):
                    return self._cached_inventory
                if not refresh and now < self._cached_inventory_expires_at:
                    return self._cached_inventory

            inventory = self._discover_inventory()
            completed_at = self.clock()
            ttl_seconds = (
                self.negative_cache_ttl_seconds
                if inventory.cache_policy == "negative"
                else self.cache_ttl_seconds
            )
            self._cached_inventory = inventory
            self._cached_inventory_expires_at = completed_at + ttl_seconds
            self._cached_inventory_was_refresh = refresh
            if refresh:
                self._next_refresh_allowed_at = completed_at + self.minimum_refresh_interval_seconds
            return inventory

    def _discover_inventory(self) -> LlamaDeviceInventory:
        managed_by_environment = llama_compute_managed_by_environment(self.environment)
        configured_runner = self.environment.get("GEIST_LOCAL_RUNNER", "").strip()
        if configured_runner and configured_runner != "llama_server":
            return _cpu_inventory(
                available=False,
                managed_by_environment=managed_by_environment,
                reason=(
                    f"GEIST_LOCAL_RUNNER selects {configured_runner}, so llama.cpp compute "
                    "settings are unavailable."
                ),
            )

        explicit = self.environment.get("GEIST_LLAMA_SERVER_PATH", "").strip()
        if explicit:
            return _cpu_inventory(
                available=False,
                managed_by_environment=managed_by_environment,
                reason="An explicit llama-server binary is configured by the environment.",
            )

        acceleration = self.environment.get("GEIST_LLAMA_ACCELERATION", "auto").strip().lower()
        if acceleration not in {"auto", "cpu", "vulkan"}:
            return _cpu_inventory(
                available=False,
                managed_by_environment=managed_by_environment,
                reason="The llama.cpp acceleration environment override is invalid.",
                error="GEIST_LLAMA_ACCELERATION must be auto, cpu, or vulkan",
            )
        forced_backend = {"cpu": "cpu", "vulkan": "gpu"}.get(acceleration)

        root_value = self.environment.get("GEIST_LLAMA_RUNTIME_ROOT")
        if not root_value:
            return _cpu_inventory(
                available=False,
                managed_by_environment=managed_by_environment,
                forced_backend=forced_backend,
                reason="A managed llama.cpp runtime is not installed on this platform.",
            )

        root = Path(root_value).expanduser().resolve()
        cpu_executable = root / "cpu" / llama_server_filename()
        vulkan_executable = root / "vulkan" / llama_server_filename()
        if not cpu_executable.is_file():
            return _cpu_inventory(
                available=False,
                managed_by_environment=managed_by_environment,
                forced_backend=forced_backend,
                reason="The managed CPU llama.cpp runtime is unavailable.",
                error="The CPU llama-server executable is required",
            )
        if not vulkan_executable.is_file():
            return _cpu_inventory(
                available=True,
                managed_by_environment=managed_by_environment,
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
            logger.warning("llama.cpp device discovery failed: %s", error)
            return _cpu_inventory(
                available=True,
                managed_by_environment=managed_by_environment,
                forced_backend=forced_backend,
                reason="GPU discovery failed, so CPU is recommended.",
                error=str(error),
                cache_policy="negative",
            )

        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode != 0:
            logger.warning(
                "llama.cpp device discovery exited with code %s",
                result.returncode,
            )
            return _cpu_inventory(
                available=True,
                managed_by_environment=managed_by_environment,
                forced_backend=forced_backend,
                reason="The Vulkan runtime did not report usable GPUs, so CPU is recommended.",
                error=f"llama-server --list-devices exited with code {result.returncode}",
                cache_policy="negative",
            )

        try:
            devices = parse_device_inventory(output)
        except AmbiguousLlamaDeviceError as error:
            logger.warning("llama.cpp device discovery was ambiguous: %s", error)
            if forced_backend == "gpu":
                guidance = (
                    "GEIST_LLAMA_ACCELERATION=vulkan is forcing Vulkan without per-device "
                    "selection."
                )
                reason = "GPU discovery returned ambiguous device identities. " + guidance
            else:
                guidance = (
                    "Set GEIST_LLAMA_ACCELERATION=vulkan to use Vulkan without per-device "
                    "selection."
                )
                reason = (
                    "GPU discovery returned ambiguous device identities, so CPU is recommended. "
                    + guidance
                )
            return _cpu_inventory(
                available=True,
                managed_by_environment=managed_by_environment,
                forced_backend=forced_backend,
                reason=reason,
                error=f"{error}. {guidance}",
            )
        if not devices:
            logger.info("llama.cpp device discovery returned no parsed Vulkan devices")
        recommended = recommend_device(devices)
        if recommended is None:
            return _cpu_inventory(
                available=True,
                managed_by_environment=managed_by_environment,
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
            managed_by_environment=managed_by_environment,
            forced_backend=forced_backend,
            devices=marked_devices,
            recommended_backend="gpu",
            recommended_device_ids=(recommended.id,),
            reason=f"{recommended.name} is the recommended discrete GPU.",
        )

    def resolve_runtime_ids(
        self,
        device_ids: Sequence[str],
        *,
        inventory: LlamaDeviceInventory | None = None,
    ) -> tuple[str, ...]:
        """Resolve IDs, optionally against a caller-owned inventory snapshot."""

        snapshot = inventory if inventory is not None else self.inventory()
        return snapshot.resolve_runtime_ids(device_ids)


_default_service: LlamaDeviceService | None = None
_default_service_lock = threading.Lock()


def get_llama_device_service() -> LlamaDeviceService:
    global _default_service
    with _default_service_lock:
        if _default_service is None:
            _default_service = LlamaDeviceService()
        return _default_service
