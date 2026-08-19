"""Tests for managed llama.cpp Vulkan device discovery."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading
from pathlib import Path

import pytest

from agents.architectures.llama_devices import (
    DISCOVERY_IN_PROGRESS_ERROR,
    LlamaDeviceService,
    classify_device,
    llama_compute_managed_by_environment,
    llama_server_filename,
    parse_device_inventory,
    recommend_device,
)


DEVICE_OUTPUT = """Available devices:
  Vulkan0: NVIDIA GeForce RTX 3080 Laptop GPU (16175 MiB, 15407 MiB free)
  Vulkan1: Intel(R) UHD Graphics (32659 MiB, 48167 MiB free)
  Vulkan2: AMD Radeon RX 6800 (16384 MiB, 12000 MiB free)
"""


def _runtime_tree(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    filename = "llama-server.exe" if os.name == "nt" else "llama-server"
    for backend in ("cpu", "vulkan"):
        directory = root / backend
        directory.mkdir(parents=True)
        (directory / filename).write_bytes(b"binary")
    return root


def _pre_stable_id(name: str, ordinal: int = 0) -> str:
    normalized_name = " ".join(name.casefold().split())
    digest = hashlib.sha256(f"{normalized_name}\0{ordinal}".encode()).hexdigest()[:16]
    return f"gpu-{digest}"


def test_device_output_parses_stable_ids_memory_and_kinds() -> None:
    first = parse_device_inventory("ggml_vulkan: Found 3 Vulkan devices:\n" + DEVICE_OUTPUT)
    second = parse_device_inventory(
        """Available devices:
  Vulkan0: AMD Radeon RX 6800 (16384 MiB, 12000 MiB free)
  Vulkan1: NVIDIA GeForce RTX 3080 Laptop GPU (16175 MiB, 15407 MiB free)
  Vulkan2: Intel(R) UHD Graphics (32659 MiB, 48167 MiB free)
"""
    )

    assert len(first) == 3
    first_ids = {device.name: device.id for device in first}
    second_ids = {device.name: device.id for device in second}
    assert first_ids == second_ids
    assert first[0].id == "gpu-37a868c553a4b745"
    assert first[0].runtime_id == "Vulkan0"
    assert second[1].runtime_id == "Vulkan1"
    assert first[0].free_memory_mib == 15407
    assert first[0].kind == "discrete"
    assert first[1].kind == "integrated"
    assert recommend_device(first) == first[0]


@pytest.mark.parametrize(
    ("first_identity", "second_identity"),
    [
        (
            "UUID=11111111-1111-1111-1111-111111111111",
            "UUID=22222222-2222-2222-2222-222222222222",
        ),
        ("PCI=0000:01:00.0", "PCI=0000:02:00.0"),
    ],
)
def test_identical_devices_with_stable_identities_survive_reordering(
    first_identity: str,
    second_identity: str,
) -> None:
    first = parse_device_inventory(
        f"""Available devices:
  Vulkan0: NVIDIA GeForce RTX 4090 [{first_identity}] (24564 MiB, 22000 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4090 [{second_identity}] (24564 MiB, 21000 MiB free)
"""
    )
    reordered = parse_device_inventory(
        f"""Available devices:
  Vulkan0: NVIDIA GeForce RTX 4090 [{second_identity}] (24564 MiB, 21000 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4090 [{first_identity}] (24564 MiB, 22000 MiB free)
"""
    )

    assert [device.name for device in first] == ["NVIDIA GeForce RTX 4090"] * 2
    assert first[0].id == reordered[1].id
    assert first[1].id == reordered[0].id
    assert first[0].runtime_id == reordered[0].runtime_id == "Vulkan0"


@pytest.mark.parametrize(
    "stable_identity",
    [
        "UUID=11111111-1111-1111-1111-111111111111",
        "PCI=0000:01:00.0",
    ],
)
def test_stable_identity_before_or_after_memory_preserves_memory(
    stable_identity: str,
) -> None:
    before_memory = parse_device_inventory(
        "Available devices:\n"
        f"  Vulkan0: NVIDIA GeForce RTX 4090 [{stable_identity}] "
        "(24564 MiB, 22000 MiB free)\n"
    )[0]
    after_memory = parse_device_inventory(
        "Available devices:\n"
        "  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 22000 MiB free) "
        f"[{stable_identity}]\n"
    )[0]

    assert before_memory.name == after_memory.name == "NVIDIA GeForce RTX 4090"
    assert before_memory.id == after_memory.id
    assert before_memory.total_memory_mib == after_memory.total_memory_mib == 24564
    assert before_memory.free_memory_mib == after_memory.free_memory_mib == 22000


def test_unique_legacy_name_id_resolves_after_stable_identity_appears(
    tmp_path: Path,
) -> None:
    runtime = _runtime_tree(tmp_path)
    legacy = parse_device_inventory(
        "Available devices:\n" "  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 22000 MiB free)\n"
    )[0]
    stable_output = (
        "Available devices:\n"
        "  Vulkan7: NVIDIA GeForce RTX 4090 "
        "[UUID=11111111-1111-1111-1111-111111111111] "
        "(24564 MiB, 22000 MiB free)\n"
    )
    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=stable_output, stderr=""
        ),
    )

    inventory = service.inventory()

    assert inventory.devices[0].id != legacy.id
    assert inventory.resolve_device_ids((legacy.id,)) == (inventory.devices[0].id,)
    assert service.resolve_runtime_ids((legacy.id,), inventory=inventory) == ("Vulkan7",)
    with pytest.raises(ValueError, match="resolve to unique devices"):
        service.resolve_runtime_ids(
            (legacy.id, inventory.devices[0].id),
            inventory=inventory,
        )


def test_public_device_contract_exposes_safe_compatibility_ids_only() -> None:
    stable_identity = "UUID=11111111-1111-1111-1111-111111111111"
    device = parse_device_inventory(
        "Available devices:\n"
        f"  Vulkan7: NVIDIA GeForce RTX 4090 [{stable_identity}] "
        "(24564 MiB, 22000 MiB free)\n"
    )[0]

    payload = device.public_dict()

    assert payload["compatibility_ids"] == list(device.compatibility_ids)
    assert set(payload["compatibility_ids"]) == {
        _pre_stable_id("NVIDIA GeForce RTX 4090"),
        _pre_stable_id(f"NVIDIA GeForce RTX 4090 [{stable_identity}]"),
    }
    assert "runtime_id" not in payload


@pytest.mark.parametrize(
    "stable_identity",
    [
        "UUID=11111111-1111-1111-1111-111111111111",
        "PCI=0000:01:00.0",
    ],
)
@pytest.mark.parametrize("identity_after_memory", [False, True])
def test_pre_stable_parser_id_with_identity_annotation_still_resolves(
    tmp_path: Path,
    stable_identity: str,
    identity_after_memory: bool,
) -> None:
    runtime = _runtime_tree(tmp_path)
    base_name = "NVIDIA GeForce RTX 4090"
    memory = "(24564 MiB, 22000 MiB free)"
    if identity_after_memory:
        old_parser_name = f"{base_name} {memory} [{stable_identity}]"
        description = old_parser_name
    else:
        old_parser_name = f"{base_name} [{stable_identity}]"
        description = f"{old_parser_name} {memory}"
    legacy_id = _pre_stable_id(old_parser_name)
    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout=f"Available devices:\n  Vulkan7: {description}\n",
            stderr="",
        ),
    )

    inventory = service.inventory()

    assert inventory.devices[0].id != legacy_id
    assert inventory.resolve_runtime_ids((legacy_id,)) == ("Vulkan7",)


def test_legacy_name_id_never_aliases_duplicate_stable_devices(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    legacy_id = parse_device_inventory(
        "Available devices:\n" "  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 22000 MiB free)\n"
    )[0].id
    stable_output = """Available devices:
  Vulkan0: NVIDIA GeForce RTX 4090 [UUID=11111111-1111-1111-1111-111111111111]
  Vulkan1: NVIDIA GeForce RTX 4090 [UUID=22222222-2222-2222-2222-222222222222]
"""
    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=stable_output, stderr=""
        ),
    )
    inventory = service.inventory()

    with pytest.raises(ValueError, match="unavailable"):
        service.resolve_runtime_ids((legacy_id,), inventory=inventory)

    assert all(legacy_id not in device.compatibility_ids for device in inventory.devices)
    first_raw_id = _pre_stable_id(
        "NVIDIA GeForce RTX 4090 [UUID=11111111-1111-1111-1111-111111111111]"
    )
    second_raw_id = _pre_stable_id(
        "NVIDIA GeForce RTX 4090 [UUID=22222222-2222-2222-2222-222222222222]"
    )
    assert inventory.resolve_runtime_ids((first_raw_id, second_raw_id)) == (
        "Vulkan0",
        "Vulkan1",
    )
    public_aliases = [
        set(device["compatibility_ids"]) for device in inventory.public_dict()["devices"]
    ]
    assert public_aliases[0].isdisjoint(public_aliases[1])
    assert all(legacy_id not in aliases for aliases in public_aliases)
    assert all("runtime_id" not in device for device in inventory.public_dict()["devices"])


def test_identical_devices_without_stable_identities_fail_safe_when_reordered() -> None:
    outputs = (
        """Available devices:
  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 22000 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4090 (24564 MiB, 21000 MiB free)
""",
        """Available devices:
  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 21000 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4090 (24564 MiB, 22000 MiB free)
""",
    )

    for output in outputs:
        with pytest.raises(ValueError, match="duplicate GPU names without stable UUID or PCI"):
            parse_device_inventory(output)


def test_classification_keeps_integrated_and_software_out_of_auto_selection() -> None:
    assert classify_device("Intel(R) Iris Xe Graphics") == "integrated"
    assert classify_device("Intel(R) Arc(TM) Graphics") == "integrated"
    assert classify_device("AMD Radeon 780M Graphics") == "integrated"
    assert classify_device("llvmpipe (LLVM 18.1.8)") == "software"
    assert classify_device("Intel(R) Arc(TM) A770 Graphics") == "discrete"
    assert classify_device("Future Vulkan Accelerator") == "unknown"


def test_inventory_recommends_best_single_non_integrated_gpu(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    calls = []
    now = [100.0]

    def run(*args, **_kwargs):
        calls.append(args)
        return subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr="")

    service = LlamaDeviceService(
        environment={
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LOCAL_RUNNER": "llama_server",
        },
        command_runner=run,
        clock=lambda: now[0],
    )
    inventory = service.inventory()

    assert inventory.available is True
    assert inventory.recommended_backend == "gpu"
    assert len(inventory.recommended_device_ids) == 1
    recommended = next(device for device in inventory.devices if device.recommended)
    assert recommended.name == "NVIDIA GeForce RTX 3080 Laptop GPU"
    assert service.resolve_runtime_ids(inventory.recommended_device_ids) == ("Vulkan0",)

    now[0] = 144.9
    assert service.inventory() is inventory
    assert len(calls) == 1

    now[0] = 145.0
    refreshed = service.inventory()
    assert refreshed is not inventory
    assert len(calls) == 2

    assert service.inventory(refresh=True) is refreshed
    assert len(calls) == 2
    now[0] = 147.0
    service.inventory(refresh=True)
    assert len(calls) == 3


def test_failed_probe_backs_off_and_successful_empty_probe_uses_positive_ttl(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _runtime_tree(tmp_path)
    now = [100.0]
    responses = [
        subprocess.CompletedProcess([], 1, stdout="", stderr="driver unavailable"),
        subprocess.CompletedProcess([], 0, stdout="Available devices:\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr=""),
    ]
    calls = []

    def run(*_args, **_kwargs):
        calls.append(True)
        return responses.pop(0)

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=run,
        clock=lambda: now[0],
    )
    caplog.set_level(logging.INFO, logger="agents.architectures.llama_devices")

    failed = service.inventory()
    assert failed.error is not None
    assert failed.cache_policy == "negative"
    assert failed.selection_detection_error == failed.error
    now[0] = 102.9
    assert service.inventory() is failed
    assert len(calls) == 1

    now[0] = 103.0
    empty = service.inventory()
    assert empty.devices == ()
    assert empty.error is None
    assert empty.cache_policy == "positive"
    now[0] = 147.9
    assert service.inventory() is empty
    assert len(calls) == 2

    successful = service.inventory(refresh=True)
    assert successful.devices
    assert service.inventory() is successful
    assert len(calls) == 3
    assert "device discovery exited with code 1" in caplog.text
    assert "returned no parsed Vulkan devices" in caplog.text


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, False),
        ({"GEIST_LOCAL_RUNNER": "llama_server"}, False),
        ({"GEIST_LOCAL_RUNNER": "transformers"}, True),
        ({"GEIST_LLAMA_SERVER_PATH": "/operator/llama-server"}, True),
        ({"GEIST_LLAMA_ACCELERATION": "cpu"}, True),
        ({"GEIST_LLAMA_ACCELERATION": "vulkan"}, True),
        ({"GEIST_LLAMA_ACCELERATION": "invalid"}, True),
    ],
)
def test_environment_management_predicate(
    environment: dict[str, str],
    expected: bool,
) -> None:
    assert llama_compute_managed_by_environment(environment) is expected


def test_forced_refresh_obeys_minimum_interval_after_every_probe(
    tmp_path: Path,
) -> None:
    runtime = _runtime_tree(tmp_path)
    now = [100.0]
    responses = [
        subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr=""),
        subprocess.CompletedProcess([], 1, stdout="", stderr="driver unavailable"),
        subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr=""),
    ]

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=lambda *_args, **_kwargs: responses.pop(0),
        clock=lambda: now[0],
    )

    ordinary = service.inventory()
    assert ordinary.devices
    assert service.inventory(refresh=True) is ordinary
    now[0] = 101.999
    assert service.inventory(refresh=True) is ordinary
    now[0] = 102.0
    failed = service.inventory(refresh=True)
    assert failed.error is not None
    assert service.inventory() is failed
    assert service.inventory(refresh=True) is failed
    now[0] = 103.999
    assert service.inventory(refresh=True) is failed
    now[0] = 104.0
    assert service.inventory(refresh=True).devices
    assert responses == []


def test_concurrent_forced_refreshes_share_one_completion_limited_probe(
    tmp_path: Path,
) -> None:
    runtime = _runtime_tree(tmp_path)
    now = [100.0]
    refresh_probe_entered = threading.Event()
    release_refresh_probe = threading.Event()
    second_refresh_started = threading.Event()
    second_refresh_returned = threading.Event()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 2:
            refresh_probe_entered.set()
            assert release_refresh_probe.wait(timeout=2)
            now[0] = 110.0
        return subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr="")

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=run,
        clock=lambda: now[0],
    )
    ordinary = service.inventory()
    now[0] = 102.0
    first_results = []
    second_results = []

    first = threading.Thread(target=lambda: first_results.append(service.inventory(refresh=True)))

    def second_refresh() -> None:
        second_refresh_started.set()
        second_results.append(service.inventory(refresh=True))
        second_refresh_returned.set()

    second = threading.Thread(target=second_refresh)
    first.start()
    assert refresh_probe_entered.wait(timeout=2)
    second.start()
    assert second_refresh_started.wait(timeout=2)
    returned_before_release = second_refresh_returned.wait(timeout=1)
    release_refresh_probe.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert returned_before_release is True
    assert len(calls) == 2
    assert second_results == [ordinary]
    assert len(first_results) == 1
    refreshed = first_results[0]
    assert refreshed is not ordinary
    assert service.inventory(refresh=True) is refreshed
    assert len(calls) == 2

    now[0] = 112.0
    assert service.inventory(refresh=True) is not refreshed
    assert len(calls) == 3


def test_concurrent_cold_caller_returns_transient_result_without_waiting(
    tmp_path: Path,
) -> None:
    runtime = _runtime_tree(tmp_path)
    probe_entered = threading.Event()
    release_probe = threading.Event()
    second_returned = threading.Event()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(True)
        probe_entered.set()
        assert release_probe.wait(timeout=3)
        return subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr="")

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=run,
    )
    discovered = []
    transient = []
    first = threading.Thread(target=lambda: discovered.append(service.inventory()))

    def second_inventory() -> None:
        transient.append(service.inventory())
        second_returned.set()

    second = threading.Thread(target=second_inventory)
    first.start()
    assert probe_entered.wait(timeout=2)
    second.start()
    returned_before_release = second_returned.wait(timeout=1)
    release_probe.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert returned_before_release is True
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(calls) == 1
    assert len(transient) == 1
    in_progress = transient[0]
    assert in_progress.recommended_backend == "cpu"
    assert in_progress.error == DISCOVERY_IN_PROGRESS_ERROR
    assert in_progress.selection_detection_error == DISCOVERY_IN_PROGRESS_ERROR
    assert in_progress.discovery_in_progress is True
    assert in_progress.public_dict()["discovery_in_progress"] is True
    assert len(discovered) == 1
    assert discovered[0].recommended_backend == "gpu"
    assert discovered[0].discovery_in_progress is False
    assert service.inventory() is discovered[0]
    assert service.inventory() is not in_progress


@pytest.mark.parametrize("runner", ["transformers", "mlx_llama"])
def test_explicit_non_llama_runner_locks_and_hides_inventory(runner: str) -> None:
    def unexpected_probe(*_args, **_kwargs):
        pytest.fail("non-llama runners must not probe the llama.cpp runtime")

    inventory = LlamaDeviceService(
        environment={
            "GEIST_LOCAL_RUNNER": runner,
            "GEIST_LLAMA_RUNTIME_ROOT": "unused",
        },
        command_runner=unexpected_probe,
    ).inventory()

    assert inventory.available is False
    assert inventory.managed_by_environment is True
    assert inventory.devices == ()
    assert runner in inventory.reason


def test_integrated_only_and_probe_failure_recommend_cpu(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    integrated_output = (
        "Available devices:\n" "  Vulkan0: Intel(R) UHD Graphics (2048 MiB, 1024 MiB free)\n"
    )
    integrated = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=integrated_output, stderr=""
        ),
    ).inventory()
    failed = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout="", stderr="driver unavailable"
        ),
    ).inventory()

    assert integrated.recommended_backend == "cpu"
    assert integrated.devices[0].kind == "integrated"
    assert failed.recommended_backend == "cpu"
    assert failed.error is not None
    assert failed.selection_detection_error == failed.error


def test_probe_timeout_recommends_cpu_warns_and_uses_safe_process_options(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _runtime_tree(tmp_path)
    captured: dict[str, object] = {}

    def timeout(command, **options):
        captured["command"] = command
        captured.update(options)
        raise subprocess.TimeoutExpired(command, options["timeout"])

    caplog.set_level(logging.WARNING, logger="agents.architectures.llama_devices")
    inventory = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=timeout,
        timeout_seconds=0.25,
    ).inventory()

    assert inventory.recommended_backend == "cpu"
    assert inventory.error is not None
    assert inventory.selection_detection_error == inventory.error
    assert captured["command"][-1] == "--list-devices"
    assert captured["timeout"] == 0.25
    assert captured["shell"] is False
    assert "device discovery failed" in caplog.text


def test_probe_os_error_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    runtime = _runtime_tree(tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError("Vulkan loader unavailable")

    caplog.set_level(logging.WARNING, logger="agents.architectures.llama_devices")
    inventory = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=fail,
    ).inventory()

    assert inventory.recommended_backend == "cpu"
    assert "Vulkan loader unavailable" in inventory.error
    assert inventory.selection_detection_error == inventory.error
    assert "device discovery failed: Vulkan loader unavailable" in caplog.text


def test_duplicate_production_devices_fail_safe_with_vulkan_escape_hatch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _runtime_tree(tmp_path)
    production_output = """ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 22000 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4090 (24564 MiB, 21000 MiB free)
"""
    caplog.set_level(logging.WARNING, logger="agents.architectures.llama_devices")
    now = [100.0]
    calls = []

    def ambiguous_probe(*_args, **_kwargs):
        calls.append(True)
        return subprocess.CompletedProcess([], 0, stdout=production_output, stderr="")

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=ambiguous_probe,
        clock=lambda: now[0],
    )
    inventory = service.inventory()

    assert inventory.recommended_backend == "cpu"
    assert inventory.devices == ()
    assert "GEIST_LLAMA_ACCELERATION=vulkan" in inventory.reason
    assert "GEIST_LLAMA_ACCELERATION=vulkan" in inventory.error
    assert inventory.cache_policy == "positive"
    assert inventory.selection_detection_error == inventory.error
    assert "device discovery was ambiguous" in caplog.text
    now[0] = 102.9
    assert service.inventory() is inventory
    assert len(calls) == 1
    now[0] = 103.0
    assert service.inventory() is inventory
    assert len(calls) == 1
    now[0] = 144.999
    assert service.inventory() is inventory
    assert len(calls) == 1
    now[0] = 145.0
    assert service.inventory() is not inventory
    assert len(calls) == 2

    forced_inventory = LlamaDeviceService(
        environment={
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LLAMA_ACCELERATION": "vulkan",
        },
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=production_output, stderr=""
        ),
    ).inventory()

    assert forced_inventory.forced_backend == "gpu"
    assert "GEIST_LLAMA_ACCELERATION=vulkan is forcing Vulkan" in forced_inventory.reason
    assert "CPU is recommended" not in forced_inventory.reason
    assert "GEIST_LLAMA_ACCELERATION=vulkan is forcing Vulkan" in forced_inventory.error


def test_cpu_only_partial_runtime_remains_available(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    (runtime / "vulkan" / llama_server_filename()).unlink()
    now = [100.0]

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        clock=lambda: now[0],
    )
    inventory = service.inventory()

    assert inventory.available is True
    assert inventory.devices == ()
    assert inventory.recommended_backend == "cpu"
    assert inventory.error == "The Vulkan llama-server executable is unavailable"
    assert inventory.cache_policy == "positive"
    assert inventory.selection_detection_error is None
    assert "cache_policy" not in inventory.public_dict()
    assert "selection_detection_error" not in inventory.public_dict()
    assert inventory.public_dict()["discovery_in_progress"] is False
    now[0] = 103.0
    assert service.inventory() is inventory
    now[0] = 145.0
    assert service.inventory() is not inventory
