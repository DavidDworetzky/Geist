"""Tests for managed llama.cpp Vulkan device discovery."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agents.architectures.llama_devices import (
    LlamaDeviceService,
    classify_device,
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
        cache_ttl_seconds=5.0,
        clock=lambda: now[0],
    )
    inventory = service.inventory()

    assert inventory.available is True
    assert inventory.recommended_backend == "gpu"
    assert len(inventory.recommended_device_ids) == 1
    recommended = next(device for device in inventory.devices if device.recommended)
    assert recommended.name == "NVIDIA GeForce RTX 3080 Laptop GPU"
    assert service.resolve_runtime_ids(inventory.recommended_device_ids) == ("Vulkan0",)

    now[0] = 104.9
    assert service.inventory() is inventory
    assert len(calls) == 1

    now[0] = 105.0
    assert service.inventory() is not inventory
    assert len(calls) == 2

    service.inventory(refresh=True)
    assert len(calls) == 3


def test_failed_and_empty_probes_are_retried_instead_of_cached(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
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
    )

    assert service.inventory().error is not None
    assert service.inventory().devices == ()
    successful = service.inventory()
    assert successful.devices
    assert service.inventory() is successful
    assert len(calls) == 3


def test_failed_forced_refresh_clears_the_previous_success(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    responses = [
        subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr=""),
        subprocess.CompletedProcess([], 1, stdout="", stderr="driver unavailable"),
        subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr=""),
    ]

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=lambda *_args, **_kwargs: responses.pop(0),
    )

    assert service.inventory().devices
    assert service.inventory(refresh=True).error is not None
    assert service.inventory().devices
    assert responses == []


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


def test_probe_timeout_recommends_cpu_and_uses_safe_process_options(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    captured: dict[str, object] = {}

    def timeout(command, **options):
        captured["command"] = command
        captured.update(options)
        raise subprocess.TimeoutExpired(command, options["timeout"])

    inventory = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=timeout,
        timeout_seconds=0.25,
    ).inventory()

    assert inventory.recommended_backend == "cpu"
    assert inventory.error is not None
    assert captured["command"][-1] == "--list-devices"
    assert captured["timeout"] == 0.25
    assert captured["shell"] is False


def test_cpu_only_partial_runtime_remains_available(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    (runtime / "vulkan" / llama_server_filename()).unlink()

    inventory = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)}
    ).inventory()

    assert inventory.available is True
    assert inventory.devices == ()
    assert inventory.recommended_backend == "cpu"
    assert inventory.error == "The Vulkan llama-server executable is unavailable"
