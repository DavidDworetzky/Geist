"""Tests for managed llama.cpp Vulkan device discovery."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
    second = parse_device_inventory(DEVICE_OUTPUT)

    assert len(first) == 3
    assert [device.id for device in first] == [device.id for device in second]
    assert first[0].runtime_id == "Vulkan0"
    assert first[0].free_memory_mib == 15407
    assert first[0].kind == "discrete"
    assert first[1].kind == "integrated"
    assert recommend_device(first) == first[0]


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

    def run(*args, **_kwargs):
        calls.append(args)
        return subprocess.CompletedProcess([], 0, stdout=DEVICE_OUTPUT, stderr="")

    service = LlamaDeviceService(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        command_runner=run,
    )
    inventory = service.inventory()

    assert inventory.available is True
    assert inventory.recommended_backend == "gpu"
    assert len(inventory.recommended_device_ids) == 1
    recommended = next(device for device in inventory.devices if device.recommended)
    assert recommended.name == "NVIDIA GeForce RTX 3080 Laptop GPU"
    assert service.resolve_runtime_ids(inventory.recommended_device_ids) == ("Vulkan0",)

    service.inventory()
    assert len(calls) == 1


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
