"""Process lifecycle tests for the private llama-server supervisor."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from agents.architectures.llama_devices import (
    LlamaDevice,
    LlamaDeviceInventory,
    LlamaDeviceService,
    llama_server_filename,
)
from agents.architectures.llama_server_process import LlamaServerManager
from agents.architectures.llama_server_process_posix import process_options as posix_options
from agents.architectures.llama_server_process_windows import process_options as windows_options


class FakeProcess:
    def __init__(self, args):
        self.args = args
        self.pid = 99_999_999
        self.returncode = None
        self.stdout = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _runtime_tree(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    for backend in ("cpu", "vulkan"):
        directory = root / backend
        directory.mkdir(parents=True)
        (directory / executable).write_bytes(b"binary")
    return root


class StaticDeviceService:
    def __init__(self, *, device_count: int = 1, error: str | None = None):
        self.refresh_requests = []
        self.resolve_requests = []
        self.error = error
        self.devices = tuple(
            LlamaDevice(
                id=f"gpu-{index}",
                runtime_id=f"Vulkan{index}",
                name=f"Discrete GPU {index}",
                total_memory_mib=8192 + index,
                free_memory_mib=4096 + index,
                kind="discrete",
                recommended=index == 0,
            )
            for index in range(device_count)
        )

    def inventory(self, *, refresh=False):
        self.refresh_requests.append(refresh)
        return LlamaDeviceInventory(
            available=True,
            managed_by_environment=False,
            forced_backend=None,
            devices=self.devices,
            recommended_backend="gpu" if self.devices else "cpu",
            recommended_device_ids=(self.devices[0].id,) if self.devices else (),
            reason="test inventory",
            error=self.error,
        )

    def resolve_runtime_ids(self, device_ids):
        self.resolve_requests.append(tuple(device_ids))
        by_id = {device.id: device.runtime_id for device in self.devices}
        missing = [device_id for device_id in device_ids if device_id not in by_id]
        if missing:
            raise ValueError("unavailable")
        return tuple(by_id[device_id] for device_id in device_ids)


def test_platform_process_options_are_isolated() -> None:
    class WindowsSubprocess:
        CREATE_NEW_PROCESS_GROUP = 0x200
        CREATE_NO_WINDOW = 0x8000000

    assert posix_options() == {"start_new_session": True}
    assert windows_options(WindowsSubprocess) == {"creationflags": 0x8000200}


def test_device_inventory_forwards_refresh_to_service() -> None:
    device_service = StaticDeviceService()
    manager = LlamaServerManager(environment={}, device_service=device_service)

    inventory = manager.device_inventory(refresh=True)

    assert inventory["recommended_backend"] == "gpu"
    assert device_service.refresh_requests == [True]


def test_auto_prefers_vulkan_and_uses_private_authenticated_flags(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []
    probes = []

    def process_factory(args, **options):
        calls.append((args, options))
        return FakeProcess(args)

    def health_probe(base_url, api_key, process, timeout):
        probes.append((base_url, api_key, process, timeout))

    device_service = StaticDeviceService()
    manager = LlamaServerManager(
        environment={
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LLAMA_ACCELERATION": "auto",
        },
        process_factory=process_factory,
        health_probe=health_probe,
        port_factory=lambda: 43123,
        device_service=device_service,
    )

    connection = manager.start(model, "test/model")
    args, options = calls[0]

    assert connection.backend == "vulkan"
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    assert args[0] == str(runtime / "vulkan" / executable)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "43123"
    assert "--api-key" in args
    assert "--no-webui" in args
    assert "--jinja" in args
    assert args[args.index("--ctx-size") + 1] == "32768"
    assert args[args.index("--n-gpu-layers") + 1] == "999"
    assert options["cwd"] == str(runtime / "vulkan")
    assert options["shell"] is False
    if os.name == "nt":
        assert options["creationflags"]
    else:
        assert options["start_new_session"] is True
    assert probes[0][0] == "http://127.0.0.1:43123"
    assert device_service.refresh_requests == [False]
    assert device_service.resolve_requests == []
    assert manager.public_status()["status"] == "ready"
    manager.stop()


@pytest.mark.parametrize(
    ("discovery_error", "expected_detection_error"),
    [("Vulkan loader unavailable", "Vulkan loader unavailable"), (None, None)],
)
def test_auto_cpu_carries_exact_inventory_error_provenance(
    tmp_path: Path,
    discovery_error: str | None,
    expected_detection_error: str | None,
) -> None:
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    device_service = StaticDeviceService(device_count=0, error=discovery_error)
    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=lambda args, **_options: FakeProcess(args),
        health_probe=lambda *_args: None,
        port_factory=lambda: 43123,
        device_service=device_service,
    )

    connection = manager.start(model, "test/model")

    assert connection.backend == "cpu"
    assert connection.selection_backend == "auto"
    assert connection.selection_device_ids == ()
    assert connection.detection_error == expected_detection_error
    assert device_service.refresh_requests == [False]
    assert device_service.resolve_requests == []
    manager.stop()


def test_auto_falls_back_to_cpu_when_vulkan_fails_health(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    def health_probe(_base_url, _api_key, process, _timeout):
        if "vulkan" in process.args[0]:
            raise TimeoutError("driver unavailable")

    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=process_factory,
        health_probe=health_probe,
        port_factory=iter((43123, 43124)).__next__,
        device_service=StaticDeviceService(),
    )

    connection = manager.start(model, "test/model")

    assert connection.backend == "cpu"
    assert connection.device_ids == ()
    assert connection.selection_backend == "auto"
    assert connection.selection_device_ids == ()
    assert connection.detection_error == "driver unavailable"
    assert calls[0].terminated is True
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    assert calls[1].args[0] == str(runtime / "cpu" / executable)
    manager.stop()


def test_auto_cached_gpu_recommendation_carries_missing_vulkan_error_to_cpu(
    tmp_path: Path,
) -> None:
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    environment = {"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)}
    device_service = LlamaDeviceService(
        environment=environment,
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout="Available devices:\n  Vulkan0: NVIDIA RTX 4090\n",
            stderr="",
        ),
    )
    cached_inventory = device_service.inventory()
    assert cached_inventory.recommended_backend == "gpu"

    vulkan_executable = runtime / "vulkan" / llama_server_filename()
    vulkan_executable.unlink()
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    manager = LlamaServerManager(
        environment=environment,
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        port_factory=lambda: 43123,
        device_service=device_service,
    )

    connection = manager.start(model, "test/model")

    expected_error = f"llama-server executable does not exist: {vulkan_executable}"
    assert connection.backend == "cpu"
    assert connection.device_ids == ()
    assert connection.selection_backend == "auto"
    assert connection.selection_device_ids == ()
    assert connection.detection_error == expected_error
    assert device_service.inventory() is cached_inventory
    assert len(calls) == 1
    assert Path(calls[0].args[0]).parent.name == "cpu"
    manager.stop()


def test_explicit_cpu_starts_without_gpu_flags_and_reuses_same_runtime(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        port_factory=lambda: 43123,
        device_service=StaticDeviceService(),
    )

    first = manager.start(model, "test/model", backend="cpu")
    second = manager.start(
        model,
        "test/model",
        backend="cpu",
        device_ids=("ignored-for-cpu",),
    )

    assert first is second
    assert first.backend == "cpu"
    assert first.selection_backend == "cpu"
    assert first.selection_device_ids == ()
    assert first.detection_error is None
    assert len(calls) == 1
    assert "--device" not in calls[0].args
    assert "--n-gpu-layers" not in calls[0].args
    manager.stop()


def test_explicit_multiple_gpus_pass_exact_device_list(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    device_service = StaticDeviceService(device_count=2)
    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        port_factory=lambda: 43123,
        device_service=device_service,
    )

    connection = manager.start(
        model,
        "test/model",
        backend="gpu",
        device_ids=("gpu-0", "gpu-1"),
    )

    assert connection.backend == "vulkan"
    assert connection.device_ids == ("gpu-0", "gpu-1")
    assert calls[0].args[calls[0].args.index("--device") + 1] == "Vulkan0,Vulkan1"
    assert device_service.resolve_requests == [("gpu-0", "gpu-1")]
    manager.stop()


def test_operator_forced_vulkan_ignores_saved_devices_and_reuses_runtime(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    manager = LlamaServerManager(
        environment={
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LLAMA_ACCELERATION": "vulkan",
        },
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        port_factory=lambda: 43123,
        device_service=StaticDeviceService(),
    )

    first = manager.start(
        model,
        "test/model",
        backend="cpu",
        device_ids=("gpu-stale",),
    )
    second = manager.start(
        model,
        "test/model",
        backend="gpu",
        device_ids=("gpu-other",),
    )

    assert first is second
    assert first.backend == "vulkan"
    assert first.device_ids == ()
    assert len(calls) == 1
    assert "--device" not in calls[0].args
    manager.stop()


def test_explicit_gpu_does_not_fall_back_to_cpu(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    def fail_health(*_args):
        raise RuntimeError("driver rejected startup")

    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=process_factory,
        health_probe=fail_health,
        port_factory=lambda: 43123,
        device_service=StaticDeviceService(),
    )

    try:
        manager.start(model, "test/model", backend="gpu", device_ids=("gpu-0",))
    except RuntimeError as error:
        assert "driver rejected startup" in str(error)
    else:
        raise AssertionError("explicit GPU startup should fail")

    assert len(calls) == 1
    assert "vulkan" in calls[0].args[0]


def test_unavailable_saved_gpu_fails_and_sets_error_status(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")

    def process_factory(*_args, **_options):
        raise AssertionError("unavailable selection must fail before process startup")

    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        device_service=StaticDeviceService(),
    )

    with pytest.raises(ValueError, match="unavailable"):
        manager.start(model, "test/model", backend="gpu", device_ids=("gpu-missing",))

    status = manager.public_status()
    assert status["status"] == "error"
    assert "unavailable" in status["detail"]


def test_stop_during_device_discovery_preserves_stopped_status(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    entered_inventory = threading.Event()
    release_inventory = threading.Event()
    result = []

    class BlockingFailedDeviceService:
        def inventory(self):
            entered_inventory.set()
            assert release_inventory.wait(timeout=2)
            raise RuntimeError("inventory failed")

    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=lambda *_args, **_options: None,
        health_probe=lambda *_args: None,
        device_service=BlockingFailedDeviceService(),
    )

    def start_and_capture_error():
        try:
            manager.start(model, "test/model")
        except RuntimeError as error:
            result.append(error)

    starter = threading.Thread(target=start_and_capture_error)
    starter.start()
    assert entered_inventory.wait(timeout=2)

    manager.stop()
    release_inventory.set()
    starter.join(timeout=2)

    assert not starter.is_alive()
    assert result and "cancelled" in str(result[0])
    assert manager.public_status()["status"] == "stopped"


def test_same_model_and_artifact_restart_when_device_selection_changes(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    manager = LlamaServerManager(
        environment={"GEIST_LLAMA_RUNTIME_ROOT": str(runtime)},
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        port_factory=iter((43123, 43124)).__next__,
        device_service=StaticDeviceService(device_count=2),
    )

    manager.start(model, "test/model", backend="gpu", device_ids=("gpu-0",))
    manager.start(model, "test/model", backend="gpu", device_ids=("gpu-1",))

    assert len(calls) == 2
    assert calls[0].terminated is True
    assert calls[1].args[calls[1].args.index("--device") + 1] == "Vulkan1"
    manager.stop()


def test_same_model_id_with_different_artifact_restarts_server(tmp_path):
    runtime = _runtime_tree(tmp_path)
    first_model = tmp_path / "first.gguf"
    second_model = tmp_path / "second.gguf"
    first_model.write_bytes(b"GGUFfirst")
    second_model.write_bytes(b"GGUFsecond")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    manager = LlamaServerManager(
        environment={
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LLAMA_ACCELERATION": "cpu",
        },
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        port_factory=iter((43123, 43124)).__next__,
    )

    first = manager.start(first_model, "custom/model")
    second = manager.start(second_model, "custom/model")

    assert first.model_path == str(first_model.resolve())
    assert second.model_path == str(second_model.resolve())
    assert len(calls) == 2
    assert calls[0].terminated is True
    manager.stop()


def test_stop_can_terminate_process_while_health_probe_is_waiting(tmp_path):
    runtime = _runtime_tree(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    entered_health = threading.Event()
    process_holder = []
    result = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        process_holder.append(process)
        return process

    def health_probe(_base_url, _api_key, process, _timeout):
        entered_health.set()
        while process.poll() is None:
            time.sleep(0.01)
        raise RuntimeError("process stopped")

    manager = LlamaServerManager(
        environment={
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LLAMA_ACCELERATION": "cpu",
        },
        process_factory=process_factory,
        health_probe=health_probe,
        port_factory=lambda: 43123,
    )

    def start_and_capture_error():
        try:
            manager.start(model, "test/model")
        except RuntimeError as error:
            result.append(error)

    starter = threading.Thread(target=start_and_capture_error)
    starter.start()
    assert entered_health.wait(timeout=2)

    manager.stop()
    starter.join(timeout=2)

    assert not starter.is_alive()
    assert result and "cancelled" in str(result[0])
    assert process_holder[0].terminated is True
    assert manager.public_status()["status"] == "stopped"
