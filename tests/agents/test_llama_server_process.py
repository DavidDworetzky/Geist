"""Process lifecycle tests for the private llama-server supervisor."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from agents.architectures.llama_server_process import (
    LlamaServerManager,
    default_llama_runtime_root,
    llama_server_runtime_available,
)
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
        server = directory / executable
        server.write_bytes(b"binary")
        server.chmod(0o755)
    return root


def test_default_runtime_root_uses_geist_home_sibling(tmp_path: Path) -> None:
    assert default_llama_runtime_root({"GEIST_HOME": str(tmp_path / "geist")}) == (
        tmp_path / "geist-runtime" / "llama.cpp"
    )


def test_explicit_runtime_root_overrides_packaged_default(tmp_path: Path) -> None:
    configured = tmp_path / "operator-runtime"

    assert (
        default_llama_runtime_root(
            {
                "GEIST_HOME": str(tmp_path / "geist"),
                "GEIST_LLAMA_RUNTIME_ROOT": str(configured),
            }
        )
        == configured
    )


def test_frozen_runtime_defaults_beside_the_application(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "geist" / "geist"))

    assert default_llama_runtime_root({"GEIST_HOME": "/ignored"}) == (
        tmp_path / "geist" / "runtimes" / "llama.cpp"
    )


def test_runtime_availability_requires_an_executable_server(tmp_path: Path) -> None:
    runtime = _runtime_tree(tmp_path)
    server = runtime / "cpu" / ("llama-server.exe" if os.name == "nt" else "llama-server")

    assert llama_server_runtime_available(
        {
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LLAMA_ACCELERATION": "cpu",
        }
    )

    if os.name != "nt":
        server.chmod(0o644)
        assert not llama_server_runtime_available(
            {
                "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
                "GEIST_LLAMA_ACCELERATION": "cpu",
            }
        )


def test_manager_uses_packaged_default_runtime(tmp_path: Path) -> None:
    packaged_root = tmp_path / "geist-runtime" / "llama.cpp"
    runtime = _runtime_tree(tmp_path)
    packaged_root.parent.mkdir()
    runtime.rename(packaged_root)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFtest")
    calls = []

    def process_factory(args, **_options):
        process = FakeProcess(args)
        calls.append(process)
        return process

    manager = LlamaServerManager(
        environment={
            "GEIST_HOME": str(tmp_path / "geist"),
            "GEIST_LLAMA_ACCELERATION": "cpu",
        },
        process_factory=process_factory,
        health_probe=lambda *_args: None,
        port_factory=lambda: 43123,
    )

    connection = manager.start(model, "test/model")

    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    assert connection.backend == "cpu"
    assert calls[0].args[0] == str(packaged_root / "cpu" / executable)
    manager.stop()


def test_platform_process_options_are_isolated() -> None:
    class WindowsSubprocess:
        CREATE_NEW_PROCESS_GROUP = 0x200
        CREATE_NO_WINDOW = 0x8000000

    assert posix_options() == {"start_new_session": True}
    assert windows_options(WindowsSubprocess) == {"creationflags": 0x8000200}


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

    manager = LlamaServerManager(
        environment={
            "GEIST_LLAMA_RUNTIME_ROOT": str(runtime),
            "GEIST_LLAMA_ACCELERATION": "auto",
        },
        process_factory=process_factory,
        health_probe=health_probe,
        port_factory=lambda: 43123,
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
    assert manager.public_status()["status"] == "ready"
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
    )

    connection = manager.start(model, "test/model")

    assert connection.backend == "cpu"
    assert calls[0].terminated is True
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    assert calls[1].args[0] == str(runtime / "cpu" / executable)
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
