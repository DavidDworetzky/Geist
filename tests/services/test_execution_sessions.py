import sys
import threading
import time

import pytest

from app.services.execution.base import ExecutionResult
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.factory import create_session_manager
from app.services.execution.session import (
    DockerSessionManager,
    build_session_create_args,
    build_session_exec_args,
    session_container_name,
)


def _result(code=0, stdout="", stderr="", *, timed_out=False):
    return ExecutionResult(code, stdout, stderr, 0, timed_out=timed_out)


class FakeRuntime:
    def __init__(self, manager):
        self.manager = manager
        self.containers = {}
        self.calls = []

    def __call__(self, runtime, args, deadline):
        assert deadline > time.monotonic()
        self.calls.append(args)
        if args[0] == "inspect":
            value = self.containers.get(args[-1])
            return _result(1) if value is None else _result(stdout=value)
        if args[0] == "run":
            self.containers[args[args.index("--name") + 1]] = f"{self.manager.owner} true"
            return _result(stdout="container-id")
        if args[0] == "rm":
            self.containers.pop(args[-1], None)
            return _result()
        if args[0] == "container":
            return _result()
        return _result(stdout="done\n")


def _manager(monkeypatch, **kwargs):
    manager = DockerSessionManager(DockerExecutionEnvironment(runtime_path="/docker"), **kwargs)
    runtime = FakeRuntime(manager)
    monkeypatch.setattr(manager, "_runtime_call", runtime)
    return manager, runtime


def test_names_are_safe_without_sanitization_collisions():
    assert session_container_name("a/b", "owner") != session_container_name("a-b", "owner")
    assert session_container_name("a/b", "one") != session_container_name("a/b", "two")
    assert "/" not in session_container_name("a/b", "owner")


def test_bounded_file_tool_input_and_output_round_trip():
    payload = "hello \u2603" * 20_000
    result = DockerSessionManager._runtime_call(
        sys.executable,
        ["-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        time.monotonic() + 10,
        input_text=payload,
        output_limit=200_000,
    )
    assert result.exit_code == 0
    assert result.stdout == payload
    assert not result.truncated


def test_oversized_stdin_is_rejected_before_starting_process(monkeypatch):
    started = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: started.append(True))
    result = DockerSessionManager._runtime_call(
        sys.executable,
        ["-"],
        time.monotonic() + 10,
        input_text="x" * 1_500_001,
    )
    assert result.exit_code == 125
    assert started == []


def test_interactive_file_call_retains_hardened_session_lifecycle(monkeypatch):
    manager, runtime = _manager(monkeypatch)
    inputs = []

    def capture(binary, args, deadline, **kwargs):
        if args[0] == "exec":
            inputs.append((args, kwargs))
        return runtime(binary, args, deadline)

    monkeypatch.setattr(manager, "_runtime_call", capture)
    assert (
        manager.run_in_session(
            "chat", "python -", input_text="print(1)", output_limit=200_000
        ).exit_code
        == 0
    )
    assert "--interactive" in inputs[0][0]
    assert inputs[0][1] == {"input_text": "print(1)", "output_limit": 200_000}
    creation = next(args for args in runtime.calls if args[0] == "run")
    assert "--read-only" in creation
    assert "--cap-drop" in creation
    assert "none" in creation


def test_session_create_args_keep_hardening_and_bound_crash_lifetime():
    args = build_session_create_args(name="test", image="python:3.11-slim", owner="owner")
    joined = " ".join(args)
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined
    assert "--user 65534:65534" in joined
    assert "--network none" in joined
    assert "--label org.geist.session-owner=owner" in joined
    assert "--rm" in args
    assert args[-2:] == ["sleep", "86400"]
    mounted = build_session_create_args(name="test", image="image", workspace="/safe/path")
    assert "--volume" not in mounted
    assert "type=bind,source=/safe/path,target=/workspace" in mounted
    with pytest.raises(ValueError):
        build_session_create_args(name="test", image="image", workspace="/bad,path")


def test_session_exec_has_term_to_kill_deadline():
    args = build_session_exec_args(name="test", command="echo hi", timeout=10)
    assert args[:4] == ["exec", "--workdir", "/workspace", "test"]
    assert args[-1].startswith(
        "export HOME=/tmp XDG_CACHE_HOME=/tmp; timeout --kill-after=1 10 bash -c "
    )


def test_creates_once_and_reuses_running_container(monkeypatch):
    manager, runtime = _manager(monkeypatch)
    assert manager.run_in_session("chat", "echo hi").stdout == "done\n"
    assert manager.run_in_session("chat", "echo again").exit_code == 0
    assert [args[0] for args in runtime.calls] == ["inspect", "run", "exec", "inspect", "exec"]


def test_owned_stopped_container_is_replaced(monkeypatch):
    manager, runtime = _manager(monkeypatch)
    manager.run_in_session("chat", "true")
    name = next(iter(runtime.containers))
    runtime.containers[name] = f"{manager.owner} false"
    runtime.calls.clear()
    assert manager.run_in_session("chat", "true").exit_code == 0
    assert [args[0] for args in runtime.calls] == ["inspect", "rm", "run", "exec"]


def test_foreign_container_is_never_reused_or_removed(monkeypatch):
    manager, runtime = _manager(monkeypatch)
    manager.run_in_session("chat", "true")
    runtime.containers[next(iter(runtime.containers))] = "another-owner true"
    runtime.calls.clear()
    result = manager.run_in_session("chat", "true")
    assert result.exit_code == 125
    assert "ownership mismatch" in result.stderr
    assert all(args[0] == "inspect" for args in runtime.calls)


def test_create_failure_surfaces_and_absence_releases_capacity(monkeypatch):
    manager, runtime = _manager(monkeypatch, max_sessions=1)

    def fail_create(binary, args, deadline):
        if args[0] == "run":
            return _result(125, stderr="image not found")
        return runtime(binary, args, deadline)

    monkeypatch.setattr(manager, "_runtime_call", fail_create)
    result = manager.run_in_session("chat", "true")
    assert result.exit_code == 125
    assert "image not found" in result.stderr
    assert manager._sessions == {}


def test_idle_reaping_and_shutdown_only_remove_owned_sessions(monkeypatch):
    now = [0.0]
    manager, runtime = _manager(monkeypatch, ttl_seconds=10, clock=lambda: now[0])
    manager.run_in_session("one", "true")
    first = next(iter(runtime.containers))
    now[0] = 11
    manager.run_in_session("two", "true")
    assert first not in runtime.containers
    manager.shutdown()
    assert runtime.containers == {}
    assert manager.run_in_session("three", "true").exit_code == 125


def test_session_cap_and_close_scope(monkeypatch):
    manager, runtime = _manager(monkeypatch, max_sessions=1)
    manager.run_in_session("one", "true")
    assert manager.run_in_session("two", "true").exit_code == 125
    manager.close_scope("one")
    assert not runtime.containers
    assert manager.run_in_session("two", "true").exit_code == 0


def test_retired_scope_retries_failed_cleanup_without_another_chat(monkeypatch):
    manager, runtime = _manager(monkeypatch, max_sessions=1)
    manager.run_in_session("chat", "true")

    def failing_remove(binary, args, deadline):
        return (
            _result(1, stderr="daemon busy") if args[0] == "rm" else runtime(binary, args, deadline)
        )

    monkeypatch.setattr(manager, "_runtime_call", failing_remove)
    manager.close_scope("chat")
    assert manager._sessions["chat"].retired
    monkeypatch.setattr(manager, "_runtime_call", runtime)
    assert manager.run_in_session("chat", "true").exit_code == 0
    assert not manager._sessions["chat"].retired


def test_session_manager_can_restart_after_clean_shutdown(monkeypatch):
    manager, _ = _manager(monkeypatch)
    manager.run_in_session("chat", "true")
    manager.shutdown()
    manager.startup()
    assert manager.run_in_session("chat", "true").exit_code == 0


def test_first_run_session_moves_to_chat_without_losing_its_container(monkeypatch):
    manager, runtime = _manager(monkeypatch)
    manager.run_in_session("workspace:1:run:new", "true")
    names = set(runtime.containers)
    manager.promote_scope("workspace:1:run:new", "workspace:1:chat:42")
    assert "workspace:1:run:new" not in manager._sessions
    manager.run_in_session("workspace:1:chat:42", "true")
    assert set(runtime.containers) == names


def test_hardline_policy_precedes_any_host_reaching_runtime_call(monkeypatch, tmp_path):
    manager, runtime = _manager(monkeypatch)
    manager.environment = DockerExecutionEnvironment(
        workspace=str(tmp_path), runtime_path="/docker"
    )
    result = manager.run_in_session("chat", "rm -rf /")
    assert result.blocked
    assert runtime.calls == []


@pytest.mark.parametrize("command", ["rm -rf /workspace", "git reset --hard", "git clean -fd"])
def test_persistent_ephemeral_workspace_also_retains_hardline_floor(monkeypatch, command):
    manager, runtime = _manager(monkeypatch)
    assert manager.run_in_session("chat", command).blocked
    assert runtime.calls == []


def test_same_scope_serializes_and_active_execution_is_not_reaped(monkeypatch):
    now = [0.0]
    manager, runtime = _manager(monkeypatch, ttl_seconds=10, clock=lambda: now[0])
    entered = threading.Event()
    release = threading.Event()
    active = [0]
    maximum = [0]

    def slow_exec(binary, args, deadline):
        if args[0] == "exec":
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
            entered.set()
            assert release.wait(2)
            active[0] -= 1
        return runtime(binary, args, deadline)

    monkeypatch.setattr(manager, "_runtime_call", slow_exec)
    first = threading.Thread(target=manager.run_in_session, args=("chat", "true"))
    second = threading.Thread(target=manager.run_in_session, args=("chat", "true"))
    first.start()
    assert entered.wait(1)
    now[0] = 100
    manager._reap_idle("/docker", deadline=time.monotonic() + 1)
    assert not any(args[0] == "rm" for args in runtime.calls)
    second.start()
    release.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert maximum[0] == 1
    assert len([args for args in runtime.calls if args[0] == "run"]) == 1


def test_runtime_capture_bounds_output_and_timeout():
    result = DockerSessionManager._runtime_call(
        sys.executable,
        ["-c", "print('x' * 200000)"],
        time.monotonic() + 5,
    )
    assert result.exit_code == 0
    assert result.truncated and len(result.stdout) <= 65536
    started = time.monotonic()
    timeout = DockerSessionManager._runtime_call(
        sys.executable,
        ["-c", "import time; time.sleep(30)"],
        started + 0.1,
    )
    assert timeout.timed_out
    assert time.monotonic() - started < 2


def test_factory_persistent_flag_and_invalid_limits(monkeypatch):
    environment = DockerExecutionEnvironment(runtime_path="/docker")
    monkeypatch.delenv("GEIST_EXEC_PERSISTENT", raising=False)
    assert create_session_manager(environment) is None
    monkeypatch.setenv("GEIST_EXEC_PERSISTENT", "1")
    monkeypatch.setenv("GEIST_EXEC_SESSION_TTL_SECONDS", "60")
    assert create_session_manager(environment).ttl_seconds == 60
    for value in ("nan", "inf", "0", "-1", "bad"):
        monkeypatch.setenv("GEIST_EXEC_SESSION_TTL_SECONDS", value)
        assert create_session_manager(environment) is None
    assert create_session_manager(None) is None
