import subprocess
from unittest.mock import call, patch

from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.factory import create_session_manager
from app.services.execution.session import (
    DockerSessionManager,
    build_session_create_args,
    build_session_exec_args,
    session_container_name,
)


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_session_container_name_is_sanitized():
    assert session_container_name("chat-7") == "geist-exec-chat-7"
    assert session_container_name("a b/c$") == "geist-exec-a-b-c-"


def test_session_create_args_keep_hardening_posture():
    args = build_session_create_args(name="geist-exec-chat-7", image="python:3.11-slim")
    joined = " ".join(args)
    assert args[:4] == ["run", "--detach", "--name", "geist-exec-chat-7"]
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined
    assert "--user 65534:65534" in joined
    assert "--network none" in joined
    assert args[-2:] == ["sleep", "infinity"]


def test_session_exec_args_bound_the_command():
    args = build_session_exec_args(name="geist-exec-chat-7", command="echo hi", timeout=10)
    assert args[:4] == ["exec", "--workdir", "/workspace", "geist-exec-chat-7"]
    assert args[-1].startswith("timeout 10 bash -c ")


def _manager(clock=None):
    env = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")
    ticks = iter(range(0, 10_000))
    return DockerSessionManager(
        env, ttl_seconds=100.0, clock=clock or (lambda: next(ticks))
    )


def test_run_in_session_reuses_running_container():
    manager = _manager()
    responses = [
        _completed(0, stdout="true\n"),  # inspect: running
        _completed(0, stdout="hi\n"),  # exec
    ]
    with patch("subprocess.run", side_effect=responses) as mock_run:
        result = manager.run_in_session("chat-7", "echo hi")

    assert result.exit_code == 0
    assert result.stdout == "hi\n"
    first_argv = mock_run.call_args_list[0].args[0]
    assert first_argv[:2] == ["/usr/bin/docker", "inspect"]
    second_argv = mock_run.call_args_list[1].args[0]
    assert second_argv[1] == "exec"


def test_run_in_session_creates_container_when_missing():
    manager = _manager()
    responses = [
        _completed(1, stderr="No such object"),  # inspect: absent
        _completed(0, stdout="containerid\n"),  # run -d
        _completed(0, stdout="done\n"),  # exec
    ]
    with patch("subprocess.run", side_effect=responses) as mock_run:
        result = manager.run_in_session("chat-7", "echo done")

    assert result.exit_code == 0
    create_argv = mock_run.call_args_list[1].args[0]
    assert create_argv[1] == "run"
    assert "--detach" in create_argv
    assert "geist-exec-chat-7" in create_argv


def test_run_in_session_replaces_stopped_container():
    manager = _manager()
    responses = [
        _completed(0, stdout="false\n"),  # inspect: exists, stopped
        _completed(0),  # rm -f
        _completed(0, stdout="containerid\n"),  # run -d
        _completed(0, stdout="ok\n"),  # exec
    ]
    with patch("subprocess.run", side_effect=responses) as mock_run:
        result = manager.run_in_session("chat-7", "echo ok")

    assert result.exit_code == 0
    rm_argv = mock_run.call_args_list[1].args[0]
    assert rm_argv[1:3] == ["rm", "-f"]


def test_create_failure_surfaces_as_error_result():
    manager = _manager()
    responses = [
        _completed(1, stderr="No such object"),
        _completed(125, stderr="image not found"),
    ]
    with patch("subprocess.run", side_effect=responses):
        result = manager.run_in_session("chat-7", "echo hi")

    assert result.exit_code == 125
    assert "image not found" in result.stderr


def test_idle_sessions_are_reaped_on_next_run():
    now = {"value": 0.0}
    manager = _manager(clock=lambda: now["value"])
    exec_ok = [
        _completed(0, stdout="true\n"),
        _completed(0, stdout="a\n"),
    ]
    with patch("subprocess.run", side_effect=exec_ok):
        manager.run_in_session("chat-1", "echo a")

    # 500s later (ttl=100): running for chat-2 must reap chat-1's container.
    now["value"] = 500.0
    second = [
        _completed(0),  # rm -f geist-exec-chat-1 (reap)
        _completed(0, stdout="true\n"),  # inspect chat-2
        _completed(0, stdout="b\n"),  # exec chat-2
    ]
    with patch("subprocess.run", side_effect=second) as mock_run:
        manager.run_in_session("chat-2", "echo b")

    reap_argv = mock_run.call_args_list[0].args[0]
    assert reap_argv[1:3] == ["rm", "-f"]
    assert "geist-exec-chat-1" in reap_argv


def test_shutdown_removes_all_tracked_sessions():
    manager = _manager()
    with patch("subprocess.run", side_effect=[
        _completed(0, stdout="true\n"),
        _completed(0, stdout="a\n"),
    ]):
        manager.run_in_session("chat-1", "echo a")

    with patch("subprocess.run", return_value=_completed(0)) as mock_run:
        manager.shutdown()

    assert mock_run.call_args_list == [
        call(
            ["/usr/bin/docker", "rm", "-f", "geist-exec-chat-1"],
            capture_output=True,
            text=True,
        )
    ]


def test_factory_persistent_flag_gates_manager(monkeypatch):
    docker_env = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")

    monkeypatch.delenv("GEIST_EXEC_PERSISTENT", raising=False)
    assert create_session_manager(docker_env) is None

    monkeypatch.setenv("GEIST_EXEC_PERSISTENT", "1")
    manager = create_session_manager(docker_env)
    assert isinstance(manager, DockerSessionManager)

    monkeypatch.setenv("GEIST_EXEC_SESSION_TTL_SECONDS", "60")
    assert create_session_manager(docker_env).ttl_seconds == 60.0

    # Local backend never gets persistence.
    from app.services.execution.local import LocalExecutionEnvironment

    assert create_session_manager(LocalExecutionEnvironment()) is None
    assert create_session_manager(None) is None
