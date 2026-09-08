import json
import os
import subprocess
import time
from unittest.mock import patch

import pytest

from agents.models.tool_calling import ToolCall, ToolContext, tool_requires_approval
from app.services.execution.base import ExecutionResult, truncate_output
from app.services.execution.docker import DockerExecutionEnvironment, build_docker_run_args
from app.services.execution.factory import create_execution_environment
from app.services.execution.local import LocalExecutionEnvironment, scrub_environment
from app.services.tool_registry import build_default_tool_registry


@pytest.fixture(autouse=True)
def isolated_execution_configuration(monkeypatch, tmp_path):
    for name in (
        "GEIST_EXEC_BACKEND",
        "GEIST_EXEC_WORKSPACE",
        "GEIST_EXEC_DOCKER_NETWORK",
        "GEIST_EXEC_RUNTIME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))
    monkeypatch.setenv("GEIST_ENABLED_CHAT_TOOLS", "terminal.run")


@pytest.mark.parametrize("mode", ["default", "auto_approve", "require_approval"])
@pytest.mark.parametrize(
    "backend,network,mounted",
    [("local", False, False), ("docker", True, False), ("docker", False, True)],
)
def test_external_access_requires_fresh_approval(
    monkeypatch, tmp_path, mode, backend, network, mounted
):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", backend)
    monkeypatch.setenv("GEIST_EXEC_DOCKER_NETWORK", str(int(network)))
    if mounted:
        monkeypatch.setenv("GEIST_EXEC_WORKSPACE", str(tmp_path))
    registry = build_default_tool_registry()
    context = ToolContext(
        1, 1, "test", permission_mode=mode, always_allow_tools=frozenset({"terminal.run"})
    )
    definition = registry.get("terminal.run")
    assert definition.allows_standing_grant is False
    assert tool_requires_approval(definition, context)
    with (
        patch.object(DockerExecutionEnvironment, "is_available", return_value=True),
        patch.object(LocalExecutionEnvironment, "run") as local_run,
        patch.object(DockerExecutionEnvironment, "run") as docker_run,
    ):
        result = registry.execute(ToolCall.create("terminal.run", {"command": "echo hi"}), context)
    assert result.status == "awaiting_approval"
    local_run.assert_not_called()
    docker_run.assert_not_called()


def test_docker_host_mount_blocks_before_runtime(tmp_path):
    environment = DockerExecutionEnvironment(workspace=str(tmp_path))
    with patch("subprocess.run") as run:
        result = environment.run("rm -rf /")
    assert result.blocked and result.exit_code == 126
    run.assert_not_called()


def test_mount_free_docker_keeps_hardline_scoped_to_host():
    environment = DockerExecutionEnvironment()
    assert environment.command_rejection_reason("rm -rf /") is None


def test_container_root_is_read_only_and_podman_posture_is_named():
    args = build_docker_run_args(image="python:3.11-slim", command="true")
    assert "--read-only" in args
    assert "/workspace:rw,nosuid,size=256m,mode=0777" in args
    assert (
        DockerExecutionEnvironment(runtime_preference="/usr/bin/podman")
        .describe()
        .startswith("podman")
    )


def test_mandatory_approval_changes_definition_fingerprint():
    from dataclasses import replace

    definition = build_default_tool_registry().get("terminal.run")
    assert (
        replace(
            definition, requires_per_call_approval=not definition.requires_per_call_approval
        ).approval_fingerprint()
        != definition.approval_fingerprint()
    )


def test_docker_timeout_cleans_up_only_its_named_container():
    environment = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")
    with (
        patch("subprocess.Popen") as launch,
        patch(
            "app.services.execution.docker.capture_process",
            return_value=ExecutionResult(124, "partial", "", 1, timed_out=True),
        ),
        patch("subprocess.run") as run,
    ):
        result = environment.run("sleep 10", timeout_seconds=1)
    argv = launch.call_args.args[0]
    name = argv[argv.index("--name") + 1]
    assert name.startswith("geist-exec-")
    assert run.call_args.args[0] == ["/usr/bin/docker", "rm", "--force", name]
    assert run.call_args.kwargs["timeout"] == 5
    assert run.call_args.kwargs["stdout"] == subprocess.DEVNULL
    assert result.stdout == "partial"
    assert result.timed_out


@pytest.mark.parametrize(
    "failure", [OSError("unavailable"), subprocess.TimeoutExpired("cleanup", 5)]
)
def test_docker_cleanup_failure_returns_timeout(failure):
    environment = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")
    with (
        patch("subprocess.Popen"),
        patch(
            "app.services.execution.docker.capture_process",
            return_value=ExecutionResult(124, "", "", 1, timed_out=True),
        ),
        patch("subprocess.run", side_effect=failure),
    ):
        assert environment.run("sleep 10", timeout_seconds=1).timed_out


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_local_timeout_stops_descendant_without_waiting_for_its_pipes(tmp_path):
    environment = LocalExecutionEnvironment(workdir=str(tmp_path))
    started = time.monotonic()
    result = environment.run("(sleep 3; touch escaped-marker) & wait", timeout_seconds=1)
    assert result.timed_out
    assert time.monotonic() - started < 3
    time.sleep(2.5)
    assert not (tmp_path / "escaped-marker").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_local_missing_workdir_returns_structured_error(tmp_path):
    result = LocalExecutionEnvironment(str(tmp_path / "missing")).run("echo hi")
    assert result.exit_code == 127
    assert result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_local_honors_relative_workdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project").mkdir()
    environment = LocalExecutionEnvironment("project")
    assert environment.workdir == str(tmp_path / "project")
    assert environment.run("pwd").stdout.strip() == str(tmp_path / "project")


def test_scrub_additional_credential_names():
    assert scrub_environment(
        {"AWS_ACCESS_KEY_ID": "x", "GITHUB_PAT": "x", "SSH_AUTH_SOCK": "x", "PATH": "/usr/bin"}
    ) == {"PATH": "/usr/bin"}


@pytest.mark.parametrize("limit", [-1, 0, 1, 18, 19, 20])
def test_small_output_limits_are_bounded(limit):
    text, truncated = truncate_output("x" * 50, limit)
    assert truncated
    assert len(text) <= max(0, limit)


@pytest.mark.parametrize("workspace", ["relative", "/tmp/ambiguous,path"])
def test_docker_mount_builder_rejects_ambiguous_sources(workspace):
    with pytest.raises(ValueError):
        build_docker_run_args(image="test", command="echo hi", workspace=workspace)


def test_factory_invalid_workspace_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setenv("GEIST_EXEC_WORKSPACE", str(tmp_path / "missing"))
    assert create_execution_environment() is None


def test_factory_resolves_relative_mount(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setenv("GEIST_EXEC_WORKSPACE", "project")
    assert create_execution_environment().workspace == str(tmp_path / "project")


def test_registry_policy_refusal_is_failed_not_successful(monkeypatch):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "local")
    registry = build_default_tool_registry()
    call = ToolCall.create("terminal.run", {"command": "rm -rf /"})
    context = ToolContext(1, 1, "test", approved_call_ids=frozenset({call.id}))
    result = registry.execute(call, context)
    assert result.status == "failed"
    assert result.error == "policy_blocked"
    assert "blocked by policy" in result.summary
    assert json.loads(result.content)["blocked"] is True


def test_registry_preserves_ordinary_exit_status(monkeypatch):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "local")
    registry = build_default_tool_registry()
    call = ToolCall.create("terminal.run", {"command": "exit 3"})
    context = ToolContext(1, 1, "test", approved_call_ids=frozenset({call.id}))
    with patch.object(
        LocalExecutionEnvironment,
        "run",
        return_value=ExecutionResult(3, "out", "err", 0.2, truncated=True),
    ):
        result = registry.execute(call, context)
    assert result.status == "succeeded"
    assert json.loads(result.content)["exit_code"] == 3
    assert json.loads(result.content)["truncated"] is True


def test_missing_pinned_runtime_hides_terminal_from_model(monkeypatch):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setenv("GEIST_EXEC_RUNTIME", "missing-runtime")
    registry = build_default_tool_registry()
    with patch("app.services.execution.docker.find_container_runtime", return_value=None):
        names = [tool.name for tool in registry.definitions_for_context(ToolContext(1, 1, "test"))]
    assert "terminal.run" not in names


def test_runtime_timeout_and_cleanup_fit_registry_budget(monkeypatch):
    from app.services.execution.base import MAX_COMMAND_TIMEOUT_SECONDS
    from app.services.execution.docker import _DOCKER_OVERHEAD_SECONDS

    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    definition = build_default_tool_registry().get("terminal.run")
    assert definition.timeout_seconds > MAX_COMMAND_TIMEOUT_SECONDS + _DOCKER_OVERHEAD_SECONDS + 5
