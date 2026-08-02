import subprocess
from unittest.mock import patch

import pytest

from app.services.execution.base import (
    MAX_COMMAND_TIMEOUT_SECONDS,
    clamp_timeout,
    truncate_output,
)
from app.services.execution.docker import (
    DEFAULT_IMAGE,
    DockerExecutionEnvironment,
    build_docker_run_args,
)
from app.services.execution.factory import create_execution_environment
from app.services.execution.hardline import detect_hardline_command
from app.services.execution.local import LocalExecutionEnvironment, scrub_environment
from app.services.tool_registry import build_default_tool_registry


# ---------------------------------------------------------------------------
# base helpers


def test_clamp_timeout_bounds():
    assert clamp_timeout(None) == 30
    assert clamp_timeout(0) == 1
    assert clamp_timeout(15) == 15
    assert clamp_timeout(10_000) == MAX_COMMAND_TIMEOUT_SECONDS


def test_truncate_output_marks_dropped_content():
    text, truncated = truncate_output("x" * 50, limit=20)
    assert truncated is True
    assert text.endswith("[output truncated]")
    assert len(text) == 20

    text, truncated = truncate_output("short")
    assert truncated is False
    assert text == "short"


# ---------------------------------------------------------------------------
# hardline floor


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "sudo rm -fr /*",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown -h now",
        "reboot",
        ":(){ :|:& };:",
        "kill -9 -1",
        "echo hi; rm -rf /",
        "echo done > /dev/sda",
    ],
)
def test_hardline_blocks_unrecoverable_commands(command):
    assert detect_hardline_command(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "rm -rf ./build",
        "rm -rf /tmp/scratch",
        "echo 'rm -rf /' is dangerous",
        "ddgr search",
        "git checkout -- .",
        "mkdir -p /tmp/x && rm -r /tmp/x",
    ],
)
def test_hardline_allows_ordinary_commands(command):
    assert detect_hardline_command(command) is None


# ---------------------------------------------------------------------------
# local backend


def test_local_runs_command_and_captures_output():
    env = LocalExecutionEnvironment()
    result = env.run("echo hello && echo err >&2 && exit 3")
    assert result.exit_code == 3
    assert result.stdout.strip() == "hello"
    assert result.stderr.strip() == "err"
    assert env.is_sandboxed is False


def test_local_blocks_hardline_before_execution():
    env = LocalExecutionEnvironment()
    result = env.run("rm -rf /")
    assert result.exit_code == 126
    assert "BLOCKED" in result.stderr


def test_local_times_out():
    env = LocalExecutionEnvironment()
    result = env.run("sleep 5", timeout_seconds=1)
    assert result.timed_out is True
    assert result.exit_code == 124


def test_local_scrubs_credentials_from_child_env():
    env = LocalExecutionEnvironment()
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-secret", "MY_TOKEN": "t", "HARMLESS_VAR": "ok"},
    ):
        result = env.run("echo openai=${OPENAI_API_KEY:-unset} plain=${HARMLESS_VAR:-unset}")
    assert "openai=unset" in result.stdout
    assert "plain=ok" in result.stdout


def test_scrub_environment_name_matching():
    scrubbed = scrub_environment(
        {
            "AWS_SECRET_ACCESS_KEY": "x",
            "DATABASE_URL": "postgres://",
            "GITHUB_TOKEN": "x",
            "PASSWORD": "x",
            "PATH": "/usr/bin",
            "LANG": "C",
        }
    )
    assert set(scrubbed) == {"PATH", "LANG"}


# ---------------------------------------------------------------------------
# docker backend (no runtime required: assert on constructed argv)


def test_docker_run_args_hardening_posture():
    args = build_docker_run_args(image=DEFAULT_IMAGE, command="echo hi")
    joined = " ".join(args)
    assert args[:2] == ["run", "--rm"]
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined
    assert "--user 65534:65534" in joined
    assert "--network none" in joined
    assert "--pids-limit 256" in joined
    assert "/workspace:rw,nosuid,size=256m,mode=0777" in joined
    assert "--volume" not in joined
    assert args[-4:] == [DEFAULT_IMAGE, "bash", "-c", "echo hi"]


def test_docker_run_args_workspace_mount_replaces_tmpfs():
    args = build_docker_run_args(
        image=DEFAULT_IMAGE, command="ls", workspace="/home/user/project"
    )
    joined = " ".join(args)
    assert "--volume /home/user/project:/workspace" in joined
    assert "/workspace:rw,nosuid" not in joined


def test_docker_run_args_network_opt_in():
    args = build_docker_run_args(image=DEFAULT_IMAGE, command="curl x", network=True)
    assert "none" not in args


def test_docker_sandbox_posture_flips_with_workspace():
    sandboxed = DockerExecutionEnvironment()
    assert sandboxed.is_sandboxed is True
    host_reaching = DockerExecutionEnvironment(workspace="/home/user/project")
    assert host_reaching.is_sandboxed is False
    assert host_reaching.has_host_access is True


def test_docker_reports_missing_runtime():
    env = DockerExecutionEnvironment(runtime_path=None)
    with patch(
        "app.services.execution.docker.find_container_runtime", return_value=None
    ):
        result = env.run("echo hi")
    assert result.exit_code == 127
    assert "container runtime" in result.stderr


def test_docker_run_invokes_runtime_with_bounded_command():
    env = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="hi\n", stderr=""
    )
    with patch("subprocess.run", return_value=completed) as mock_run:
        result = env.run("echo hi", timeout_seconds=10)

    assert result.exit_code == 0
    assert result.stdout == "hi\n"
    argv = mock_run.call_args.args[0]
    assert argv[0] == "/usr/bin/docker"
    assert argv[-1].startswith("timeout 10 bash -c ")
    assert mock_run.call_args.kwargs["timeout"] == 30  # command bound + overhead


def test_docker_inner_timeout_maps_to_timed_out():
    env = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")
    completed = subprocess.CompletedProcess(args=[], returncode=124, stdout="", stderr="")
    with patch("subprocess.run", return_value=completed):
        result = env.run("sleep 999", timeout_seconds=1)
    assert result.timed_out is True
    assert result.exit_code == 124


# ---------------------------------------------------------------------------
# factory


def test_factory_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GEIST_EXEC_BACKEND", raising=False)
    assert create_execution_environment() is None


def test_factory_unknown_backend_disables(monkeypatch):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "warpdrive")
    assert create_execution_environment() is None


def test_factory_builds_docker_with_options(monkeypatch):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setenv("GEIST_EXEC_DOCKER_IMAGE", "alpine:3")
    monkeypatch.setenv("GEIST_EXEC_DOCKER_NETWORK", "true")
    monkeypatch.setenv("GEIST_EXEC_WORKSPACE", "/srv/work")
    env = create_execution_environment()
    assert isinstance(env, DockerExecutionEnvironment)
    assert env.image == "alpine:3"
    assert env.network is True
    assert env.workspace == "/srv/work"
    assert env.is_sandboxed is False


def test_factory_builds_local(monkeypatch):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "local")
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    env = create_execution_environment()
    assert isinstance(env, LocalExecutionEnvironment)


# ---------------------------------------------------------------------------
# registry wiring


def test_registry_omits_terminal_tool_when_backend_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("GEIST_EXEC_BACKEND", raising=False)
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))
    registry = build_default_tool_registry()
    assert registry.get("terminal.run") is None


def test_registry_sandboxed_docker_tool_needs_no_approval(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))
    registry = build_default_tool_registry()
    definition = registry.get("terminal.run")
    assert definition is not None
    assert definition.requires_approval is False
    assert definition.enabled_by_default is False
    assert definition.side_effect == "process"


def test_registry_host_reaching_backends_require_approval(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))

    monkeypatch.setenv("GEIST_EXEC_BACKEND", "local")
    assert build_default_tool_registry().get("terminal.run").requires_approval is True

    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setenv("GEIST_EXEC_WORKSPACE", str(tmp_path))
    assert build_default_tool_registry().get("terminal.run").requires_approval is True
