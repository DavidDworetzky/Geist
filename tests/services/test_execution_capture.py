import os
import subprocess
import tracemalloc
from unittest.mock import patch

import pytest

from app.services.execution.base import MAX_OUTPUT_CHARS, ExecutionResult
from app.services.execution.capture import MAX_CAPTURE_BYTES, capture_process
from app.services.execution.docker import DockerExecutionEnvironment, build_docker_run_args
from app.services.execution.local import LocalExecutionEnvironment


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
@pytest.mark.parametrize("redirect", ["", " >&2"])
def test_fifty_megabyte_producer_is_stopped_with_bounded_host_allocation(redirect):
    tracemalloc.start()
    try:
        result = LocalExecutionEnvironment().run("head -c 50000000 /dev/zero" + redirect, 5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.truncated
    assert result.exit_code == 125
    assert len(result.stdout) <= MAX_OUTPUT_CHARS
    assert len(result.stderr) <= MAX_OUTPUT_CHARS
    assert peak < MAX_CAPTURE_BYTES * 32


def test_capture_reaps_child_and_keeps_output_before_timeout():
    process = subprocess.Popen(
        ["bash", "-c", "printf before; exec sleep 5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    result = capture_process(process, 0.1, process.kill)
    assert result.timed_out
    assert result.stdout == "before"
    assert process.returncode is not None


def test_docker_output_overflow_removes_its_container():
    environment = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")
    with (
        patch("subprocess.Popen") as launch,
        patch(
            "app.services.execution.docker.capture_process",
            return_value=ExecutionResult(125, "bounded", "", 0.1, truncated=True),
        ),
        patch("subprocess.run") as cleanup,
    ):
        result = environment.run("yes")
    argv = launch.call_args.args[0]
    name = argv[argv.index("--name") + 1]
    assert result.truncated
    assert cleanup.call_args.args[0] == ["/usr/bin/docker", "rm", "--force", name]


def test_container_args_do_not_inherit_environment():
    args = build_docker_run_args(image="test", command="echo hi")
    assert not set(args) & {"-e", "--env", "--env-file"}


def test_podman_uses_same_hardening_and_capture():
    environment = DockerExecutionEnvironment(runtime_path="/usr/bin/podman")
    with (
        patch("subprocess.Popen") as launch,
        patch(
            "app.services.execution.docker.capture_process",
            return_value=ExecutionResult(0, "hi", "", 0.1),
        ),
    ):
        result = environment.run("echo hi")
    argv = launch.call_args.args[0]
    assert argv[0] == "/usr/bin/podman"
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert result.stdout == "hi"
