import os
import subprocess
import threading
import tracemalloc
from unittest.mock import Mock, patch

import pytest

from app.services.execution.base import MAX_OUTPUT_CHARS, ExecutionResult
from app.services.execution.capture import MAX_CAPTURE_BYTES, capture_process
from app.services.execution.docker import DockerExecutionEnvironment, build_docker_run_args
from app.services.execution.local import LocalExecutionEnvironment


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
@pytest.mark.parametrize("redirect", ["", " >&2"])
def test_fifty_megabyte_producer_completes_with_bounded_host_allocation(redirect):
    tracemalloc.start()
    try:
        # Scheduling allowance for the memory regression, not an output-volume cap.
        result = LocalExecutionEnvironment().run("head -c 50000000 /dev/zero" + redirect, 15)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.truncated
    assert result.exit_code == 0
    assert len(result.stdout) <= MAX_OUTPUT_CHARS
    assert len(result.stderr) <= MAX_OUTPUT_CHARS
    assert peak < MAX_CAPTURE_BYTES * 32


@pytest.mark.skipif(os.name != "posix", reason="POSIX shell")
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
    assert process.stdout.closed and process.stderr.closed


def test_verbose_docker_output_does_not_trigger_container_cleanup():
    environment = DockerExecutionEnvironment(runtime_path="/usr/bin/docker")
    with (
        patch("subprocess.Popen"),
        patch(
            "app.services.execution.docker.capture_process",
            return_value=ExecutionResult(3, "bounded", "", 0.1, truncated=True),
        ),
        patch("subprocess.run") as cleanup,
    ):
        result = environment.run("verbose build")
    assert result.truncated
    assert result.exit_code == 3
    cleanup.assert_not_called()


@pytest.mark.parametrize("count", [30000, 1000000])
@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_verbose_local_command_retains_real_exit_status(count):
    result = LocalExecutionEnvironment().run(f"head -c {count} /dev/zero; exit 3", 5)
    assert result.exit_code == 3
    assert result.truncated and not result.timed_out


def test_only_reader_closes_a_pipe_held_by_a_descendant():
    release = threading.Event()
    closed = threading.Event()
    stdout = Mock()
    stdout.fileno.return_value = 42
    stdout.close.side_effect = closed.set
    stderr = Mock()
    stderr.fileno.return_value = 43
    process = Mock(stdout=stdout, stderr=stderr, returncode=0)
    process.poll.return_value = 0

    def delayed_read(descriptor, size):
        if descriptor == 42:
            release.wait(2)
        return b""

    with (
        patch("app.services.execution.capture.os.read", side_effect=delayed_read),
        patch(
            "app.services.execution.capture.select.select",
            side_effect=lambda r, w, x, t: (r, [], []),
        ),
    ):
        try:
            result = capture_process(process, 0.02, lambda: None)
            assert result.timed_out
            stdout.close.assert_not_called()
        finally:
            release.set()
            assert closed.wait(1)
    stdout.close.assert_called_once()


@pytest.mark.skipif(os.name != "posix", reason="POSIX pipe polling")
def test_timeout_closes_readers_even_while_other_process_holds_pipe():
    read_fd, write_fd = os.pipe()
    error_fd, error_write_fd = os.pipe()
    stdout = os.fdopen(read_fd, "rb", buffering=0)
    stderr = os.fdopen(error_fd, "rb", buffering=0)
    process = Mock(stdout=stdout, stderr=stderr, returncode=0)
    process.poll.return_value = 0
    try:
        result = capture_process(process, 0.02, lambda: None)
        assert result.timed_out
        assert stdout.closed and stderr.closed
    finally:
        os.close(write_fd)
        os.close(error_write_fd)


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
