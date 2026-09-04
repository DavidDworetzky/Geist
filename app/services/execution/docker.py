"""Hardened one-shot Docker sandbox for agent commands.

Each run is a fresh ``docker run --rm`` with the hardening posture ported
from Hermes-agent's Docker environment: all capabilities dropped, privilege
escalation blocked, a non-root user, size-limited tmpfs for every writable
path, no network by default, and PID/memory/CPU ceilings. The container
receives an empty environment, so host credentials cannot leak by
construction.

Approval-free execution only applies while nothing is bind-mounted and the
network is disabled. A host workspace or network access makes every terminal
command require a fresh approval (the Hermes "external access" rule).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404 - argv-only calls to the configured container runtime
import time

from app.services.execution.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ExecutionEnvironment,
    ExecutionResult,
    clamp_timeout,
    truncate_output,
)
from app.services.execution.hardline import detect_hardline_command


DEFAULT_IMAGE = "python:3.11-slim"

# Grace period added to the subprocess timeout so docker's own startup and
# teardown never masquerade as a command timeout.
_DOCKER_OVERHEAD_SECONDS = 20

# nobody:nogroup — the container never runs as root, so no capability
# add-backs are needed at all (stricter than images that drop privileges
# after starting as root).
_SANDBOX_USER = "65534:65534"

_BASE_SECURITY_ARGS = [
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--user",
    _SANDBOX_USER,
    "--pids-limit",
    "256",
    "--memory",
    "512m",
    "--cpus",
    "1",
    "--tmpfs",
    "/tmp:rw,nosuid,size=256m",  # nosec B108 - path exists only inside the container
    "--env",
    "HOME=/tmp",
]


logger = logging.getLogger(__name__)


def find_container_runtime(preferred: str | None = None) -> str | None:
    """Locate a container runtime CLI.

    ``preferred`` pins a specific runtime — a name resolved on PATH
    (``podman``) or an absolute binary path — mirroring Hermes-agent's
    ``HERMES_DOCKER_BINARY`` override. When the pinned runtime cannot be
    found this fails closed (no silent fallback to a runtime the user
    explicitly chose against); without a preference, docker then podman
    are probed on PATH (podman is drop-in compatible for these args).
    """
    if preferred:
        found = shutil.which(preferred)
        if found:
            return found
        if os.path.isfile(preferred) and os.access(preferred, os.X_OK):
            return preferred
        logger.warning(
            "Pinned container runtime %r not found or not executable; "
            "sandbox execution unavailable",
            preferred,
        )
        return None
    for executable in ("docker", "podman"):
        found = shutil.which(executable)
        if found:
            return found
    return None


def build_docker_run_args(
    *,
    image: str,
    command: str,
    network: bool = False,
    workspace: str | None = None,
) -> list[str]:
    """Assemble the argv (after the runtime executable) for one sandboxed run.

    Kept as a pure function so the hardening posture is unit-testable without
    a container runtime present.
    """
    args = ["run", "--rm", *_BASE_SECURITY_ARGS]
    if not network:
        args += ["--network", "none"]
    if workspace:
        args += ["--volume", f"{workspace}:/workspace"]
    else:
        # mode=0777 lets the non-root sandbox user write; the tmpfs is
        # per-run and size-bounded so this grants nothing on the host.
        args += ["--tmpfs", "/workspace:rw,nosuid,size=256m,mode=0777"]
    args += ["--workdir", "/workspace", image, "bash", "-c", command]
    return args


class DockerExecutionEnvironment(ExecutionEnvironment):
    name = "docker"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        network: bool = False,
        workspace: str | None = None,
        runtime_path: str | None = None,
        runtime_preference: str | None = None,
    ):
        self.image = image
        self.network = network
        self.workspace = workspace
        self._runtime_path = runtime_path
        self.runtime_preference = runtime_preference

    @property
    def has_host_access(self) -> bool:
        return bool(self.workspace)

    @property
    def is_sandboxed(self) -> bool:
        return not self.has_host_access

    @property
    def requires_per_call_approval(self) -> bool:
        return self.has_host_access or self.network

    def command_rejection_reason(self, command: str) -> str | None:
        if not self.has_host_access:
            return None
        return detect_hardline_command(command)

    def runtime(self) -> str | None:
        if self._runtime_path is None:
            self._runtime_path = find_container_runtime(self.runtime_preference)
        return self._runtime_path

    def is_available(self) -> bool:
        return self.runtime() is not None

    def run(
        self,
        command: str,
        timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> ExecutionResult:
        hardline = self.command_rejection_reason(command)
        if hardline is not None:
            return ExecutionResult(
                exit_code=126,
                stdout="",
                stderr=f"BLOCKED: refusing unrecoverable command ({hardline})",
                duration_seconds=0.0,
                timed_out=False,
            )
        runtime = self.runtime()
        if runtime is None:
            return ExecutionResult(
                exit_code=127,
                stdout="",
                stderr="No container runtime (docker/podman) is available on this host",
                duration_seconds=0.0,
            )

        timeout = clamp_timeout(timeout_seconds)
        # ``timeout`` inside the container bounds the command itself; the
        # outer subprocess timeout only guards a wedged runtime.
        bounded_command = f"timeout {timeout} bash -c {_shell_quote(command)}"
        args = build_docker_run_args(
            image=self.image,
            command=bounded_command,
            network=self.network,
            workspace=self.workspace,
        )

        started = time.monotonic()
        try:
            completed = subprocess.run(  # nosec B603 - argv invokes the resolved runtime
                [runtime, *args],
                capture_output=True,
                text=True,
                timeout=timeout + _DOCKER_OVERHEAD_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Sandbox did not finish within {timeout} seconds",
                duration_seconds=time.monotonic() - started,
                timed_out=True,
            )

        stdout, stdout_truncated = truncate_output(completed.stdout)
        stderr, stderr_truncated = truncate_output(completed.stderr)
        # GNU timeout reports 124 when the inner command exceeded its bound.
        timed_out = completed.returncode == 124
        return ExecutionResult(
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
        )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
