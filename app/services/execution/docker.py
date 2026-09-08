"""Hardened one-shot Docker sandbox for agent commands.

Each run is a fresh ``docker run --rm`` with the hardening posture ported
from Hermes-agent's Docker environment: all capabilities dropped, privilege
escalation blocked, a non-root user, size-limited tmpfs for every writable
path, no network by default, and PID/memory/CPU ceilings. The container
does not inherit host environment variables. Secrets inside an explicitly
mounted workspace remain readable; this is not a secret-filtering boundary.

The sandbox claim only holds while nothing is bind-mounted: configuring a
host ``workspace`` flips ``is_sandboxed`` to False. Either a workspace or network
access requires fresh approval regardless of permission mode or standing grants.
Custom images must provide bash and GNU timeout.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404 - argv-only calls to the configured runtime
import time
import uuid
from dataclasses import replace

from app.services.execution.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ExecutionEnvironment,
    ExecutionResult,
    clamp_timeout,
)
from app.services.execution.capture import capture_process


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
    "/tmp:rw,nosuid,size=256m",  # nosec B108 - isolated container mount, not host temp files
]


logger = logging.getLogger(__name__)


def workspace_mount_args(workspace: str) -> list[str]:
    if not os.path.isabs(workspace) or "," in workspace:
        raise ValueError("Docker workspace must be absolute and cannot contain commas")
    return ["--mount", f"type=bind,source={workspace},target=/workspace"]


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
    container_name: str | None = None,
) -> list[str]:
    """Assemble the argv (after the runtime executable) for one sandboxed run.

    Kept as a pure function so the hardening posture is unit-testable without
    a container runtime present.
    """
    args = ["run", "--rm", *_BASE_SECURITY_ARGS]
    if container_name:
        args += ["--name", container_name]
    if not network:
        args += ["--network", "none"]
    if workspace:
        args += workspace_mount_args(workspace)
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
        self.workspace = os.path.abspath(workspace) if workspace else None
        if self.workspace:
            workspace_mount_args(self.workspace)
            if not os.path.isdir(self.workspace):
                raise ValueError("Docker workspace must be an existing directory")
        self._runtime_path = runtime_path
        self.runtime_preference = runtime_preference
        if os.path.basename(runtime_path or runtime_preference or "") == "podman":
            self.name = "podman"

    @property
    def has_host_access(self) -> bool:
        return bool(self.workspace)

    @property
    def is_sandboxed(self) -> bool:
        return not self.has_host_access

    @property
    def requires_per_call_approval(self) -> bool:
        return self.has_host_access or self.network

    def describe(self) -> str:
        return super().describe() + ("; network enabled" if self.network else "")

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
            return ExecutionResult(126, "", f"BLOCKED: {hardline}", 0.0, blocked=True)
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
        bounded_command = f"timeout --kill-after=1 {timeout} bash -c {_shell_quote(command)}"
        container_name = f"geist-exec-{uuid.uuid4().hex}"
        args = build_docker_run_args(
            image=self.image,
            command=bounded_command,
            network=self.network,
            workspace=self.workspace,
            container_name=container_name,
        )

        started = time.monotonic()
        try:
            process = subprocess.Popen(  # nosec B603 - resolved runtime and argv
                [runtime, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as error:
            return ExecutionResult(127, "", str(error), time.monotonic() - started)
        result = capture_process(process, timeout + _DOCKER_OVERHEAD_SECONDS, process.kill)
        if result.timed_out:
            try:
                subprocess.run(  # nosec B603 - only this invocation's random container
                    [runtime, "rm", "--force", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                logger.warning(
                    "Could not confirm cleanup of execution container %s", container_name
                )
        # GNU timeout uses124/137; an explicit command exit with either is ambiguous.
        return replace(
            result,
            duration_seconds=time.monotonic() - started,
            timed_out=result.timed_out or result.exit_code in (124, 137),
        )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
