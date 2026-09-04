"""Persistent per-chat Docker sandbox sessions.

The one-shot sandbox gives every ``terminal.run`` call a fresh container, so
multi-step workflows (install a package, then use it) cannot work. When
``GEIST_EXEC_PERSISTENT`` is enabled, each chat session gets a named
long-lived container (``sleep infinity``) with the same hardening posture as
the one-shot runner, and commands run via ``docker exec``. Filesystem state
in /workspace and /tmp survives between calls for the container's lifetime.

Idle containers are reaped opportunistically: every ``run_in_session`` call
also removes sessions that have been unused longer than the TTL, so no
background thread is needed (the same write-time pruning pattern used for
agent snapshots).
"""

from __future__ import annotations

import logging
import re
import subprocess  # nosec B404 - argv-only calls to the configured container runtime
import threading
import time

from app.services.execution.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ExecutionResult,
    clamp_timeout,
    truncate_output,
)
from app.services.execution.docker import (
    _BASE_SECURITY_ARGS,
    _DOCKER_OVERHEAD_SECONDS,
    DockerExecutionEnvironment,
    _shell_quote,
)


logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_SECONDS = 1800.0
_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]")


def session_container_name(scope_key: str) -> str:
    """Deterministic, docker-safe container name for one session scope."""
    return f"geist-exec-{_NAME_SAFE_RE.sub('-', str(scope_key))}"


def build_session_create_args(
    *,
    name: str,
    image: str,
    network: bool = False,
    workspace: str | None = None,
) -> list[str]:
    """argv (after the runtime executable) that starts one session container."""
    args = ["run", "--detach", "--name", name, *_BASE_SECURITY_ARGS]
    if not network:
        args += ["--network", "none"]
    if workspace:
        args += ["--volume", f"{workspace}:/workspace"]
    else:
        args += ["--tmpfs", "/workspace:rw,nosuid,size=256m,mode=0777"]
    args += ["--workdir", "/workspace", image, "sleep", "infinity"]
    return args


def build_session_exec_args(
    *, name: str, command: str, timeout: int, stdin: bool = False
) -> list[str]:
    """argv (after the runtime executable) that runs one command in a session."""
    bounded = f"timeout {timeout} bash -c {_shell_quote(command)}"
    return [
        "exec",
        *(["--interactive"] if stdin else []),
        "--workdir",
        "/workspace",
        name,
        "bash",
        "-c",
        bounded,
    ]


class DockerSessionManager:
    """Lazily creates, reuses, and reaps per-scope sandbox containers."""

    def __init__(
        self,
        environment: DockerExecutionEnvironment,
        *,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        clock=time.monotonic,
    ):
        self.environment = environment
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._last_used: dict[str, float] = {}

    def run_in_session(
        self,
        scope_key: str,
        command: str,
        timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        *,
        input_text: str | None = None,
        output_limit: int = 10_000,
    ) -> ExecutionResult:
        hardline = self.environment.command_rejection_reason(command)
        if hardline is not None:
            return ExecutionResult(
                exit_code=126,
                stdout="",
                stderr=f"BLOCKED: refusing unrecoverable command ({hardline})",
                duration_seconds=0.0,
                timed_out=False,
            )
        runtime = self.environment.runtime()
        if runtime is None:
            return ExecutionResult(
                exit_code=127,
                stdout="",
                stderr="No container runtime (docker/podman) is available on this host",
                duration_seconds=0.0,
            )

        self._reap_idle(runtime, exclude=scope_key)

        name = session_container_name(scope_key)
        timeout = clamp_timeout(timeout_seconds)
        started = self._clock()

        create_error = self._ensure_container(runtime, name)
        if create_error is not None:
            return ExecutionResult(
                exit_code=125,
                stdout="",
                stderr=f"Could not start sandbox session: {create_error}",
                duration_seconds=self._clock() - started,
            )

        try:
            completed = subprocess.run(  # nosec B603 - argv invokes the resolved runtime
                [
                    runtime,
                    *build_session_exec_args(
                        name=name, command=command, timeout=timeout, stdin=input_text is not None
                    ),
                ],
                capture_output=True,
                text=True,
                input=input_text,
                timeout=timeout + _DOCKER_OVERHEAD_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Sandbox session did not respond within {timeout} seconds",
                duration_seconds=self._clock() - started,
                timed_out=True,
            )
        finally:
            with self._lock:
                self._last_used[scope_key] = self._clock()

        stdout, stdout_truncated = truncate_output(
            completed.stdout, limit=min(output_limit, 1_500_000)
        )
        stderr, stderr_truncated = truncate_output(completed.stderr)
        return ExecutionResult(
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=self._clock() - started,
            timed_out=completed.returncode == 124,
            truncated=stdout_truncated or stderr_truncated,
        )

    def _ensure_container(self, runtime: str, name: str) -> str | None:
        """Start the session container if it isn't already running."""
        inspect = subprocess.run(  # nosec B603 - argv invokes the resolved runtime
            [runtime, "inspect", "--format", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0 and inspect.stdout.strip() == "true":
            return None
        if inspect.returncode == 0:
            # Exists but stopped (host restart, OOM-kill): replace it.
            subprocess.run(  # nosec B603 - argv invokes the resolved runtime
                [runtime, "rm", "-f", name], capture_output=True, text=True
            )

        created = subprocess.run(  # nosec B603 - argv invokes the resolved runtime
            [
                runtime,
                *build_session_create_args(
                    name=name,
                    image=self.environment.image,
                    network=self.environment.network,
                    workspace=self.environment.workspace,
                ),
            ],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            return created.stderr.strip() or f"exit {created.returncode}"
        return None

    def _reap_idle(self, runtime: str, exclude: str | None = None) -> None:
        now = self._clock()
        with self._lock:
            expired = [
                scope
                for scope, last_used in self._last_used.items()
                if scope != exclude and now - last_used > self.ttl_seconds
            ]
            for scope in expired:
                self._last_used.pop(scope, None)
        for scope in expired:
            self._remove(runtime, scope)

    def _remove(self, runtime: str, scope_key: str) -> None:
        name = session_container_name(scope_key)
        try:
            subprocess.run(  # nosec B603 - argv invokes the resolved runtime
                [runtime, "rm", "-f", name], capture_output=True, text=True
            )
            logger.info("Reaped idle sandbox session %s", name)
        except Exception:
            logger.warning("Could not remove sandbox session %s", name, exc_info=True)

    def shutdown(self) -> None:
        """Remove every tracked session container (app shutdown)."""
        runtime = self.environment.runtime()
        with self._lock:
            scopes = list(self._last_used)
            self._last_used.clear()
        if runtime is None:
            return
        for scope in scopes:
            self._remove(runtime, scope)
