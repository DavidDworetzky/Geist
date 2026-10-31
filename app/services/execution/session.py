"""Persistent per-chat Docker sandbox sessions.

The one-shot sandbox gives every ``terminal.run`` call a fresh container, so
multi-step workflows (install a package, then use it) cannot work. When
``GEIST_EXEC_PERSISTENT`` is enabled, each chat session gets a named
long-lived container with the same hardening posture as
the one-shot runner, and commands run via ``docker exec``. Filesystem state
in /workspace and /tmp survives between calls for the container's lifetime.

Idle containers are reaped opportunistically: every ``run_in_session`` call
also removes sessions that have been unused longer than the TTL, so no
background thread is needed (the same write-time pruning pattern used for
agent snapshots).

Each backend instance owns only its randomly named, labeled containers. They
self-expire after 24 hours with --rm, bounding crash leftovers without sweeping
another backend's sessions. Persistence is temporary, not durable file storage.
"""

from __future__ import annotations

import hashlib
import logging
import math
import subprocess  # nosec B404 - bounded argv-only calls to the configured runtime
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field, replace

from app.services.execution.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ExecutionResult,
    clamp_timeout,
)
from app.services.execution.capture import capture_process
from app.services.execution.docker import (
    _BASE_SECURITY_ARGS,
    _DOCKER_OVERHEAD_SECONDS,
    DockerExecutionEnvironment,
    _shell_quote,
    workspace_mount_args,
)
from app.services.execution.hardline import detect_hardline_command


logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_SECONDS = 1800.0
OWNER_LABEL = "org.geist.session-owner"


def session_container_name(scope_key: str, owner: str = "") -> str:
    """Hash the full scope to avoid sanitized-name collisions."""
    digest = hashlib.sha256(scope_key.encode()).hexdigest()[:24]
    return f"geist-exec-{owner}-{digest}"


def build_session_create_args(
    *,
    name: str,
    image: str,
    network: bool = False,
    workspace: str | None = None,
    owner: str | None = None,
) -> list[str]:
    """argv (after the runtime executable) that starts one session container."""
    args = ["run", "--detach", "--name", name, "--rm", *_BASE_SECURITY_ARGS]
    if owner is not None:
        args += ["--label", f"{OWNER_LABEL}={owner}"]
    if not network:
        args += ["--network", "none"]
    if workspace:
        args += workspace_mount_args(workspace)
    else:
        args += ["--tmpfs", "/workspace:rw,nosuid,size=256m,mode=0777"]
    args += ["--workdir", "/workspace", image, "sleep", "86400"]
    return args


def build_session_exec_args(
    *, name: str, command: str, timeout: int, stdin: bool = False
) -> list[str]:
    """argv (after the runtime executable) that runs one command in a session."""
    bounded = (
        "export HOME=/tmp XDG_CACHE_HOME=/tmp; "
        f"timeout --kill-after=1 {timeout} bash -c {_shell_quote(command)}"
    )
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


@dataclass
class _Session:
    name: str
    last_used: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    retired: bool = False


class DockerSessionManager:
    """Lazily creates, reuses, and reaps per-scope sandbox containers."""

    def __init__(
        self,
        environment: DockerExecutionEnvironment,
        *,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        max_sessions: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("Session TTL must be finite and positive")
        if not 1 <= max_sessions <= 64:
            raise ValueError("Session limit must be between 1 and 64")
        self.environment = environment
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.owner = uuid.uuid4().hex
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._closed = False

    @staticmethod
    def _runtime_call(
        runtime: str,
        args: list[str],
        deadline: float,
        *,
        input_text: str | None = None,
        output_limit: int = 10_000,
    ) -> ExecutionResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ExecutionResult(
                124, "", "Container runtime deadline exhausted", 0, timed_out=True
            )
        with ExitStack() as stack:
            stdin = None
            if input_text is not None:
                if len(input_text) > 1_500_000:
                    return ExecutionResult(125, "", "Sandbox input exceeds the size limit", 0)
                # Bounded file input avoids a pipe writer blocking output capture.
                stdin = stack.enter_context(tempfile.TemporaryFile())
                stdin.write(input_text.encode("utf-8"))
                stdin.seek(0)
            try:
                process = subprocess.Popen(  # nosec B603 - resolved runtime, argv-only invocation
                    [runtime, *args],
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except OSError as error:
                return ExecutionResult(127, "", str(error), 0)
            return capture_process(
                process,
                max(0, deadline - time.monotonic()),
                process.kill,
                stdout_limit=output_limit,
            )

    def run_in_session(
        self,
        scope_key: str,
        command: str,
        timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        *,
        input_text: str | None = None,
        output_limit: int = 10_000,
        workspace: str | None = None,
    ) -> ExecutionResult:
        rejection = self.environment.command_rejection_reason(command) or detect_hardline_command(
            command
        )
        if rejection is not None:
            return ExecutionResult(126, "", f"BLOCKED: {rejection}", 0, blocked=True)
        runtime = self.environment.runtime()
        if runtime is None:
            return ExecutionResult(
                exit_code=127,
                stdout="",
                stderr="No container runtime (docker/podman) is available on this host",
                duration_seconds=0.0,
            )

        timeout = clamp_timeout(timeout_seconds)
        started = time.monotonic()
        deadline = started + timeout + _DOCKER_OVERHEAD_SECONDS
        self._reap_idle(runtime, deadline=min(deadline, started + 5), exclude=scope_key)
        with self._lock:
            if self._closed:
                return ExecutionResult(125, "", "Sandbox session manager is shutting down", 0)
            session = self._sessions.get(scope_key)
            if session is None:
                if len(self._sessions) >= self.max_sessions:
                    return ExecutionResult(125, "", "Sandbox session limit reached", 0)
                session = _Session(
                    f"{session_container_name(scope_key, self.owner)}-{uuid.uuid4().hex[:12]}",
                    self._clock(),
                )
                self._sessions[scope_key] = session
        if not session.lock.acquire(timeout=max(0, deadline - time.monotonic())):
            return ExecutionResult(
                124, "", "Sandbox session is busy", time.monotonic() - started, timed_out=True
            )
        try:
            if session.retired:
                return ExecutionResult(
                    125, "", "Sandbox session is closed", time.monotonic() - started
                )
            create_error = self._ensure_container(runtime, session.name, deadline)
            if create_error is not None:
                session.retired = True
                return ExecutionResult(
                    125,
                    "",
                    f"Could not start sandbox session: {create_error}",
                    time.monotonic() - started,
                )
            if session.retired:
                return ExecutionResult(125, "", "Sandbox session was closed during startup", 0)
            exec_args = build_session_exec_args(
                name=session.name,
                command=command,
                timeout=timeout,
                stdin=input_text is not None,
            )
            if input_text is None and output_limit == 10_000:
                result = self._runtime_call(runtime, exec_args, deadline)
            else:
                result = self._runtime_call(
                    runtime, exec_args, deadline, input_text=input_text, output_limit=output_limit
                )
            # A wedged exec can leave descendants running: retire the whole
            # owned sandbox rather than reusing potentially active state.
            if result.timed_out:
                session.retired = True
            return replace(
                result,
                duration_seconds=time.monotonic() - started,
                # GNU timeout exit codes are ambiguous with explicit command exits.
                timed_out=result.timed_out or result.exit_code in (124, 137),
            )
        finally:
            with self._lock:
                session.last_used = self._clock()
            try:
                if session.retired:
                    self._remove(runtime, scope_key, session, time.monotonic() + 5)
            finally:
                session.lock.release()

    def _ensure_container(self, runtime: str, name: str, deadline: float) -> str | None:
        """Start the session container if it isn't already running."""
        inspected = self._inspect(runtime, name, deadline)
        if inspected.timed_out:
            return inspected.stderr or "Container inspection timed out"
        if inspected.exit_code == 0:
            owner, _, running = inspected.stdout.strip().partition(" ")
            if owner != self.owner:
                return "Container ownership mismatch; refusing to reuse or remove it"
            if running == "true":
                return None
            removed = self._runtime_call(runtime, ["rm", "--force", name], deadline)
            if removed.exit_code != 0:
                return removed.stderr or "Could not replace stopped sandbox"
        created = self._runtime_call(
            runtime,
            build_session_create_args(
                name=name,
                image=self.environment.image,
                network=self.environment.network,
                workspace=self.environment.workspace,
                owner=self.owner,
            ),
            deadline,
        )
        if created.exit_code != 0:
            return created.stderr.strip() or f"exit {created.exit_code}"
        return None

    def _inspect(self, runtime: str, name: str, deadline: float) -> ExecutionResult:
        return self._runtime_call(
            runtime,
            [
                "inspect",
                "--format",
                '{{ index .Config.Labels "org.geist.session-owner" }} {{.State.Running}}',
                name,
            ],
            deadline,
        )

    def _reap_idle(self, runtime: str, *, deadline: float, exclude: str | None = None) -> None:
        now = self._clock()
        with self._lock:
            expired = [
                (scope, session)
                for scope, session in self._sessions.items()
                if (scope != exclude or session.retired)
                and (session.retired or now - session.last_used > self.ttl_seconds)
            ]
        for scope, session in expired:
            if time.monotonic() >= deadline:
                break
            if not session.lock.acquire(blocking=False):
                continue
            try:
                with self._lock:
                    if (
                        not session.retired
                        and self._clock() - session.last_used <= self.ttl_seconds
                    ):
                        continue
                    session.retired = True
                self._remove(runtime, scope, session, deadline)
            finally:
                session.lock.release()

    def _remove(self, runtime: str, scope: str, session: _Session, deadline: float) -> None:
        inspected = self._inspect(runtime, session.name, deadline)
        if inspected.exit_code != 0 and not inspected.timed_out:
            # A daemon failure is not evidence that an exited --rm container is absent.
            listed = self._runtime_call(
                runtime,
                [
                    "container",
                    "ls",
                    "--all",
                    "--filter",
                    f"name=^/{session.name}$",
                    "--format",
                    "{{.Names}}",
                ],
                deadline,
            )
            if listed.exit_code == 0 and not listed.stdout.strip():
                with self._lock:
                    if self._sessions.get(scope) is session:
                        self._sessions.pop(scope)
                return
        if inspected.exit_code != 0 or inspected.stdout.split(" ", 1)[0] != self.owner:
            logger.warning("Could not verify owned sandbox cleanup: %s", session.name)
            return
        removed = self._runtime_call(runtime, ["rm", "--force", session.name], deadline)
        if removed.exit_code == 0:
            with self._lock:
                if self._sessions.get(scope) is session:
                    self._sessions.pop(scope)
        else:
            logger.warning("Could not confirm sandbox cleanup: %s", session.name)

    def promote_scope(self, old_scope: str, new_scope: str) -> None:
        with self._lock:
            if old_scope in self._sessions and new_scope not in self._sessions:
                self._sessions[new_scope] = self._sessions.pop(old_scope)

    def close_scope(self, scope_key: str) -> None:
        with self._lock:
            session = self._sessions.get(scope_key)
            if session is None:
                return
            session.retired = True
        runtime = self.environment.runtime()
        if runtime is not None:
            self._reap_idle(runtime, deadline=time.monotonic() + 5)

    def startup(self) -> None:
        with self._lock:
            if any(session.lock.locked() for session in self._sessions.values()):
                raise RuntimeError("Previous sandbox execution is still stopping")
            self._closed = False

    def shutdown(self) -> None:
        """Remove every tracked session container (app shutdown)."""
        with self._lock:
            self._closed = True
            for session in self._sessions.values():
                session.retired = True
        runtime = self.environment.runtime()
        if runtime is not None:
            self._reap_idle(runtime, deadline=time.monotonic() + 5)
