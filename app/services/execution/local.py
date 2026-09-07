"""Unsandboxed host execution with mitigations.

Runs commands directly on the host via bash. Not an isolation boundary —
the tool built on this backend must register with ``requires_approval=True``.
Mitigations mirror Hermes-agent's local mode: secrets are scrubbed from the
child environment to reduce accidental exposure. Host files remain readable,
and regex hardline checks are best-effort, not an isolation boundary.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess  # nosec B404 - explicitly enabled host runner
import sys
import time
from contextlib import suppress

from app.services.execution.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ExecutionEnvironment,
    ExecutionResult,
    clamp_timeout,
)
from app.services.execution.capture import capture_process


# Environment variables whose names match any of these fragments are withheld
# from child processes. Deny-by-name is imperfect, but it removes the obvious
# credential surface (provider keys, tokens, connection strings).
_SECRET_NAME_FRAGMENTS = (
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "_PAT",
    "SSH_AUTH_SOCK",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "DATABASE_URL",
    "CONNECTION_STRING",
)

_SECRET_NAME_RE = re.compile(
    "|".join(re.escape(fragment) for fragment in _SECRET_NAME_FRAGMENTS),
    re.IGNORECASE,
)


def scrub_environment(env: dict[str, str]) -> dict[str, str]:
    """Drop credential-shaped variables from a child process environment."""
    return {key: value for key, value in env.items() if not _SECRET_NAME_RE.search(key)}


class LocalExecutionEnvironment(ExecutionEnvironment):
    name = "local"

    def __init__(self, workdir: str | None = None):
        self.workdir = os.path.abspath(workdir) if workdir else None

    @property
    def is_sandboxed(self) -> bool:
        return False

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
                blocked=True,
            )

        if sys.platform == "win32":
            return ExecutionResult(127, "", "Local execution requires POSIX process groups", 0.0)
        timeout = clamp_timeout(timeout_seconds)
        started = time.monotonic()
        try:
            process = subprocess.Popen(  # nosec B603 B607 - intentional approved bash runner
                ["bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
                cwd=self.workdir,
                env=scrub_environment(dict(os.environ)),
            )
        except OSError as error:
            return ExecutionResult(127, "", str(error), time.monotonic() - started)

        def terminate() -> None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)

        return capture_process(process, timeout, terminate)
