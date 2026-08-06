"""Unsandboxed host execution with mitigations.

Runs commands directly on the host via bash. Not an isolation boundary —
the tool built on this backend must register with ``requires_approval=True``.
Mitigations mirror Hermes-agent's local mode: secrets are scrubbed from the
child environment so agent commands never see provider credentials, the
hardline blocklist refuses unrecoverable commands outright, and output and
runtime are bounded.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

from app.services.execution.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ExecutionEnvironment,
    ExecutionResult,
    clamp_timeout,
    truncate_output,
)
from app.services.execution.hardline import detect_hardline_command


# Environment variables whose names match any of these fragments are withheld
# from child processes. Deny-by-name is imperfect, but it removes the obvious
# credential surface (provider keys, tokens, connection strings).
_SECRET_NAME_FRAGMENTS = (
    "API_KEY",
    "APIKEY",
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
        hardline = detect_hardline_command(command)
        if hardline is not None:
            return ExecutionResult(
                exit_code=126,
                stdout="",
                stderr=f"BLOCKED: refusing unrecoverable command ({hardline})",
                duration_seconds=0.0,
                timed_out=False,
            )

        timeout = clamp_timeout(timeout_seconds)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workdir,
                env=scrub_environment(dict(os.environ)),
            )
        except subprocess.TimeoutExpired as expired:
            stdout = expired.stdout or ""
            stderr = expired.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            stdout, _ = truncate_output(stdout)
            stderr, _ = truncate_output(stderr)
            return ExecutionResult(
                exit_code=124,
                stdout=stdout,
                stderr=stderr or f"Command timed out after {timeout} seconds",
                duration_seconds=time.monotonic() - started,
                timed_out=True,
            )

        stdout, stdout_truncated = truncate_output(completed.stdout)
        stderr, stderr_truncated = truncate_output(completed.stderr)
        return ExecutionResult(
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            truncated=stdout_truncated or stderr_truncated,
        )
