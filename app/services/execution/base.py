"""Abstract execution environment contract shared by all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


DEFAULT_COMMAND_TIMEOUT_SECONDS = 30
MAX_COMMAND_TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 10_000


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one command run in an execution environment."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False


def truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    """Bound one output stream, marking whether anything was dropped."""
    if len(text) <= limit:
        return text, False
    marker = "\n[output truncated]"
    return f"{text[: limit - len(marker)]}{marker}", True


class ExecutionEnvironment(ABC):
    """One place a shell command can run: the host, or an isolated sandbox.

    ``is_sandboxed`` drives the approval posture of the tool built on top of
    this environment; it must be True only when a command cannot reach host
    files or host processes.
    """

    name: str = "base"

    @property
    @abstractmethod
    def is_sandboxed(self) -> bool:
        """True when command effects are contained away from the host."""

    @property
    def requires_per_call_approval(self) -> bool:
        """Whether every command needs approval regardless of standing grants."""
        return not self.is_sandboxed

    def command_rejection_reason(self, command: str) -> str | None:
        """Return an unconditional policy rejection, if one applies."""
        return None

    @abstractmethod
    def run(
        self,
        command: str,
        timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> ExecutionResult:
        """Run one shell command to completion and capture bounded output."""

    def describe(self) -> str:
        """Human-facing one-liner used in tool descriptions and logs."""
        posture = "sandboxed" if self.is_sandboxed else "host (unsandboxed)"
        return f"{self.name} [{posture}]"


def clamp_timeout(timeout_seconds: int | None) -> int:
    if timeout_seconds is None:
        return DEFAULT_COMMAND_TIMEOUT_SECONDS
    return max(1, min(int(timeout_seconds), MAX_COMMAND_TIMEOUT_SECONDS))
