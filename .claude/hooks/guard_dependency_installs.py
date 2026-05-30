"""Block dependency installation commands until the user explicitly approves."""

from __future__ import annotations

import json
import re
import shlex
import sys


INSTALL_PATTERNS = (
    re.compile(r"(^|\s)python\d*(\.\d+)?\s+-m\s+pip\s+install(\s|$)"),
    re.compile(r"(^|\s)pip\d*(\.\d+)?\s+install(\s|$)"),
    re.compile(r"(^|\s)conda\s+(install|env\s+update|env\s+create)(\s|$)"),
    re.compile(r"(^|\s)mamba\s+(install|env\s+update|env\s+create)(\s|$)"),
    re.compile(r"(^|\s)poetry\s+add(\s|$)"),
    re.compile(r"(^|\s)uv\s+(add|pip\s+install|sync)(\s|$)"),
    re.compile(r"(^|\s)npm\s+(i|install|add|ci)(\s|$)"),
    re.compile(r"(^|\s)yarn\s+(add|install)(\s|$)"),
    re.compile(r"(^|\s)pnpm\s+(add|install|i)(\s|$)"),
    re.compile(r"(^|\s)brew\s+install(\s|$)"),
    re.compile(r"(^|\s)apt(-get)?\s+install(\s|$)"),
)


def _load_input() -> dict[str, object]:
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}


def _command_from_hook_input(payload: dict[str, object]) -> str:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return ""

    command = tool_input.get("command")
    return command if isinstance(command, str) else ""


def _strip_wrapping_shell(command: str) -> str:
    """Return the command body for simple `sh -c`/`bash -lc` wrappers."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command

    if len(parts) >= 3 and parts[0] in {"bash", "sh", "zsh"} and parts[1] in {"-c", "-lc"}:
        return parts[2]
    return command


def main() -> int:
    command = _strip_wrapping_shell(_command_from_hook_input(_load_input()))
    normalized = re.sub(r"\s+", " ", command.strip())

    if any(pattern.search(normalized) for pattern in INSTALL_PATTERNS):
        print(
            json.dumps(
                {
                    "decision": "deny",
                    "reason": (
                        "Dependency installation requires explicit user approval first. "
                        "Ask before running package-manager install commands."
                    ),
                }
            ),
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
