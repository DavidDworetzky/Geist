"""Block agent reads of local .env files while allowing committed templates."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import PurePosixPath


ALLOWED_ENV_SUFFIXES = (
    ".example",
    ".sample",
    ".template",
    ".templates",
    ".dist",
)


def _load_input() -> dict[str, object]:
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}


def _is_env_template(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return any(name.endswith(suffix) for suffix in ALLOWED_ENV_SUFFIXES)


def _is_local_env_path(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    if _is_env_template(path):
        return False
    return name in {".env", ".envrc"} or name.startswith(".env.")


def _candidate_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        candidates: list[str] = []
        for item in value:
            candidates.extend(_candidate_strings(item))
        return candidates
    if isinstance(value, dict):
        candidates = []
        for item in value.values():
            candidates.extend(_candidate_strings(item))
        return candidates
    return []


def _command_mentions_local_env(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    for token in tokens:
        if _is_local_env_path(token):
            return True

    # Catch simple command strings that contain paths in quotes or shell globs.
    for match in re.finditer(r"(^|[/\s'\"])(\.env(?:\.[^\s'\"/]+)?)", command):
        env_name = match.group(2)
        if not _is_env_template(env_name):
            return True

    return False


def _payload_mentions_local_env(payload: dict[str, object]) -> bool:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return False

    command = tool_input.get("command")
    if isinstance(command, str) and _command_mentions_local_env(command):
        return True

    return any(_is_local_env_path(candidate) for candidate in _candidate_strings(tool_input))


def main() -> int:
    if _payload_mentions_local_env(_load_input()):
        print(
            json.dumps(
                {
                    "decision": "deny",
                    "reason": (
                        "Local .env files are off limits for agents. "
                        "Use committed templates such as .env.example instead."
                    ),
                }
            ),
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
