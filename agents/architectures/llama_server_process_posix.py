"""POSIX process primitives for the managed llama-server child."""

from __future__ import annotations

import subprocess
from typing import Any


def process_options() -> dict[str, Any]:
    """Start llama-server in its own session so descendants are isolated."""

    return {"start_new_session": True}


def attach_process_lifetime(_process: subprocess.Popen[str]) -> None:
    """POSIX process groups do not require a retained native handle."""

    return None


def close_process_lifetime(_handle: Any | None) -> None:
    """There is no platform handle to close on POSIX."""
