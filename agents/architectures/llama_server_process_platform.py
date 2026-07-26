"""Platform boundary for llama-server child creation and termination."""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import subprocess
from typing import Any, cast

import psutil


logger = logging.getLogger(__name__)
_platform_process: Any = importlib.import_module(
    "agents.architectures.llama_server_process_windows"
    if os.name == "nt"
    else "agents.architectures.llama_server_process_posix"
)


def process_options() -> dict[str, Any]:
    return cast(dict[str, Any], _platform_process.process_options())


def attach_process_lifetime(process: subprocess.Popen[str]) -> Any | None:
    return _platform_process.attach_process_lifetime(process)


def close_process_lifetime(handle: Any | None) -> None:
    _platform_process.close_process_lifetime(handle)


def terminate_process_tree(process: subprocess.Popen[str], handle: Any | None) -> None:
    """Terminate a retained child and every descendant, then release its handle."""

    if process.poll() is not None:
        close_process_lifetime(handle)
        return
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    for child in descendants:
        with contextlib.suppress(psutil.Error):
            child.terminate()
    try:
        process.terminate()
        process.wait(timeout=3.0)
    except (OSError, subprocess.TimeoutExpired):
        for child in descendants:
            with contextlib.suppress(psutil.Error):
                child.kill()
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("Unable to confirm llama-server process termination")
    _gone, alive = psutil.wait_procs(descendants, timeout=0.5)
    for child in alive:
        with contextlib.suppress(psutil.Error):
            child.kill()
    close_process_lifetime(handle)
