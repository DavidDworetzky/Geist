"""Config-driven selection of the tool execution backend.

Environment variables (all optional; unset backend disables execution):

- ``GEIST_EXEC_BACKEND``: ``docker`` | ``podman`` | ``local``. Anything else
  (or unset) means no execution backend, and the terminal tool is not
  registered. ``podman`` is the docker backend pinned to the podman CLI —
  same sandbox hardening, daemonless runtime.
- ``GEIST_EXEC_RUNTIME``: pin the container runtime CLI (a PATH name like
  ``podman`` or an absolute binary path), mirroring Hermes-agent's
  ``HERMES_DOCKER_BINARY``. When the pinned runtime is missing, sandbox
  execution is unavailable rather than silently using another runtime.
- ``GEIST_EXEC_DOCKER_IMAGE``: sandbox image (default ``python:3.11-slim``).
- ``GEIST_EXEC_DOCKER_NETWORK``: ``1``/``true`` to give the sandbox network
  access (default: no network). Networked commands require per-call approval.
- ``GEIST_EXEC_WORKSPACE``: host directory. For the docker backend this is
  bind-mounted at /workspace and makes the environment host-reaching (the
  tool then requires per-call approval); for the local backend it is the working
  directory.
- ``GEIST_EXEC_PERSISTENT``: ``1``/``true`` to keep one long-lived sandbox
  container per chat session (docker backend only), so filesystem state
  survives between terminal.run calls.
- ``GEIST_EXEC_SESSION_TTL_SECONDS``: idle lifetime for persistent session
  containers (default 1800).
"""

from __future__ import annotations

import logging
import os

from app.services.execution.base import ExecutionEnvironment
from app.services.execution.docker import DEFAULT_IMAGE, DockerExecutionEnvironment
from app.services.execution.local import LocalExecutionEnvironment


logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def create_execution_environment() -> ExecutionEnvironment | None:
    """Build the configured execution backend, or None when disabled."""
    backend = os.getenv("GEIST_EXEC_BACKEND", "").strip().lower()
    workspace = os.getenv("GEIST_EXEC_WORKSPACE", "").strip() or None

    if backend in ("docker", "podman"):
        runtime_preference = os.getenv("GEIST_EXEC_RUNTIME", "").strip() or None
        if backend == "podman" and runtime_preference is None:
            runtime_preference = "podman"
        return DockerExecutionEnvironment(
            image=os.getenv("GEIST_EXEC_DOCKER_IMAGE", "").strip() or DEFAULT_IMAGE,
            network=_env_flag("GEIST_EXEC_DOCKER_NETWORK"),
            workspace=workspace,
            runtime_preference=runtime_preference,
        )
    if backend == "local":
        return LocalExecutionEnvironment(workdir=workspace)
    if backend:
        logger.warning(
            "Unknown GEIST_EXEC_BACKEND %r; tool execution disabled "
            "(valid: docker, podman, local)",
            backend,
        )
    return None


def create_session_manager(environment: ExecutionEnvironment | None):
    """Build the persistent-session manager when configured (docker only)."""
    if not _env_flag("GEIST_EXEC_PERSISTENT"):
        return None
    if not isinstance(environment, DockerExecutionEnvironment):
        if environment is not None:
            logger.warning(
                "GEIST_EXEC_PERSISTENT requires the docker backend; " "persistent sessions disabled"
            )
        return None
    from app.services.execution.session import (
        DEFAULT_SESSION_TTL_SECONDS,
        DockerSessionManager,
    )

    try:
        ttl = float(os.getenv("GEIST_EXEC_SESSION_TTL_SECONDS", ""))
    except ValueError:
        ttl = DEFAULT_SESSION_TTL_SECONDS
    return DockerSessionManager(environment, ttl_seconds=ttl)
