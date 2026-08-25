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
  access (default: no network).
- ``GEIST_EXEC_WORKSPACE``: host directory. For the docker backend this is
  bind-mounted at /workspace and makes the environment host-reaching (the
  tool then requires approval); for the local backend it is the working
  directory.
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
