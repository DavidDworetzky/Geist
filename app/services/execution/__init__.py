"""Native tool execution backends.

Pluggable environments for running agent-issued shell commands, modeled on
Hermes-agent's ``tools/environments`` layer: one abstract interface, a
hardened Docker sandbox, and an unsandboxed local fallback. Backend selection
is config-driven via ``GEIST_EXEC_BACKEND`` (unset = execution disabled).

The sandbox/approval coupling follows the Hermes rule: an isolated backend
runs without per-call approval, while a backend that can reach host files
(local, or Docker with a bind-mounted workspace) registers its tool with
``requires_approval=True`` so the user permission gate applies.
"""

from app.services.execution.base import ExecutionEnvironment, ExecutionResult
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.factory import create_execution_environment
from app.services.execution.local import LocalExecutionEnvironment


__all__ = [
    "DockerExecutionEnvironment",
    "ExecutionEnvironment",
    "ExecutionResult",
    "LocalExecutionEnvironment",
    "create_execution_environment",
]
