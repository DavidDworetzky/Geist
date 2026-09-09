"""Native tool execution backends.

Pluggable environments for running agent-issued shell commands, modeled on
Hermes-agent's ``tools/environments`` layer: one abstract interface, a
hardened Docker sandbox, and an unsandboxed local fallback. Backend selection
is config-driven via ``GEIST_EXEC_BACKEND`` (unset = execution disabled).

Mount-free, network-disabled Docker can run without per-call approval. Host or
network access requires fresh approval that standing grants cannot waive.
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
