"""A shared file/terminal workspace for one execution environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from adapters import workspace_file_adapter
from adapters.workspace_file_adapter import WorkspaceFileAdapter
from agents.models.tool_calling import ToolContext
from app.services.execution.base import ExecutionEnvironment, ExecutionResult
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.factory import configured_workspace_root, create_session_manager
from app.services.execution.local import LocalExecutionEnvironment


class WorkspaceOperationError(RuntimeError):
    """Safe, actionable workspace error that may be returned to the model."""


class CodingWorkspace:
    def __init__(self, environment: ExecutionEnvironment | None) -> None:
        self.environment = environment
        root = configured_workspace_root()
        if isinstance(environment, DockerExecutionEnvironment) and environment.workspace != root:
            raise ValueError("Construct the execution environment with the configured workspace root")
        if isinstance(environment, LocalExecutionEnvironment) and environment.workdir != root:
            raise ValueError("Construct the execution environment with the configured workspace root")
        self.files = WorkspaceFileAdapter(root) if root else None
        self.sessions = None
        self.namespace = ""
        if isinstance(environment, DockerExecutionEnvironment):
            self.sessions = create_session_manager(environment, required=True)
            config = json.dumps(
                [environment.image, environment.workspace, environment.network, "readonly-v1"]
            )
            self.namespace = hashlib.sha256(config.encode()).hexdigest()[:16]

    @property
    def available(self) -> bool:
        return self.sessions is not None or self.files is not None

    @property
    def requires_per_call_approval(self) -> bool:
        return self.sessions is None or bool(
            isinstance(self.environment, DockerExecutionEnvironment)
            and self.environment.has_host_access
        )

    def scope(self, context: ToolContext) -> str:
        identity = (
            context.coding_workspace_id
            or f"workspace-{context.workspace_id}-chat-{context.chat_id or context.run_id}"
        )
        return f"{self.namespace}-{identity}"

    def file_operation(
        self, context: ToolContext, operation: str, arguments: dict[str, Any]
    ) -> Any:
        if operation not in {"list_files", "read_file", "search_text", "write_file", "edit_file"}:
            raise ValueError("Unsupported workspace operation")
        if self.sessions is None:
            if self.files is None:
                raise WorkspaceOperationError(
                    "Select an explicit GEIST_WORKSPACE_ROOT to use host file tools."
                )
            return getattr(self.files, operation)(**arguments)
        # Send the same standard-library adapter over stdin; large edits never
        # become shell arguments, and model input is decoded strictly as JSON.
        source = Path(workspace_file_adapter.__file__).read_text(encoding="utf-8")
        payload = json.dumps({"operation": operation, "arguments": arguments})
        source += "\nimport json\nrequest = json.loads(" + repr(payload) + ")\n"
        source += "print(json.dumps(getattr(WorkspaceFileAdapter('/workspace'), request['operation'])(**request['arguments'])))\n"
        result = self.sessions.run_in_session(
            self.scope(context), "python -", input_text=source, output_limit=1_500_000
        )
        if result.exit_code == 127:
            raise WorkspaceOperationError(
                "Workspace file tools require Python in GEIST_EXEC_DOCKER_IMAGE; use a Python-capable image."
            )
        if result.exit_code != 0 or result.truncated:
            raise WorkspaceOperationError(
                "Workspace operation failed. Check the sandbox configuration, file path and output size."
            )
        return json.loads(result.stdout)

    def run(self, context: ToolContext, command: str, timeout_seconds: int) -> ExecutionResult:
        if self.environment is None:
            raise RuntimeError("Terminal execution is not configured")
        if self.sessions:
            return self.sessions.run_in_session(self.scope(context), command, timeout_seconds)
        return self.environment.run(command, timeout_seconds)
