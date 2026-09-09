"""A shared file/terminal workspace for one execution environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from adapters import workspace_file_adapter
from adapters.workspace_file_adapter import WorkspaceFileAdapter
from agents.models.tool_calling import ToolContext
from app.runtime_config import default_data_dir
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
            raise ValueError(
                "Construct the execution environment with the configured workspace root"
            )
        if isinstance(environment, LocalExecutionEnvironment) and environment.workdir != root:
            raise ValueError(
                "Construct the execution environment with the configured workspace root"
            )
        self.files = WorkspaceFileAdapter(root) if root else None
        self.managed_root = default_data_dir() / "code" if root is None else None
        self.sessions = None
        self.namespace = ""
        if isinstance(environment, DockerExecutionEnvironment):
            environment.managed_workspace = self.managed_root is not None
            self.sessions = create_session_manager(environment, required=True)
            config = json.dumps(
                [
                    environment.image,
                    root,
                    environment.network,
                    str(self.managed_root),
                    "per-chat-v1",
                ]
            )
            self.namespace = hashlib.sha256(config.encode()).hexdigest()[:16]

    @property
    def available(self) -> bool:
        if isinstance(self.environment, DockerExecutionEnvironment):
            return self.environment.is_available()
        return self.files is not None or self.managed_root is not None

    @property
    def requires_per_call_approval(self) -> bool:
        # Both configured and per-chat roots are durable host storage.
        return True

    def directory(self, context: ToolContext) -> Path:
        if self.files is not None:
            return self.files.file_root
        if self.managed_root is None:
            raise WorkspaceOperationError("A configured or managed coding workspace is required")
        if context.workspace_id <= 0 or (context.chat_id is not None and context.chat_id <= 0):
            raise WorkspaceOperationError("A valid owner and chat identity is required")
        identity = (
            f"chat-{context.chat_id}"
            if context.chat_id is not None
            else "run-" + hashlib.sha256(context.run_id.encode()).hexdigest()
        )
        path = self.managed_root
        for component in (None, f"workspace-{context.workspace_id}", identity):
            if component is not None:
                path = path / component
            if path.is_symlink():
                raise WorkspaceOperationError("Managed code directories must not be symlinks")
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not path.is_dir():
                raise WorkspaceOperationError("Managed code path must be a directory")
            if component != identity:
                path.chmod(0o700)
        # Only the leaf is mounted; private ancestors protect it on the host.
        # The sandbox runs as UID 65534, not the host user's UID.
        if self.sessions is not None:
            path.chmod(0o777)
        return path

    def scope(self, context: ToolContext) -> str:
        chat = (
            str(context.chat_id)
            if context.chat_id is not None
            else hashlib.sha256(context.run_id.encode()).hexdigest()
        )
        identity = f"workspace-{context.workspace_id}-chat-{chat}"
        return f"{self.namespace}-{identity}"

    def file_operation(
        self, context: ToolContext, operation: str, arguments: dict[str, Any]
    ) -> Any:
        if operation not in {"list_files", "read_file", "search_text", "write_file", "edit_file"}:
            raise ValueError("Unsupported workspace operation")
        if self.sessions is None:
            files = self.files or WorkspaceFileAdapter(str(self.directory(context)))
            return getattr(files, operation)(**arguments)
        if not self.available:
            raise WorkspaceOperationError(
                "Workspace file tools require an available Docker or Podman runtime."
            )
        # Send the same standard-library adapter over stdin; large edits never
        # become shell arguments, and model input is decoded strictly as JSON.
        source = Path(workspace_file_adapter.__file__).read_text(encoding="utf-8")
        payload = json.dumps({"operation": operation, "arguments": arguments})
        source += "\nimport json\nrequest = json.loads(" + repr(payload) + ")\n"
        source += "print(json.dumps(getattr(WorkspaceFileAdapter('/workspace'), request['operation'])(**request['arguments'])))\n"
        result = self.sessions.run_in_session(
            self.scope(context),
            "python -",
            input_text=source,
            output_limit=1_500_000,
            workspace=str(self.directory(context)),
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
            return self.sessions.run_in_session(
                self.scope(context),
                command,
                timeout_seconds,
                workspace=str(self.directory(context)),
            )
        if (
            isinstance(self.environment, LocalExecutionEnvironment)
            and self.managed_root is not None
        ):
            return LocalExecutionEnvironment(str(self.directory(context))).run(
                command, timeout_seconds
            )
        return self.environment.run(command, timeout_seconds)
