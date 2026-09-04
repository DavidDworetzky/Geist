"""A shared file/terminal workspace for one execution environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from adapters import workspace_file_adapter
from adapters.workspace_file_adapter import WorkspaceFileAdapter
from agents.models.tool_calling import ToolContext
from app.services.execution.base import ExecutionEnvironment, ExecutionResult
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.factory import create_session_manager
from app.services.execution.local import LocalExecutionEnvironment


class CodingWorkspace:
    def __init__(self, environment: ExecutionEnvironment | None) -> None:
        self.environment = environment
        file_root = os.getenv("GEIST_WORKSPACE_ROOT", "").strip()
        exec_root = os.getenv("GEIST_EXEC_WORKSPACE", "").strip()
        if file_root and exec_root and Path(file_root).resolve() != Path(exec_root).resolve():
            raise ValueError(
                "GEIST_WORKSPACE_ROOT and GEIST_EXEC_WORKSPACE must identify the same workspace"
            )
        root = str(Path(file_root or exec_root or ".").resolve())
        self.files = WorkspaceFileAdapter(root)
        self.sessions = None
        self.namespace = ""
        if isinstance(environment, LocalExecutionEnvironment):
            environment.workdir = root
        elif isinstance(environment, DockerExecutionEnvironment):
            # An explicit file root is also the terminal's bind mount. Unconfigured
            # Docker workspaces remain isolated and shared by all tools in the run.
            environment.workspace = root if file_root or exec_root else None
            self.sessions = create_session_manager(environment, required=True)
            config = json.dumps(
                [environment.image, environment.workspace, environment.network, "readonly-v1"]
            )
            self.namespace = hashlib.sha256(config.encode()).hexdigest()[:16]

    def scope(self, context: ToolContext) -> str:
        identity = (
            context.workspace_id
            or f"user-{context.user_id}-chat-{context.chat_id or context.run_id}"
        )
        return f"{self.namespace}-{identity}"

    def file_operation(
        self, context: ToolContext, operation: str, arguments: dict[str, Any]
    ) -> Any:
        if operation not in {"list_files", "read_file", "search_text", "write_file", "edit_file"}:
            raise ValueError("Unsupported workspace operation")
        if self.sessions is None:
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
        if result.exit_code != 0 or result.truncated:
            raise RuntimeError(
                f"Workspace operation failed: {result.stderr[:1000] or 'response too large'}"
            )
        return json.loads(result.stdout)

    def run(self, context: ToolContext, command: str, timeout_seconds: int) -> ExecutionResult:
        if self.environment is None:
            raise RuntimeError("Terminal execution is not configured")
        if self.sessions:
            return self.sessions.run_in_session(self.scope(context), command, timeout_seconds)
        return self.environment.run(command, timeout_seconds)
