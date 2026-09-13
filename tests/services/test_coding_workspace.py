import os
import uuid
from unittest.mock import Mock

import pytest

from agents.models.tool_calling import ToolCall, ToolContext
from app.services.coding_workspace import CodingWorkspace
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.factory import create_execution_environment
from app.services.execution.local import LocalExecutionEnvironment
from app.services.tool_registry import build_default_tool_registry


def test_file_and_local_terminal_use_one_root(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "local")
    environment = create_execution_environment()
    workspace = CodingWorkspace(environment)
    context = ToolContext(workspace_id=1, chat_id=None, run_id="test")
    workspace.file_operation(context, "write_file", {"path": "feature.txt", "content": "updated"})
    assert environment.workdir == str(tmp_path)
    environment.run = Mock()
    workspace.run(context, "cat feature.txt", 10)
    environment.run.assert_called_once_with("cat feature.txt", 10)
    assert (tmp_path / "feature.txt").read_text() == "updated"


def test_conflicting_roots_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("GEIST_EXEC_WORKSPACE", str(tmp_path / "other"))
    with pytest.raises(ValueError, match="same workspace"):
        CodingWorkspace(LocalExecutionEnvironment())


def test_explicit_file_root_also_controls_docker_mount_and_approval(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    environment = create_execution_environment()
    workspace = CodingWorkspace(environment)
    assert environment.workspace == str(tmp_path)
    assert environment.requires_per_call_approval
    first = ToolContext(workspace_id=1, chat_id=None, run_id="first", coding_workspace_id="goal-workspace")
    resumed = ToolContext(workspace_id=1, chat_id=7, run_id="second", coding_workspace_id="goal-workspace")
    assert workspace.scope(first) == workspace.scope(resumed)


def test_default_host_workspace_is_unavailable(monkeypatch):
    monkeypatch.delenv("GEIST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    monkeypatch.delenv("GEIST_EXEC_BACKEND", raising=False)
    registry = build_default_tool_registry()
    context = ToolContext(workspace_id=1, chat_id=None, run_id="test")
    assert not any(
        t.name in {"workspace.list_files", "workspace.read_file", "workspace.search", "workspace.write_file", "workspace.edit_file"}
        for t in registry.definitions_for_context(context)
    )


@pytest.mark.parametrize("invalid", ["conflict", "missing"])
def test_invalid_workspace_disables_tools_not_app(monkeypatch, tmp_path, caplog, invalid):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "local")
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("GEIST_EXEC_WORKSPACE", str(tmp_path / invalid))
    if invalid == "missing":
        monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path / invalid))
    registry = build_default_tool_registry()
    context = ToolContext(workspace_id=1, chat_id=None, run_id="test")
    names = {tool.name for tool in registry.definitions_for_context(context)}
    assert "web.search" in names
    assert "terminal.run" not in names
    assert "workspace.read_file" not in names
    assert "Coding tools disabled" in caplog.text


@pytest.mark.parametrize("tool", ["workspace.write_file", "workspace.edit_file"])
def test_host_file_mutation_cannot_waive_per_call_approval(monkeypatch, tmp_path, tool):
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    registry = build_default_tool_registry()
    context = ToolContext(
        workspace_id=1,
        chat_id=7,
        run_id="test",
        permission_mode="auto_approve",
        always_allow_tools=frozenset({tool}),
    )
    arguments = (
        {"path": "feature.txt", "content": "new"}
        if tool.endswith("write_file")
        else {
            "path": "feature.txt",
            "old_text": "old",
            "new_text": "new",
        }
    )
    assert registry.execute(ToolCall.create(tool, arguments), context).error == "approval_required"
    assert not (tmp_path / "feature.txt").exists()


@pytest.mark.skipif(os.getenv("GEIST_TEST_DOCKER") != "1", reason="Explicit Docker integration run")
def test_docker_file_edits_and_terminal_share_new_chat_workspace(monkeypatch):
    monkeypatch.delenv("GEIST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    workspace = CodingWorkspace(DockerExecutionEnvironment())
    context = ToolContext(workspace_id=1, chat_id=None, run_id=f"qa-{uuid.uuid4().hex}")
    try:
        workspace.file_operation(context, "write_file", {"path": "feature.txt", "content": "hello"})
        result = workspace.run(context, "cat feature.txt", 10)
        assert result.exit_code == 0
        assert result.stdout == "hello"
        result = workspace.run(context, "echo changed > feature.txt", 10)
        assert result.exit_code == 0
        assert (
            workspace.file_operation(context, "read_file", {"path": "feature.txt"})["content"]
            == "changed\n"
        )
    finally:
        workspace.sessions.shutdown()
