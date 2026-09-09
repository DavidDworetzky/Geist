import os
import uuid
from unittest.mock import Mock

import pytest

from agents.models.tool_calling import ToolCall, ToolContext
from app.services.coding_workspace import CodingWorkspace, WorkspaceOperationError
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.factory import create_execution_environment
from app.services.execution.local import LocalExecutionEnvironment
from app.services.tool_registry import build_default_tool_registry


@pytest.fixture(autouse=True)
def private_data_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_DATA_DIR", str(tmp_path / "geist-data"))
    monkeypatch.delenv("GEIST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    monkeypatch.delenv("GEIST_EXEC_BACKEND", raising=False)


def test_default_directory_is_per_chat_and_owner_and_survives_restart(tmp_path):
    first = ToolContext(workspace_id=1, chat_id=7, run_id="first", coding_workspace_id="goal-a")
    resumed = ToolContext(workspace_id=1, chat_id=7, run_id="second", coding_workspace_id="goal-b")
    workspace = CodingWorkspace(None)
    workspace.file_operation(first, "write_file", {"path": "app.py", "content": "hello"})
    expected = tmp_path / "geist-data/code/workspace-1/chat-7"
    assert workspace.directory(first) == expected
    assert (
        CodingWorkspace(None).file_operation(resumed, "read_file", {"path": "app.py"})["content"]
        == "hello"
    )
    for other in (
        ToolContext(workspace_id=1, chat_id=8, run_id="other"),
        ToolContext(workspace_id=2, chat_id=7, run_id="other"),
    ):
        assert workspace.directory(other) != expected
        assert workspace.file_operation(other, "list_files", {}) == []


def test_default_local_terminal_shares_file_directory():
    workspace = CodingWorkspace(LocalExecutionEnvironment())
    context = ToolContext(workspace_id=1, chat_id=7, run_id="test")
    workspace.file_operation(context, "write_file", {"path": "app.py", "content": "hello"})
    result = workspace.run(context, "cat app.py", 10)
    assert result.exit_code == 0
    assert result.stdout == "hello"


def test_managed_docker_environment_reports_durable_host_posture():
    environment = DockerExecutionEnvironment()
    CodingWorkspace(environment)
    assert environment.workspace is None
    assert environment.has_host_access
    assert not environment.is_sandboxed
    assert environment.requires_per_call_approval
    assert "durable host workspace" in environment.describe()
    assert environment.command_rejection_reason("rm -rf /workspace") is not None


@pytest.mark.skipif(os.getenv("GEIST_TEST_DOCKER") != "1", reason="Explicit Docker integration run")
def test_missing_daemon_bind_source_is_not_created(tmp_path):
    missing = tmp_path / "absent"
    with pytest.raises(ValueError, match="existing directory"):
        DockerExecutionEnvironment(workspace=str(missing))
    assert not missing.exists()


@pytest.mark.parametrize("component", ["code", "code/workspace-1", "code/workspace-1/chat-7"])
def test_managed_directories_reject_symlinks(tmp_path, component):
    external = tmp_path / "outside"
    external.mkdir()
    link = tmp_path / "geist-data" / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(external, target_is_directory=True)
    with pytest.raises(WorkspaceOperationError, match="symlinks"):
        CodingWorkspace(None).directory(ToolContext(workspace_id=1, chat_id=7, run_id="test"))
    assert list(external.iterdir()) == []


@pytest.mark.parametrize("tool", ["workspace.write_file", "workspace.edit_file", "terminal.run"])
def test_durable_mutations_cannot_be_auto_approved(monkeypatch, tool):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setattr(DockerExecutionEnvironment, "is_available", lambda _: True)
    registry = build_default_tool_registry()
    context = ToolContext(
        workspace_id=1,
        chat_id=7,
        run_id="test",
        permission_mode="auto_approve",
        always_allow_tools=frozenset({tool}),
    )
    definition = registry.get(tool)
    assert definition.requires_per_call_approval
    arguments = (
        {"command": "echo hi"}
        if tool == "terminal.run"
        else {"path": "app.py", "content": "hello"}
        if tool.endswith("write_file")
        else {"path": "app.py", "old_text": "hello", "new_text": "hi"}
    )
    assert registry.execute(ToolCall.create(tool, arguments), context).error == "approval_required"


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
    first = ToolContext(workspace_id=1, chat_id=7, run_id="first", coding_workspace_id="goal-a")
    resumed = ToolContext(workspace_id=1, chat_id=7, run_id="second", coding_workspace_id="goal-b")
    assert workspace.scope(first) == workspace.scope(resumed)


def test_default_host_workspace_is_available_with_managed_root(monkeypatch):
    monkeypatch.setattr(DockerExecutionEnvironment, "is_available", lambda _: True)
    monkeypatch.delenv("GEIST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    monkeypatch.delenv("GEIST_EXEC_BACKEND", raising=False)
    registry = build_default_tool_registry()
    context = ToolContext(workspace_id=1, chat_id=None, run_id="test")
    names = {tool.name for tool in registry.definitions_for_context(context)}
    assert {
        "workspace.list_files",
        "workspace.read_file",
        "workspace.search",
        "workspace.write_file",
        "workspace.edit_file",
    } <= names


def test_missing_container_runtime_hides_file_tools_and_reports_correct_remedy(monkeypatch):
    monkeypatch.setattr(DockerExecutionEnvironment, "is_available", lambda _: False)
    workspace = CodingWorkspace(DockerExecutionEnvironment())
    assert not workspace.available
    with pytest.raises(WorkspaceOperationError, match="Docker or Podman"):
        workspace.file_operation(
            ToolContext(workspace_id=1, chat_id=7, run_id="test"), "list_files", {}
        )


def test_chat_cleanup_uses_coding_workspace_scope(monkeypatch):
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setattr(DockerExecutionEnvironment, "is_available", lambda _: True)
    registry = build_default_tool_registry()
    registry.session_manager = Mock()
    registry.close_chat(1, 7)
    registry.session_manager.close_scope.assert_called_once_with(
        registry.coding_workspace.scope(ToolContext(workspace_id=1, chat_id=7, run_id=""))
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
        workspace.sessions.shutdown()
        restarted = CodingWorkspace(DockerExecutionEnvironment())
        try:
            result = restarted.run(context, "cat feature.txt", 10)
            assert result.exit_code == 0
            assert result.stdout == "changed\n"
        finally:
            restarted.sessions.shutdown()
    finally:
        workspace.sessions.shutdown()
