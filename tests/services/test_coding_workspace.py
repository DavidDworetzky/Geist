import os
import uuid
from unittest.mock import Mock

import pytest

from agents.models.tool_calling import ToolContext
from app.services.coding_workspace import CodingWorkspace
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.local import LocalExecutionEnvironment


def test_file_and_local_terminal_use_one_root(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    environment = LocalExecutionEnvironment()
    workspace = CodingWorkspace(environment)
    context = ToolContext(user_id=1, chat_id=None, run_id="test")
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
    environment = DockerExecutionEnvironment()
    workspace = CodingWorkspace(environment)
    assert environment.workspace == str(tmp_path)
    assert environment.requires_per_call_approval
    first = ToolContext(user_id=1, chat_id=None, run_id="first", workspace_id="goal-workspace")
    resumed = ToolContext(user_id=1, chat_id=7, run_id="second", workspace_id="goal-workspace")
    assert workspace.scope(first) == workspace.scope(resumed)


@pytest.mark.skipif(os.getenv("GEIST_TEST_DOCKER") != "1", reason="Explicit Docker integration run")
def test_docker_file_edits_and_terminal_share_new_chat_workspace(monkeypatch):
    monkeypatch.delenv("GEIST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    workspace = CodingWorkspace(DockerExecutionEnvironment())
    context = ToolContext(user_id=1, chat_id=None, run_id=f"qa-{uuid.uuid4().hex}")
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
