from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agents.models.tool_calling import (
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolCall,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.agent_permissions import AgentPermissions
from app.services.chat_orchestrator import ChatOrchestrator
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.tool_registry import ToolRegistry, build_default_tool_registry


class EmptyArguments(BaseModel):
    pass


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("workspace.write_file", {"path": "app.py", "content": "hello"}),
        ("workspace.edit_file", {"path": "app.py", "old_text": "hello", "new_text": "hi"}),
        ("terminal.run", {"command": "echo hi"}),
    ],
)
def test_unattended_durable_mutations_are_denied_even_with_auto_approve(
    monkeypatch, tmp_path, tool, arguments
):
    monkeypatch.setenv("GEIST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GEIST_EXEC_BACKEND", "docker")
    monkeypatch.setattr(DockerExecutionEnvironment, "is_available", lambda _: True)
    registry = build_default_tool_registry()

    def unexpected_execution(*args):
        pytest.fail("An unattended durable mutation executed")

    registry.get(tool).handler = unexpected_execution
    observed = []

    class Backend:
        supports_native_tool_calling = True

        def stream_model_turn(self, messages, tools, config):
            if messages[-1].role == "tool":
                observed.append(messages[-1].content)
                turn = ModelTurn(text="Needs interactive approval")
            else:
                turn = ModelTurn(tool_calls=[ToolCall.create(tool, arguments)])
            yield ModelEvent.turn_complete(turn)

    orchestrator = ChatOrchestrator(
        registry,
        history_loader=lambda _: [],
        history_writer=lambda **_: SimpleNamespace(chat_session_id=42),
        permissions_loader=lambda _: AgentPermissions(
            mode="auto_approve", always_allow=frozenset({tool})
        ),
    )
    events = list(
        orchestrator.stream(
            backend=Backend(),
            prompt="Work",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            interactive=False,
        )
    )
    assert len(observed) == 1
    assert "unattended" in observed[0]
    assert any(event.event == "final" for event in events)


def test_new_chat_has_persistent_identity_before_first_tool():
    creations, contexts, writes = [], [], []

    def create(owner, **options):
        creations.append((owner, options))
        return 42

    def read(context, arguments):
        assert creations
        contexts.append((context.workspace_id, context.chat_id))
        return ToolExecutionOutput(content="found")

    class Backend:
        supports_native_tool_calling = True

        def stream_model_turn(self, messages, tools, config):
            turn = (
                ModelTurn(text="Done")
                if messages[-1].role == "tool"
                else ModelTurn(tool_calls=[ToolCall.create("files.read")])
            )
            yield ModelEvent.turn_complete(turn)

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="files.read", description="Read", arguments_model=EmptyArguments, handler=read
        )
    )
    orchestrator = ChatOrchestrator(
        registry,
        chat_creator=create,
        history_loader=lambda _: [],
        history_writer=lambda **kwargs: writes.append(kwargs)
        or SimpleNamespace(chat_session_id=42),
        permissions_loader=lambda _: AgentPermissions(mode="default"),
    )
    for chat_id in (None, 42):
        events = list(
            orchestrator.stream(
                backend=Backend(),
                prompt="Read",
                workspace_id=7,
                chat_id=chat_id,
                config=ModelRequestConfig(),
                system_prompt=None,
                memory_enabled=False,
                memory_mode="private",
                folder_id=3,
            )
        )
        assert next(e.payload for e in events if e.event == "final").chat_id == 42
        assert next(e.payload for e in events if e.event == "run_started")["chat_id"] == 42
    assert creations == [(7, {"memory_enabled": False, "memory_mode": "private", "folder_id": 3})]
    assert contexts == [(7, 42), (7, 42)]
    assert all(write["session_id"] == 42 for write in writes)


def test_failed_chat_allocation_cannot_execute_tools():
    def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    orchestrator = ChatOrchestrator(ToolRegistry(), chat_creator=fail)
    events = list(
        orchestrator.stream(
            backend=object(),
            prompt="Work",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    assert [event.event for event in events] == ["error", "done"]
    assert events[0].payload["code"] == "chat_allocation_failed"
    assert "database availability" in events[0].payload["message"]


def test_permission_loading_failure_retains_identity_but_denies_execution():
    created = []
    observed = []

    def fail(owner):
        raise RuntimeError("permissions unavailable")

    class Backend:
        supports_native_tool_calling = True

        def stream_model_turn(self, messages, tools, config):
            if messages[-1].role == "tool":
                observed.append(messages[-1].content)
                turn = ModelTurn(text="Approval required")
            else:
                turn = ModelTurn(tool_calls=[ToolCall.create("files.read")])
            yield ModelEvent.turn_complete(turn)

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="files.read",
            description="Read",
            arguments_model=EmptyArguments,
            handler=lambda *_: pytest.fail("Executed after permissions failed"),
        )
    )
    orchestrator = ChatOrchestrator(
        registry,
        permissions_loader=fail,
        chat_creator=lambda *args, **kwargs: created.append(args) or 42,
        history_loader=lambda _: [],
        history_writer=lambda **_: SimpleNamespace(chat_session_id=42),
    )
    list(
        orchestrator.stream(
            backend=Backend(),
            prompt="Work",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            interactive=False,
        )
    )
    assert created == [(1,)]
    assert len(observed) == 1
    assert "unattended" in observed[0]
