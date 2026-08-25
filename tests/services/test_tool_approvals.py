import threading
import time
from types import SimpleNamespace

from pydantic import BaseModel

from agents.models.tool_calling import (
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolCall,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.chat_orchestrator import ChatOrchestrator
from app.services.tool_approvals import ToolApprovalRegistry
from app.services.tool_registry import ToolRegistry


class Arguments(BaseModel):
    value: str


class Backend:
    supports_native_tool_calling = True

    def __init__(self):
        self.turns = iter(
            [
                ModelTurn(
                    tool_calls=[
                        ToolCall(id="call-1", name="external.write", arguments={"value": "x"})
                    ]
                ),
                ModelTurn(text="done"),
            ]
        )
        self.requests = []

    def stream_model_turn(self, messages, tools, config):
        self.requests.append([message.to_dict() for message in messages])
        yield ModelEvent.turn_complete(next(self.turns))


def _orchestrator(approvals, calls):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="external.write",
            description="Write externally",
            arguments_model=Arguments,
            handler=lambda context, arguments: (
                calls.append(arguments.value),
                ToolExecutionOutput(content="written"),
            )[1],
            requires_approval=True,
        )
    )
    return ChatOrchestrator(
        registry,
        approvals=approvals,
        approval_timeout_seconds=2,
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1),
    )


def _resolve_next(approvals, decision):
    def resolve():
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            pending = approvals.pending()
            if pending:
                approvals.resolve(
                    pending[0].run_id,
                    pending[0].call_id,
                    decision,
                    workspace_id=pending[0].workspace_id,
                )
                return
            time.sleep(0.01)

    thread = threading.Thread(target=resolve, daemon=True)
    thread.start()
    return thread


def test_approval_resolution_is_scoped_to_the_owning_user():
    approvals = ToolApprovalRegistry()
    pending = approvals.request(
        "run-1",
        "call-1",
        "external.write",
        workspace_id=7,
        arguments_fingerprint="arguments",
        definition_fingerprint="definition",
    )

    assert not approvals.resolve("run-1", "call-1", "approve", workspace_id=8)
    assert approvals.pending() == [pending]
    assert approvals.resolve("run-1", "call-1", "approve", workspace_id=7)
    assert pending.event.is_set()


def test_approved_call_resumes_and_executes_once():
    approvals = ToolApprovalRegistry()
    calls = []
    backend = Backend()
    _resolve_next(approvals, "approve")

    events = list(
        _orchestrator(approvals, calls).stream(
            backend=backend,
            prompt="write",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    assert calls == ["x"]
    assert [event.payload.status for event in events if event.event == "tool_call"] == [
        "proposed",
        "awaiting_approval",
        "running",
        "succeeded",
    ]


def test_denied_call_fails_closed_and_reenters_model_as_blocked():
    approvals = ToolApprovalRegistry()
    calls = []
    backend = Backend()
    _resolve_next(approvals, "deny")

    events = list(
        _orchestrator(approvals, calls).stream(
            backend=backend,
            prompt="write",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    assert calls == []
    assert [event.payload.status for event in events if event.event == "tool_call"] == [
        "proposed",
        "awaiting_approval",
        "failed",
    ]
    assert "BLOCKED" in backend.requests[1][-1]["content"]


def test_unattended_call_denies_without_waiting():
    approvals = ToolApprovalRegistry()
    calls = []
    backend = Backend()

    events = list(
        _orchestrator(approvals, calls).stream(
            backend=backend,
            prompt="write",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            interactive=False,
        )
    )

    assert calls == []
    assert any(
        event.event == "tool_call" and event.payload.error == "approval_denied" for event in events
    )


def test_nonstream_completion_never_waits_for_interactive_approval():
    approvals = ToolApprovalRegistry()
    calls = []
    backend = Backend()
    started_at = time.monotonic()

    completion = _orchestrator(approvals, calls).complete(
        backend=backend,
        prompt="write",
        workspace_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )

    assert time.monotonic() - started_at < 0.5
    assert calls == []
    assert completion.message == ["done"]
    assert approvals.pending() == []
