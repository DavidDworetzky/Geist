import threading
from unittest.mock import patch

import pytest

from app.services.tool_approvals import (
    SessionGrantRegistry,
    ToolApprovalRegistry,
    persist_always_allow,
)


def _request(registry, run_id, call_id, tool_name):
    return registry.request(
        run_id,
        call_id,
        tool_name,
        workspace_id=1,
        arguments_fingerprint="arguments",
        definition_fingerprint="definition",
        can_grant=True,
    )


def test_resolve_unblocks_waiter_with_decision():
    registry = ToolApprovalRegistry()
    pending = _request(registry, "run_1", "call_1", "web.search")

    resolver = threading.Timer(
        0.05, lambda: registry.resolve("run_1", "call_1", "session", workspace_id=1)
    )
    resolver.start()
    decision = registry.wait(pending, timeout_seconds=2.0)
    assert decision == "session"
    assert registry.has_pending("run_1") is False


def test_wait_times_out_to_deny_permission_stack():
    registry = ToolApprovalRegistry()
    pending = _request(registry, "run_1", "call_1", "web.search")
    assert registry.wait(pending, timeout_seconds=0.05) == "deny"
    assert registry.has_pending("run_1") is False


def test_wait_cancellation_denies_permission_stack():
    registry = ToolApprovalRegistry()
    pending = _request(registry, "run_1", "call_1", "web.search")
    cancellation = threading.Event()
    cancellation.set()
    assert registry.wait(pending, timeout_seconds=5.0, cancellation=cancellation) == "deny"


def test_resolve_unknown_call_returns_false_and_bad_decision_raises_permission_stack():
    registry = ToolApprovalRegistry()
    assert registry.resolve("run_x", "call_x", "approve", workspace_id=1) is False
    _request(registry, "run_1", "call_1", "web.search")
    with pytest.raises(ValueError):
        registry.resolve("run_1", "call_1", "yolo", workspace_id=1)


def test_cancel_run_denies_only_its_pending_calls():
    registry = ToolApprovalRegistry()
    first = _request(registry, "run_1", "call_1", "a")
    second = _request(registry, "run_1", "call_2", "b")
    other = _request(registry, "run_2", "call_1", "c")

    registry.cancel_run("run_1")
    assert first.decision == "deny" and first.event.is_set()
    assert second.decision == "deny" and second.event.is_set()
    assert other.decision is None
    assert registry.has_pending("run_2") is True


def test_session_grants_are_scoped_permission_stack():
    grants = SessionGrantRegistry()
    grants.grant("chat:1", "web.search")
    assert grants.granted("chat:1") == frozenset({"web.search"})
    assert grants.granted("chat:2") == frozenset()
    grants.clear("chat:1")
    assert grants.granted("chat:1") == frozenset()


def test_new_instruction_retires_pending_approval_atomically():
    registry = ToolApprovalRegistry()
    pending = _request(registry, "run", "call", "tool")
    interruption = threading.Event()
    interruption.set()
    assert registry.wait(pending, 5, interruption=interruption) == "interrupted"
    assert pending.denial_reason == "superseded"
    assert not registry.resolve("run", "call", "approve", workspace_id=1)
    assert registry.pending() == []


def test_session_grants_expire_and_bind_the_reviewed_definition():
    now = [0.0]
    grants = SessionGrantRegistry(ttl_seconds=10, clock=lambda: now[0])
    grants.grant("workspace:1:chat:1", "web.search", "first")
    assert grants.allows("workspace:1:chat:1", "web.search", "first")
    assert not grants.allows("workspace:1:chat:1", "web.search", "changed")
    assert not grants.allows("workspace:2:chat:1", "web.search", "first")
    now[0] = 10
    assert not grants.allows("workspace:1:chat:1", "web.search", "first")
    assert grants.granted("workspace:1:chat:1") == frozenset()


def test_first_turn_grant_moves_to_persisted_chat_without_extending_ttl():
    now = [0.0]
    grants = SessionGrantRegistry(ttl_seconds=10, clock=lambda: now[0])
    grants.grant("workspace:1:run:first", "tool", "fingerprint")
    now[0] = 9
    grants.promote_run(1, "first", 7)
    assert grants.allows("workspace:1:chat:7", "tool", "fingerprint")
    assert not grants.allows("workspace:1:run:first", "tool", "fingerprint")
    now[0] = 10
    assert not grants.allows("workspace:1:chat:7", "tool", "fingerprint")


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan")])
def test_invalid_session_grant_ttl_is_rejected(ttl):
    with pytest.raises(ValueError, match="finite and positive"):
        SessionGrantRegistry(ttl_seconds=ttl)


def test_expired_approval_cannot_win_a_late_resolve():
    registry = ToolApprovalRegistry()
    pending = registry.request(
        "run",
        "call",
        "tool",
        workspace_id=1,
        arguments_fingerprint="a",
        definition_fingerprint="d",
        timeout_seconds=0,
    )
    assert not registry.resolve("run", "call", "approve", workspace_id=1)
    assert registry.poll(pending) == "deny"
    assert pending.denial_reason == "timeout"
    assert registry.pending() == []


@pytest.mark.parametrize("decision", ["session", "always"])
def test_invocation_only_approval_rejects_standing_grants(decision):
    registry = ToolApprovalRegistry()
    pending = registry.request(
        "run",
        "call",
        "tool",
        workspace_id=1,
        arguments_fingerprint="a",
        definition_fingerprint="d",
        can_grant=False,
    )
    with pytest.raises(ValueError, match="standing grants"):
        registry.resolve("run", "call", decision, workspace_id=1)
    assert registry.pending() == [pending]
    assert registry.resolve("run", "call", "approve", workspace_id=1)


def test_old_pending_handle_does_not_settle_reused_key():
    registry = ToolApprovalRegistry()
    old = _request(registry, "run", "call", "tool")
    registry.resolve("run", "call", "deny", workspace_id=1)
    new = _request(registry, "run", "call", "tool")
    cancelled = threading.Event()
    cancelled.set()
    assert registry.poll(old, cancelled) == "deny"
    assert registry.pending() == [new]
    assert new.decision is None


@pytest.mark.parametrize("existing", [["a.tool"], ["web.search"]])
def test_persist_always_allow_merges_without_losing_settings(existing):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.database.database import Base
    from app.models.database.user_settings import UserSettings

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    with sessions() as session:
        session.add(
            UserSettings(
                user_id=1,
                agent_permissions={
                    "mode": "require_approval",
                    "always_allow": existing,
                },
            )
        )
        session.commit()
    with patch("app.models.database.database.SessionLocal", sessions):
        persist_always_allow(1, "web.search")
    with sessions() as session:
        stored = session.query(UserSettings).one()
        assert stored.agent_permissions == {
            "mode": "require_approval",
            "always_allow": sorted({*existing, "web.search"}),
        }
    engine.dispose()


def test_standing_grant_retries_after_concurrent_settings_change(tmp_path):
    import json

    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    from app.models.database.database import Base
    from app.models.database.user_settings import UserSettings

    engine = create_engine(f"sqlite:///{tmp_path / 'concurrent-grant.sqlite3'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    with sessions() as session:
        session.add(
            UserSettings(user_id=1, agent_permissions={"mode": "default", "always_allow": []})
        )
        session.commit()
    changed = []

    def concurrent_update(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE user_settings") and not changed:
            changed.append(True)
            with engine.begin() as other:
                other.exec_driver_sql(
                    "UPDATE user_settings SET agent_permissions = ?",
                    (json.dumps({"mode": "require_approval", "always_allow": ["other.tool"]}),),
                )

    event.listen(engine, "before_cursor_execute", concurrent_update)
    try:
        with patch("app.models.database.database.SessionLocal", sessions):
            persist_always_allow(1, "web.search")
        with sessions() as session:
            assert session.query(UserSettings).one().agent_permissions == {
                "mode": "require_approval",
                "always_allow": ["other.tool", "web.search"],
            }
        assert changed == [True]
    finally:
        event.remove(engine, "before_cursor_execute", concurrent_update)
        engine.dispose()


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
