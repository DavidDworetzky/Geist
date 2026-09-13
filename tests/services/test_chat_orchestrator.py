import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agents.models.chat_result import WorkArtifact
from agents.models.tool_calling import (
    ChatMessage,
    MalformedToolCallError,
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolCall,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.agent_permissions import AgentPermissions
from app.services.chat_orchestrator import ChatOrchestrator, RunControlRegistry
from app.services.goal_runtime import GoalRuntime, GoalRuntimeRegistry
from app.services.tool_approvals import SessionGrantRegistry, ToolApprovalRegistry
from app.services.tool_intent_router import ToolIntentRouter
from app.services.tool_registry import ToolRegistry, build_default_tool_registry


class LookupArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


def test_instruction_checkpoint_does_not_hold_global_control_lock():
    started, release = threading.Event(), threading.Event()
    controls = RunControlRegistry()

    def persist(item):
        started.set()
        assert release.wait(2)

    cancellation = controls.start("run", workspace_id=1, on_instruction=persist)
    worker = threading.Thread(target=lambda: controls.enqueue("run", 1, "one", "Use local"))
    worker.start()
    try:
        assert started.wait(1)
        assert not controls.seal("run")
        controls.cancel("run", workspace_id=1)
        assert cancellation.is_set()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()


def test_failed_instruction_enqueue_is_not_accepted_or_replayed():
    def fail(snapshot):
        raise RuntimeError("database unavailable")

    runtime = GoalRuntime(objective="Build", tasks=[], max_turns=5, on_change=fail)
    controls = RunControlRegistry()
    controls.start("run", workspace_id=1, on_instruction=runtime.add_instruction)
    with pytest.raises(RuntimeError):
        controls.enqueue("run", 1, "one", "Use local")
    assert controls.drain("run") == []
    assert runtime.state.instructions == []
    assert controls.seal("run")


def test_model_context_bounds_all_roles_without_orphaning_tool_results():
    orchestrator = ChatOrchestrator(ToolRegistry(), max_history_chars=1200, max_history_entries=4)
    call = ToolCall.create("read", {"huge": "x" * 2000})
    messages = [
        ChatMessage(role="user", content="old" * 2000),
        ChatMessage(role="assistant", tool_calls=[call]),
        ChatMessage(role="tool", tool_call_id=call.id, content="output"),
        ChatMessage(role="user", content="latest instruction"),
    ]
    context = orchestrator._model_context(messages)
    assert len(json.dumps([item.to_dict() for item in context], ensure_ascii=False)) <= 1200
    assert orchestrator._has_complete_tool_sequence(context)
    assert context[-1].content == "latest instruction"


def test_recent_denial_survives_bounded_context_and_trusted_checkpoint():
    orchestrator = ChatOrchestrator(ToolRegistry(), max_history_chars=1200, max_history_entries=4)
    call = ToolCall.create("write", {"path": "app.py"})
    denied = ChatMessage(
        role="tool", tool_call_id=call.id, content="User denied this action.", preserve_content=True
    )
    messages = [
        ChatMessage(role="assistant", tool_calls=[call]),
        denied,
        ChatMessage(role="assistant", content="later prose" * 500),
        ChatMessage(role="user", content="Continue safely"),
    ]
    checkpoint = orchestrator._checkpoint_messages(messages)
    restored = [orchestrator._checkpoint_message(item) for item in checkpoint]
    assert any(item.preserve_content and item.content == denied.content for item in restored)
    assert orchestrator._has_complete_tool_sequence(restored)
    assert len(json.dumps([item.to_dict() for item in restored])) <= 1200
    assert not ChatMessage.from_dict(checkpoint[1]).preserve_content


class ScriptedBackend:
    supports_native_tool_calling = True

    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    def stream_model_turn(self, messages, tools, config):
        self.requests.append(
            {
                "messages": [message.to_dict() for message in messages],
                "tools": [tool.name for tool in tools],
            }
        )
        turn = next(self.turns)
        if isinstance(turn, Exception):
            raise turn
        if turn.text:
            yield ModelEvent.text_delta(turn.text)
        yield ModelEvent.turn_complete(turn)


def test_simple_agentic_answer_finishes_after_one_model_call():
    runs = GoalRuntimeRegistry()
    backend = ScriptedBackend([ModelTurn(text="Hello, how can I help?")])
    orchestrator = ChatOrchestrator(
        build_default_tool_registry(runs),
        orchestration_runs=runs,
        history_writer=lambda **_: SimpleNamespace(chat_session_id=1),
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Hi",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            agentic_mode=True,
        )
    )
    final = next(event.payload for event in events if event.event == "final")
    assert len(backend.requests) == 1
    assert final.orchestration["goal_status"] == "complete"
    assert final.message == ["Hello, how can I help?"]


def test_unknown_native_tool_is_retried_without_dispatch():
    backend = ScriptedBackend(
        [
            ModelTurn(tool_calls=[ToolCall.create("imaginary.tool")]),
            ModelTurn(text="Corrected answer"),
        ]
    )
    orchestrator = ChatOrchestrator(
        ToolRegistry(), history_writer=lambda **_: SimpleNamespace(chat_session_id=1)
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Hi",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    assert len(backend.requests) == 2
    assert not any(
        message["role"] == "assistant"
        and not message.get("content")
        and not message.get("tool_calls")
        for message in backend.requests[1]["messages"]
    )
    assert not any(event.event == "tool_call" for event in events)
    assert next(event.payload for event in events if event.event == "final").message == [
        "Corrected answer"
    ]


def test_malformed_generation_retries_without_replaying_successful_tools():
    executions = []
    registry = ToolRegistry()

    def lookup(context, arguments):
        executions.append(arguments.query)
        return ToolExecutionOutput(content="saved result")

    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search",
            arguments_model=LookupArguments,
            handler=lookup,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(tool_calls=[ToolCall("first", "documents.search", {"query": "once"})]),
            MalformedToolCallError("invalid JSON"),
            ModelTurn(text="Recovered"),
        ]
    )
    writes = []

    def persist(**kwargs):
        writes.append(kwargs)
        return SimpleNamespace(chat_session_id=42)

    orchestrator = ChatOrchestrator(registry, history_loader=lambda _: [], history_writer=persist)
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Search",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    assert executions == ["once"]
    assert len(backend.requests) == 3
    assert (
        "No tools from that attempted turn executed"
        in backend.requests[2]["messages"][-1]["content"]
    )
    assert any(
        m["role"] == "tool" and m["content"] == "saved result"
        for m in backend.requests[2]["messages"]
    )
    assert any(event.event == "final" for event in events)
    assert writes[0]["status"] == "completed"


@pytest.mark.parametrize("agentic_mode", [False, True])
@pytest.mark.parametrize("budget,expected_calls", [(1, 1), (2, 2), (8, 3)])
def test_malformed_retries_respect_run_and_goal_budgets(agentic_mode, budget, expected_calls):
    backend = ScriptedBackend([MalformedToolCallError("invalid JSON")] * 8)
    writes = []
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        history_loader=lambda _: [],
        history_writer=lambda **kwargs: writes.append(kwargs)
        or SimpleNamespace(chat_session_id=42),
        max_rounds=budget,
        goal_max_turns=budget,
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Work",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            agentic_mode=agentic_mode,
        )
    )
    assert len(backend.requests) == expected_calls
    error = next(event.payload for event in events if event.event == "error")
    assert error["code"] == "malformed_tool_call"
    assert "recovery budget" in error["message"]
    assert not any(event.event == "final" for event in events)
    if agentic_mode:
        assert writes[0]["orchestration"]["turns_used"] == expected_calls


@pytest.mark.parametrize("cancel", [False, True])
def test_recovery_closes_stream_and_accepts_user_input(cancel):
    controls = RunControlRegistry()
    closed = []

    class PartialBackend(ScriptedBackend):
        def stream_model_turn(self, messages, tools, config):
            if not closed:
                try:
                    yield ModelEvent.text_delta("Working.")
                    raise MalformedToolCallError("bad JSON")
                finally:
                    closed.append(True)
            else:
                yield from super().stream_model_turn(messages, tools, config)

    backend = PartialBackend([ModelTurn(text="Recovered")])
    writes = []
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        run_controls=controls,
        history_loader=lambda _: [],
        history_writer=lambda **kwargs: writes.append(kwargs)
        or SimpleNamespace(chat_session_id=42),
    )
    stream = orchestrator.stream(
        backend=backend,
        prompt="Work",
        workspace_id=7,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )
    run_id = next(stream).payload["run_id"]
    for event in stream:
        if event.event == "delta" and "Retrying malformed" in event.payload["text"]:
            assert closed == [True]
            if cancel:
                assert controls.cancel(run_id, workspace_id=7)
            else:
                assert controls.enqueue(run_id, 7, "steer", "Do not install packages")
            break
    events = list(stream)
    assert writes[0]["new_ai_message"].count("Working.") == 1
    if cancel:
        assert not backend.requests
        assert any(event.event == "cancelled" for event in events)
    else:
        assert any(
            "Do not install packages" in (m["content"] or "")
            for m in backend.requests[0]["messages"]
        )
        assert any(event.event == "final" for event in events)


def test_non_protocol_value_error_is_not_retried():
    backend = ScriptedBackend([ValueError("backend configuration invalid")])
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        history_loader=lambda _: [],
        history_writer=lambda **_: SimpleNamespace(chat_session_id=42),
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Work",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    assert len(backend.requests) == 1
    assert next(event.payload for event in events if event.event == "error")["code"] == "chat_error"


class RecordingGoalStore:
    def __init__(self):
        self.created = []
        self.updated = []
        self.attached = []
        self.latest = None

    def load_latest(self, user_id, chat_id):
        return deepcopy(self.latest)

    def create(self, snapshot, user_id, run_id):
        self.created.append((snapshot, user_id, run_id))

    def update(self, snapshot):
        self.updated.append(snapshot)
        self.latest = deepcopy(snapshot)

    def attach_chat(self, goal_id, chat_id):
        self.attached.append((goal_id, chat_id))


class FailingIntentRouter(ToolIntentRouter):
    def classify(self, backend, messages):
        del backend, messages
        raise RuntimeError("classifier unavailable")


def test_classifier_failure_falls_back_to_action_tools_without_image_generation():
    registry = ToolRegistry()
    for name, tags in [
        ("public.search", frozenset({"public_retrieval"})),
        ("local.search", frozenset({"local_retrieval"})),
        ("computer.use", frozenset({"action"})),
        ("image.generate", frozenset({"image_generation"})),
    ]:
        registry.register(
            ToolDefinition(
                name=name,
                description=name,
                arguments_model=LookupArguments,
                handler=lambda _context, _arguments: ToolExecutionOutput(content="unused"),
                semantic_tags=tags,
            )
        )
    backend = ScriptedBackend([ModelTurn(text="Fallback answer", finish_reason="stop")])
    orchestrator = ChatOrchestrator(
        registry,
        intent_router=FailingIntentRouter(),
        history_loader=lambda _chat_id: [],
        history_writer=lambda **_kwargs: SimpleNamespace(chat_session_id=42),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Explain this",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt="Assistant prompt",
            enable_intent_router=True,
        )
    )

    assert backend.requests[0]["tools"] == [
        "public.search",
        "local.search",
        "computer.use",
    ]
    assert next(event.payload for event in events if event.event == "final").message == [
        "Fallback answer"
    ]


@pytest.mark.parametrize("routing_options", [{}, {"enable_intent_router": False}])
def test_disabled_intent_router_exposes_full_catalog_without_classifying(routing_options):
    registry = ToolRegistry()
    for name, tags in [
        ("public.search", frozenset({"public_retrieval"})),
        ("computer.use", frozenset({"action"})),
        ("image.generate", frozenset({"image_generation"})),
    ]:
        registry.register(
            ToolDefinition(
                name=name,
                description=name,
                arguments_model=LookupArguments,
                handler=lambda _context, _arguments: ToolExecutionOutput(content="unused"),
                semantic_tags=tags,
            )
        )
    backend = ScriptedBackend([ModelTurn(text="Unrouted answer", finish_reason="stop")])
    orchestrator = ChatOrchestrator(
        registry,
        intent_router=FailingIntentRouter(),
        history_loader=lambda _chat_id: [],
        history_writer=lambda **_kwargs: SimpleNamespace(chat_session_id=42),
    )

    list(
        orchestrator.stream(
            backend=backend,
            prompt="Use the full catalog",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt="Assistant prompt",
            **routing_options,
        )
    )

    assert backend.requests[0]["tools"] == [
        "public.search",
        "computer.use",
        "image.generate",
    ]


@pytest.mark.parametrize(
    ("intent", "needs_retrieval", "expected_tools"),
    [
        ("answer", False, []),
        ("answer", True, ["public.search", "local.search"]),
        ("sensitive_answer", False, []),
        ("sensitive_answer", True, ["local.search"]),
        ("action", False, ["public.search", "local.search", "computer.use"]),
        ("image_generation", False, ["image.generate"]),
    ],
)
def test_intent_router_filters_catalog_before_assistant_turn(
    intent, needs_retrieval, expected_tools
):
    registry = ToolRegistry()
    for name, tags in [
        ("public.search", frozenset({"public_retrieval"})),
        ("local.search", frozenset({"local_retrieval"})),
        ("computer.use", frozenset({"action"})),
        ("image.generate", frozenset({"image_generation"})),
    ]:
        registry.register(
            ToolDefinition(
                name=name,
                description=name,
                arguments_model=LookupArguments,
                handler=lambda _context, _arguments: ToolExecutionOutput(content="unused"),
                semantic_tags=tags,
            )
        )
    backend = ScriptedBackend(
        [
            ModelTurn(
                text=(
                    f'{{"intent":"{intent}",' f'"needs_retrieval":{str(needs_retrieval).lower()}}}'
                ),
                finish_reason="stop",
            ),
            ModelTurn(text="Direct response", finish_reason="stop"),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        intent_router=ToolIntentRouter(),
        history_loader=lambda _chat_id: [],
        history_writer=lambda **_kwargs: SimpleNamespace(chat_session_id=42),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Handle this request",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt="Assistant prompt",
            enable_intent_router=True,
        )
    )

    assert backend.requests[0]["tools"] == []
    assert backend.requests[1]["tools"] == expected_tools
    assert [event.payload["text"] for event in events if event.event == "delta"] == [
        "Direct response"
    ]


@pytest.mark.parametrize("tools_disabled", [False, True])
def test_unoffered_tool_cannot_execute_or_enter_persisted_transcript(tools_disabled):
    executed, writes = [], []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="public.search",
            description="Search publicly",
            arguments_model=LookupArguments,
            handler=lambda *args: executed.append(True),
            semantic_tags=frozenset({"public_retrieval"}),
        )
    )
    turns = (
        []
        if tools_disabled
        else [ModelTurn(text='{"intent":"sensitive_answer","needs_retrieval":true}')]
    )
    turns.append(
        ModelTurn(
            text="Looking into it.",
            tool_calls=[ToolCall(id="bad", name="public.search", arguments={"query": "private"})],
        )
    )
    backend = ScriptedBackend(turns)

    def write(**snapshot):
        writes.append(snapshot)
        return SimpleNamespace(chat_session_id=42)

    orchestrator = ChatOrchestrator(
        registry,
        intent_router=ToolIntentRouter(),
        history_loader=lambda _: [],
        history_writer=write,
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Private request",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt="Assistant",
            enable_tools=not tools_disabled,
            enable_intent_router=True,
        )
    )
    assert executed == []
    assert any(event.event == "error" for event in events)
    assert not any(event.event == "tool_call" for event in events)
    assert writes[-1]["transcript"][-1]["content"] == "Looking into it."
    assert all(not message.get("tool_calls") for message in writes[-1]["transcript"])


def test_completed_goal_gives_way_to_new_request_preserving_chat_history():
    store = RecordingGoalStore()
    old = GoalRuntime(objective="Old work", tasks=[], max_turns=1)
    assert old.complete("Old work answered", ["Answer delivered"])["accepted"]
    store.latest = old.checkpoint()
    runtimes = GoalRuntimeRegistry()
    orchestrator = ChatOrchestrator(
        build_default_tool_registry(runtimes),
        orchestration_runs=runtimes,
        goal_store=store,
        history_loader=lambda _: [{"user": "Useful history", "ai": "Remembered"}],
        history_writer=lambda **_: SimpleNamespace(chat_session_id=7),
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="wait",
                        name="agent.goal.wait",
                        arguments={"question": "Which new feature?"},
                    )
                ]
            )
        ]
    )
    result = orchestrator.complete(
        backend=backend,
        prompt="New work",
        workspace_id=1,
        chat_id=7,
        system_prompt=None,
        config=ModelRequestConfig(),
        agentic_mode=True,
    )
    assert result.orchestration["goal_id"] != old.state.goal_id
    assert result.orchestration["objective"] == "New work"
    assert result.orchestration["tasks"] == []
    assert any(m["content"] == "Useful history" for m in backend.requests[0]["messages"])
    assert all(s["goal_id"] != old.state.goal_id for s in store.updated)


def test_invalid_checkpoint_requests_recovery_without_overwriting_it():
    store = RecordingGoalStore()
    store.latest = {"objective": "Keep me", "agentic_mode": True, "tasks": [{"invalid": True}]}
    saved = deepcopy(store.latest)
    backend = ScriptedBackend([])
    events = list(
        ChatOrchestrator(
            ToolRegistry(),
            goal_store=store,
            history_loader=lambda _: [],
            history_writer=lambda **_: SimpleNamespace(chat_session_id=7),
        ).stream(
            backend=backend,
            prompt="Continue",
            workspace_id=1,
            chat_id=7,
            system_prompt=None,
            config=ModelRequestConfig(),
            agentic_mode=True,
        )
    )
    error = next(e.payload for e in events if e.event == "error")
    assert error["code"] == "checkpoint_recovery_required"
    assert "Start a new chat" in error["message"]
    assert store.latest == saved
    assert not backend.requests


def test_duplicate_run_reports_actionable_error_and_keeps_original_run():
    controls = RunControlRegistry()
    controls.start("original", workspace_id=1, chat_id=7)
    writes = []
    backend = ScriptedBackend([])
    events = list(
        ChatOrchestrator(
            ToolRegistry(),
            run_controls=controls,
            history_loader=lambda _: [],
            history_writer=lambda **kwargs: writes.append(kwargs),
        ).stream(
            backend=backend,
            prompt="Duplicate",
            workspace_id=1,
            chat_id=7,
            system_prompt=None,
            config=ModelRequestConfig(),
        )
    )
    assert events[0].payload["code"] == "run_already_active"
    assert "send instructions" in events[0].payload["message"]
    assert controls.enqueue("original", 1, "one", "Still here")
    assert not writes and not backend.requests


def test_non_agentic_steering_is_returned_and_persisted():
    controls = RunControlRegistry()
    writes = []
    backend = ScriptedBackend([ModelTurn(text="Working"), ModelTurn(text="Updated answer")])
    backend.supports_native_tool_calling = False
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        run_controls=controls,
        history_writer=lambda **kwargs: writes.append(kwargs) or SimpleNamespace(chat_session_id=7),
    )
    run_id = None
    events = []
    for event in orchestrator.stream(
        backend=backend,
        prompt="Answer",
        workspace_id=1,
        chat_id=None,
        system_prompt=None,
        config=ModelRequestConfig(),
    ):
        events.append(event)
        if event.event == "run_started":
            run_id = event.payload["run_id"]
        if event.event == "delta" and event.payload["text"] == "Working":
            controls.enqueue(run_id, 1, "one", "Use local")
    result = next(e.payload for e in events if e.event == "final")
    assert result.orchestration is None
    assert result.instructions == [{"id": "one", "text": "Use local", "status": "applied"}]
    assert writes[0]["instructions"] == result.instructions
    assert backend.requests[1]["messages"][-1]["content"] == "Use local"


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///tmp/secret", "not-a-url"])
def test_artifact_urls_reject_unsafe_schemes(url):
    with pytest.raises(ValueError, match="HTTP"):
        WorkArtifact.from_url(url, kind="text", mime_type="text/plain")


def test_tool_result_reenters_model_context_and_turn_persists_once():
    calls = []

    def lookup(context, arguments):
        calls.append((context.workspace_id, arguments.query))
        return ToolExecutionOutput(
            content='{"answer": "2023-tax-return.pdf"}', summary="Found tax return"
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search documents",
            arguments_model=LookupArguments,
            handler=lookup,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="call_1", name="documents.search", arguments={"query": "tax return"}
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="I found 2023-tax-return.pdf.", finish_reason="stop"),
        ]
    )
    writes = []

    def write_history(**kwargs):
        writes.append(kwargs)
        return SimpleNamespace(chat_session_id=42)

    orchestrator = ChatOrchestrator(
        registry,
        history_loader=lambda chat_id: [],
        history_writer=write_history,
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Find my tax return",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt="Use tools when needed.",
        )
    )

    assert calls == [(7, "tax return")]
    assert backend.requests[0]["tools"] == ["documents.search"]
    invocation_id = next(event.payload.id for event in events if event.event == "tool_call")
    assert backend.requests[1]["messages"][-1] == {
        "role": "tool",
        "content": '{"answer": "2023-tax-return.pdf"}',
        "tool_call_id": invocation_id,
        "name": "documents.search",
    }
    assert [event.payload.status for event in events if event.event == "tool_call"] == [
        "proposed",
        "running",
        "succeeded",
    ]
    assert len(writes) == 1
    assert invocation_id != "call_1"
    assert writes[0]["transcript"][1]["tool_calls"][0]["id"] == invocation_id
    completion = next(event.payload for event in events if event.event == "final")
    assert completion.chat_id == 42
    assert completion.tool_calls[0].id == invocation_id
    assert completion.message == ["I found 2023-tax-return.pdf."]


def test_agentic_mode_lets_executor_create_plan_and_complete():
    runtime_registry = GoalRuntimeRegistry()
    goal_store = RecordingGoalStore()
    registry = build_default_tool_registry(runtime_registry)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="plan_1",
                        name="agent.plan.update",
                        arguments={
                            "updates": [
                                {
                                    "task_id": "task-1",
                                    "title": "Implement feature",
                                    "status": "completed",
                                    "evidence": "Focused test passes",
                                }
                            ]
                        },
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="goal_1",
                        name="agent.goal.complete",
                        arguments={
                            "summary": "Feature implemented",
                            "evidence": ["Focused test passes"],
                        },
                    )
                ]
            ),
            ModelTurn(text="Implemented and verified."),
        ]
    )
    writes = []
    orchestrator = ChatOrchestrator(
        registry,
        orchestration_runs=runtime_registry,
        goal_store=goal_store,
        history_writer=lambda **kwargs: writes.append(kwargs)
        or SimpleNamespace(chat_session_id=41),
        permissions_loader=lambda user_id: AgentPermissions(mode="require_approval"),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Add a feature",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt="Use tools.",
            enable_tools=False,
            agentic_mode=True,
        )
    )

    assert backend.requests[0]["tools"] == [
        "agent.plan.update",
        "agent.goal.complete",
        "agent.goal.wait",
    ]
    assert [event.event for event in events].count("plan") == 2
    assert not any(
        event.event == "tool_call" and event.payload.status == "awaiting_approval"
        for event in events
    )
    completion = next(event.payload for event in events if event.event == "final")
    assert completion.message == ["Feature implemented"]
    assert completion.orchestration["goal_status"] == "complete"
    assert completion.orchestration["turns_used"] == 2
    assert writes[0]["orchestration"]["tasks"][0]["status"] == "completed"
    assert len(goal_store.created) == 1
    assert goal_store.attached == [(completion.orchestration["goal_id"], 41)]


def test_agentic_mode_stops_without_success_claim_at_budget():
    runtime_registry = GoalRuntimeRegistry()
    registry = build_default_tool_registry(runtime_registry)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall.create(
                        "agent.plan.update",
                        {"updates": [{"task_id": "finish", "title": "Finish work"}]},
                    )
                ]
            ),
            ModelTurn(text="Still working."),
            ModelTurn(text="More remains."),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        orchestration_runs=runtime_registry,
        goal_max_turns=2,
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Do a large task",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            agentic_mode=True,
        )
    )

    completion = next(event.payload for event in events if event.event == "final")
    assert completion.orchestration["goal_status"] == "budget_limited"
    assert completion.orchestration["turns_used"] == 2
    assert "without claiming completion" in completion.message[0]
    assert len(backend.requests) == 2


def test_waiting_goal_resumes_its_objective_and_transcript():
    runtimes = GoalRuntimeRegistry()
    store = RecordingGoalStore()
    orchestrator = ChatOrchestrator(
        build_default_tool_registry(runtimes),
        orchestration_runs=runtimes,
        goal_store=store,
        history_loader=lambda _: [],
        history_writer=lambda **_: SimpleNamespace(chat_session_id=7),
        permissions_loader=lambda _: AgentPermissions(mode="default"),
    )
    first = orchestrator.complete(
        backend=ScriptedBackend(
            [
                ModelTurn(
                    tool_calls=[
                        ToolCall(
                            id="wait1",
                            name="agent.goal.wait",
                            arguments={"question": "Which provider?"},
                        )
                    ]
                )
            ]
        ),
        prompt="Implement voice notes",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
        agentic_mode=True,
    )
    assert first.message == ["Which provider?"]
    assert first.orchestration["goal_status"] == "waiting_for_user"
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="done1",
                        name="agent.goal.complete",
                        arguments={
                            "summary": "Configured local provider",
                            "evidence": ["User selected local"],
                        },
                    )
                ]
            )
        ]
    )
    second = orchestrator.complete(
        backend=backend,
        prompt="Use local",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
        agentic_mode=True,
    )
    assert second.orchestration["goal_id"] == first.orchestration["goal_id"]
    assert second.orchestration["objective"] == "Implement voice notes"
    assert second.orchestration["workspace_id"] == first.orchestration["workspace_id"]
    assert any(m["content"] == "Implement voice notes" for m in backend.requests[0]["messages"])
    assert backend.requests[0]["messages"][-1]["content"] == "Use local"
    assert len(store.created) == 1


@pytest.mark.parametrize("at_status", ["proposed", "awaiting_approval", "running"])
def test_live_instruction_invalidates_unstarted_command(at_status):
    runtimes = GoalRuntimeRegistry()
    controls = RunControlRegistry()
    registry = build_default_tool_registry(runtimes)
    executed = []
    registry.register(
        ToolDefinition(
            name="mock.write",
            description="Mock write",
            arguments_model=LookupArguments,
            handler=lambda context, args: (
                executed.append(args.query),
                ToolExecutionOutput(content="written"),
            )[1],
            requires_approval=True,
            requires_per_call_approval=True,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="write1", name="mock.write", arguments={"query": "old instruction"})
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="wait1",
                        name="agent.goal.wait",
                        arguments={"question": "What should I do instead?"},
                    )
                ]
            ),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        orchestration_runs=runtimes,
        run_controls=controls,
        history_loader=lambda _: [],
        history_writer=lambda **_: SimpleNamespace(chat_session_id=7),
        permissions_loader=lambda _: AgentPermissions(mode="default"),
    )
    run_id = None
    events = []
    for event in orchestrator.stream(
        backend=backend,
        prompt="Write it",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
        agentic_mode=True,
    ):
        events.append(event)
        if event.event == "run_started":
            run_id = event.payload["run_id"]
        if (
            at_status == "running"
            and event.event == "tool_call"
            and event.payload.status == "awaiting_approval"
        ):
            assert orchestrator.approvals.resolve(
                run_id, event.payload.id, "approve", workspace_id=1
            )
        if (
            event.event == "tool_call"
            and event.payload.name == "mock.write"
            and event.payload.status == at_status
        ):
            assert controls.enqueue(run_id, 2, "wrong-owner", "no") is None
            assert controls.enqueue(run_id, 1, "instruction1", "Do not write anything") is not None
    assert executed == []
    assert any(
        m["role"] == "user" and m["content"] == "Do not write anything"
        for m in backend.requests[-1]["messages"]
    )
    assert ChatOrchestrator._has_complete_tool_sequence(
        [ChatMessage.from_dict(m) for m in backend.requests[-1]["messages"]]
    )
    assert any(e.event == "user_instruction" for e in events)
    assert controls.enqueue(run_id, 1, "too-late", "Late") is None


def test_instruction_delivery_is_idempotent_even_after_drain():
    controls = RunControlRegistry()
    controls.start("run", workspace_id=1)
    controls.enqueue("run", 1, "instruction", "Use local")
    assert len(controls.drain("run")) == 1
    controls.enqueue("run", 1, "instruction", "Use local")
    assert controls.drain("run") == []
    with pytest.raises(ValueError, match="different text"):
        controls.enqueue("run", 1, "instruction", "Use hosted")


def test_reused_provider_id_cannot_reuse_command_approval():
    class Approvals(ToolApprovalRegistry):
        requested = 0

        def wait(self, pending, *args, **kwargs):
            self.requested += 1
            self.resolve(pending.run_id, pending.call_id, "approve", workspace_id=1)
            return "approve"

    approvals = Approvals()
    executed = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="terminal.run",
            description="Mock only",
            arguments_model=LookupArguments,
            handler=lambda context, args: (
                executed.append(args.query),
                ToolExecutionOutput(content="ok"),
            )[1],
            requires_approval=True,
            requires_per_call_approval=True,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[ToolCall(id="same", name="terminal.run", arguments={"query": "first"})]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(id="same", name="terminal.run", arguments={"query": "different"})
                ]
            ),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        approvals=approvals,
        history_writer=lambda **_: SimpleNamespace(chat_session_id=1),
        permissions_loader=lambda _: AgentPermissions(mode="default"),
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="test",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    assert executed == ["first", "different"]
    assert approvals.requested == 2
    assert any(event.event == "error" for event in events)


def test_artifact_bytes_are_live_but_not_persisted_inline():
    artifact = WorkArtifact.from_bytes(
        b"image bytes",
        kind="image",
        mime_type="image/png",
        filename="generated.png",
    )

    def generate(context, arguments):
        return ToolExecutionOutput(content="generated", artifacts=[artifact])

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="image.generate",
            description="Generate image",
            arguments_model=LookupArguments,
            handler=generate,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="image_1", name="image.generate", arguments={"query": "cat"})
                ]
            ),
            ModelTurn(text="Done"),
        ]
    )
    writes = []
    orchestrator = ChatOrchestrator(
        registry,
        history_writer=lambda **kwargs: writes.append(kwargs) or SimpleNamespace(chat_session_id=1),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Make a cat",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    live_artifact = next(event.payload for event in events if event.event == "artifact")
    assert live_artifact.data_base64
    assert "data_base64" not in writes[0]["artifacts"][0]


def test_round_limit_emits_error_and_does_not_persist():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search",
            arguments_model=LookupArguments,
            handler=lambda context, arguments: ToolExecutionOutput(content="no match"),
        )
    )
    repeating = ModelTurn(
        tool_calls=[ToolCall(id="repeat", name="documents.search", arguments={"query": "x"})]
    )
    backend = ScriptedBackend([repeating, repeating])
    writes = []
    orchestrator = ChatOrchestrator(
        registry,
        max_rounds=2,
        history_writer=lambda **kwargs: writes.append(kwargs),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="loop",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    assert len(writes) == 1
    assert writes[0]["status"] == "failed"
    assert writes[0]["tool_calls"][0].status == "succeeded"
    assert "round limit exceeded" in next(
        event.payload["message"] for event in events if event.event == "error"
    )


def test_run_can_be_cancelled_after_run_started():
    controls = RunControlRegistry()
    writes = []
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        run_controls=controls,
        history_writer=lambda **kwargs: writes.append(kwargs) or SimpleNamespace(chat_session_id=9),
    )
    backend = ScriptedBackend([ModelTurn(text="should not run")])
    stream = orchestrator.stream(
        backend=backend,
        prompt="stop",
        workspace_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )

    started = next(stream)
    assert not controls.cancel(started.payload["run_id"], workspace_id=2)
    assert controls.cancel(started.payload["run_id"], workspace_id=1)
    cancelled = next(stream)
    assert cancelled.event == "cancelled"
    assert cancelled.payload["chat_id"] == 9
    assert writes[0]["status"] == "cancelled"
    with pytest.raises(StopIteration):
        next(stream)


def test_cancel_ack_persists_even_when_browser_closes_stream():
    controls = RunControlRegistry()
    writes = []
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        run_controls=controls,
        history_writer=lambda **kwargs: writes.append(kwargs)
        or SimpleNamespace(chat_session_id=17),
    )
    stream = orchestrator.stream(
        backend=ScriptedBackend([ModelTurn(text="unused")]),
        prompt="cancel and disconnect",
        workspace_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )

    started = next(stream)
    run_id = started.payload["run_id"]
    assert controls.cancel(run_id, workspace_id=1)
    assert len(writes) == 1
    assert writes[0]["status"] == "cancelled"
    assert writes[0]["run_id"] == run_id

    # Mirrors the frontend aborting its SSE request immediately after the
    # cancellation endpoint acknowledges durable cancellation.
    stream.close()

    assert len(writes) == 1
    assert not controls.cancel(run_id, workspace_id=1)


def test_disconnect_explicitly_closes_backend_even_if_iterator_is_retained():
    closed = []

    def responses():
        try:
            yield ModelEvent.text_delta("visible")
            yield ModelEvent.turn_complete(ModelTurn(text="visible"))
        finally:
            closed.append(True)

    retained = responses()
    backend = SimpleNamespace(
        supports_native_tool_calling=False,
        stream_model_turn=lambda *args: retained,
    )
    stream = ChatOrchestrator(
        ToolRegistry(), history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1)
    ).stream(
        backend=backend,
        prompt="hello",
        workspace_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )
    while next(stream).event != "delta":
        pass
    stream.close()
    assert closed == [True]


@pytest.mark.parametrize("ending", ["malformed", "disconnect", "cancel"])
def test_partial_streamed_prose_is_persisted_once_without_unvalidated_tools(ending):
    from agents.architectures.chat_template_tools import ToolResponseStream

    def responses(*args):
        parser = ToolResponseStream({"safe": "web.search"})
        yield ModelEvent.text_delta(parser.feed("Working. "))
        parser.feed("<tool_call>{bad}</tool_call>")

    controls = RunControlRegistry()
    writes = []
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        max_rounds=1,
        run_controls=controls,
        history_writer=lambda **kwargs: writes.append(kwargs) or SimpleNamespace(chat_session_id=3),
    )
    stream = orchestrator.stream(
        backend=SimpleNamespace(supports_native_tool_calling=False, stream_model_turn=responses),
        prompt="hello",
        workspace_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )
    run_id = next(stream).payload["run_id"]
    while (event := next(stream)).event != "delta":
        pass
    assert event.payload["text"] == "Working."
    if ending == "malformed":
        events = list(stream)
        assert (
            next(event for event in events if event.event == "error").payload["code"]
            == "malformed_tool_call"
        )
    else:
        if ending == "cancel":
            assert controls.cancel(run_id, workspace_id=1)
        stream.close()
    assert len(writes) == 1
    assert writes[0]["status"] == ("failed" if ending == "malformed" else "cancelled")
    assert writes[0]["new_ai_message"] == "Working."
    assert not writes[0]["tool_calls"]
    assistant = writes[0]["transcript"][-1]
    assert assistant["content"] == "Working."
    assert not assistant.get("tool_calls")


def test_backend_without_native_tools_uses_one_direct_turn_when_agentic_is_requested():
    backend = ScriptedBackend([ModelTurn(text="local answer")])
    backend.supports_native_tool_calling = False
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="web.search",
            description="Search",
            arguments_model=LookupArguments,
            handler=lambda context, arguments: ToolExecutionOutput(content="result"),
        )
    )
    orchestrator = ChatOrchestrator(
        registry,
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=4),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="news",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            agentic_mode=True,
        )
    )

    assert backend.requests[0]["tools"] == []
    assert len(backend.requests) == 1
    assert not any(event.event in {"plan", "goal"} for event in events)
    assert next(event.payload for event in events if event.event == "final").message == [
        "local answer"
    ]


def test_aggregate_tool_result_budget_truncates_model_context():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search",
            arguments_model=LookupArguments,
            handler=lambda context, arguments: ToolExecutionOutput(content="x" * 200),
            max_result_chars=200,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[ToolCall(id="large", name="documents.search", arguments={"query": "x"})]
            ),
            ModelTurn(text="done"),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        max_tool_result_chars_total=80,
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1),
    )

    list(
        orchestrator.stream(
            backend=backend,
            prompt="search",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_message = backend.requests[1]["messages"][-1]
    assert len(tool_message["content"]) <= 80
    assert "truncated by context budget" in tool_message["content"]


def test_agentic_sequence_exceeds_old_round_and_tool_limits():
    runtimes = GoalRuntimeRegistry()
    registry = build_default_tool_registry(runtimes)
    registry.register(
        ToolDefinition(
            name="mock.read",
            description="Read",
            arguments_model=LookupArguments,
            handler=lambda context, args: ToolExecutionOutput(content=args.query * 150),
        )
    )

    class LongBackend(ScriptedBackend):
        def stream_model_turn(self, messages, tools, config):
            count = len(self.requests)
            if count < 12:
                turn = ModelTurn(
                    tool_calls=[
                        ToolCall(
                            id=f"read-{count}",
                            name="mock.read",
                            arguments={"query": f"result-{count}"},
                        )
                    ]
                )
            else:
                refs = [m.tool_call_id for m in messages if m.role == "tool"]
                turn = ModelTurn(
                    tool_calls=[
                        ToolCall(
                            id="done",
                            name="agent.goal.complete",
                            arguments={
                                "summary": "Done",
                                "evidence": ["Read all files"],
                                "evidence_refs": refs,
                            },
                        )
                    ]
                )
            self.turns = iter([turn])
            yield from super().stream_model_turn(messages, tools, config)

    backend = LongBackend([])
    store = RecordingGoalStore()
    result = ChatOrchestrator(
        registry,
        orchestration_runs=runtimes,
        goal_store=store,
        goal_max_turns=15,
        max_tool_result_chars_total=200,
        history_writer=lambda **_: SimpleNamespace(chat_session_id=7),
        permissions_loader=lambda _: AgentPermissions(mode="default"),
    ).complete(
        backend=backend,
        prompt="Read all files",
        workspace_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
        agentic_mode=True,
    )
    assert result.orchestration["goal_status"] == "complete"
    assert result.orchestration["turns_used"] == 13
    assert len(result.orchestration["observations"]) == 12
    for request in backend.requests[1:]:
        messages = [ChatMessage.from_dict(m) for m in request["messages"]]
        assert ChatOrchestrator._has_complete_tool_sequence(messages)
        outputs = [m.content for m in messages if m.role == "tool"]
        assert sum(len(output) for output in outputs) <= 200
        assert outputs[-1]  # New observations never disappear behind older results.
    assert "result-11" in backend.requests[-1]["messages"][-1]["content"]
    assert store.latest["transcript"][-1]["role"] == "tool"


def test_instruction_at_budget_is_durable_and_consumed_on_resume():
    controls = RunControlRegistry()
    runtimes = GoalRuntimeRegistry()
    store = RecordingGoalStore()
    orchestrator = ChatOrchestrator(
        build_default_tool_registry(runtimes),
        run_controls=controls,
        orchestration_runs=runtimes,
        goal_store=store,
        goal_max_turns=1,
        history_loader=lambda _: [],
        history_writer=lambda **_: SimpleNamespace(chat_session_id=7),
    )
    run_id = None

    class SteeredBackend(ScriptedBackend):
        def stream_model_turn(self, messages, tools, config):
            assert controls.enqueue(run_id, 1, "late", "Use local only")
            yield from super().stream_model_turn(messages, tools, config)

    events = []
    for event in orchestrator.stream(
        backend=SteeredBackend([ModelTurn(text="Working")]),
        prompt="Add voice notes",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
        agentic_mode=True,
    ):
        events.append(event)
        if event.event == "run_started":
            run_id = event.payload["run_id"]
    first = next(e.payload for e in events if e.event == "final")
    assert first.orchestration["goal_status"] == "budget_limited"
    assert store.latest["instructions"][-1]["status"] == "queued"
    assert controls.enqueue(run_id, 1, "closed", "Too late") is None
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="wait",
                        name="agent.goal.wait",
                        arguments={"question": "Which local model?"},
                    )
                ]
            )
        ]
    )
    second = orchestrator.complete(
        backend=backend,
        prompt="Continue",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
        agentic_mode=True,
    )
    assert first.orchestration["goal_id"] == second.orchestration["goal_id"]
    assert second.orchestration["turns_used"] == 2
    assert second.orchestration["goal_status"] == "waiting_for_user"
    assert sum(m["content"] == "Use local only" for m in backend.requests[0]["messages"]) == 1


def test_instruction_after_completion_claim_reopens_goal():
    controls = RunControlRegistry()
    runtimes = GoalRuntimeRegistry()
    orchestrator = ChatOrchestrator(
        build_default_tool_registry(runtimes),
        run_controls=controls,
        orchestration_runs=runtimes,
        history_writer=lambda **_: SimpleNamespace(chat_session_id=7),
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="done",
                        name="agent.goal.complete",
                        arguments={"summary": "Original done", "evidence": ["Answered"]},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="wait",
                        name="agent.goal.wait",
                        arguments={"question": "Which extra detail?"},
                    )
                ]
            ),
        ]
    )
    events = []
    run_id = None
    for event in orchestrator.stream(
        backend=backend,
        prompt="Answer",
        workspace_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
        agentic_mode=True,
    ):
        events.append(event)
        if event.event == "run_started":
            run_id = event.payload["run_id"]
        if event.event == "goal" and event.payload["goal_status"] == "complete":
            assert controls.enqueue(run_id, 1, "extra", "Add more detail")
    result = next(e.payload for e in events if e.event == "final")
    assert result.message == ["Which extra detail?"]
    assert backend.requests[-1]["messages"][-1]["content"] == "Add more detail"


def test_history_budget_keeps_only_complete_recent_turns():
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        max_history_entries=2,
        max_history_chars=10_000,
    )
    history = [
        {"user": "first", "ai": "one"},
        {"user": "second", "ai": "two"},
        {"user": "third", "ai": "three"},
    ]

    messages = orchestrator._history_messages(history)

    assert [message.content for message in messages] == ["second", "two", "third", "three"]


@pytest.mark.parametrize("status", ["failed", "cancelled", "completed"])
def test_history_compacts_provider_invalid_tool_sequences(status):
    orchestrator = ChatOrchestrator(ToolRegistry())
    history = [
        {
            "user": "Find both files",
            "ai": "The run ended early.",
            "status": status,
            "transcript": [
                {"role": "user", "content": "Find both files"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "documents.search",
                            "arguments": {"query": "one"},
                        },
                        {
                            "id": "call_2",
                            "name": "documents.search",
                            "arguments": {"query": "two"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": "first result",
                    "tool_call_id": "call_1",
                    "name": "documents.search",
                },
            ],
        }
    ]

    messages = orchestrator._history_messages(history)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == [
        "Find both files",
        "The run ended early.",
    ]


def test_generation_memory_failure_reaches_chat_error_event():
    from agents.model_load_errors import ModelMemoryError

    error = ModelMemoryError("unified_memory", runtime="mlx_llama", during_generation=True)
    backend = ScriptedBackend([error])
    backend.model_id = "test/model"
    writes = []
    orchestrator = ChatOrchestrator(
        ToolRegistry(),
        history_writer=lambda **kwargs: writes.append(kwargs) or 42,
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Hello",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    payload = next(event.payload for event in events if event.event == "error")
    assert payload["message"] == str(error)
    assert payload["model_load"]["model_id"] == "test/model"
    assert payload["model_load"]["error_code"] == "unified_memory"
    assert payload["model_load"]["can_offload_to_system_ram"] is False
    assert payload["model_load"]["state"] == "failed"
    assert writes[0]["status"] == "failed"
    assert events[-1].event == "done"
    assert not any(event.event == "final" for event in events)


def test_persistence_failure_does_not_emit_unpersisted_final():
    backend = ScriptedBackend([ModelTurn(text="answer")])
    write_attempts = []

    def fail_write(**kwargs):
        write_attempts.append(kwargs)
        raise RuntimeError("database unavailable")

    orchestrator = ChatOrchestrator(ToolRegistry(), history_writer=fail_write)

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="hello",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    assert not any(event.event == "final" for event in events)
    error_event = next(event for event in events if event.event == "error")
    assert error_event.payload["message"] == "Chat completion failed"
    assert "database unavailable" not in error_event.payload["message"]
    assert [attempt["status"] for attempt in write_attempts] == ["completed", "failed"]


def test_doom_loop_interrupts_repeated_identical_tool_calls():
    executions = []

    def lookup(context, arguments):
        executions.append(arguments.query)
        return ToolExecutionOutput(content='{"answer": "same thing"}')

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search documents",
            arguments_model=LookupArguments,
            handler=lookup,
        )
    )

    def repeated_turn(call_id):
        return ModelTurn(
            tool_calls=[ToolCall(id=call_id, name="documents.search", arguments={"query": "loop"})],
            finish_reason="tool_calls",
        )

    backend = ScriptedBackend(
        [repeated_turn("call_1"), repeated_turn("call_2"), repeated_turn("call_3")]
    )
    writes = []

    def write_history(**kwargs):
        writes.append(kwargs)
        return SimpleNamespace(chat_session_id=42)

    orchestrator = ChatOrchestrator(
        registry,
        history_loader=lambda chat_id: [],
        history_writer=write_history,
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Loop forever",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    # The third identical call is interrupted before execution.
    assert executions == ["loop", "loop"]
    error = next(event.payload for event in events if event.event == "error")
    assert error["message"].startswith("Doom loop detected")
    assert "documents.search" in error["message"]
    assert writes[0]["status"] == "failed"


def test_doom_loop_not_triggered_by_varied_arguments():
    def lookup(context, arguments):
        return ToolExecutionOutput(content='{"answer": "ok"}')

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search documents",
            arguments_model=LookupArguments,
            handler=lookup,
        )
    )

    def turn(call_id, query):
        return ModelTurn(
            tool_calls=[ToolCall(id=call_id, name="documents.search", arguments={"query": query})],
            finish_reason="tool_calls",
        )

    backend = ScriptedBackend(
        [
            turn("call_1", "alpha"),
            turn("call_2", "beta"),
            turn("call_3", "alpha"),
            ModelTurn(text="Done.", finish_reason="stop"),
        ]
    )

    orchestrator = ChatOrchestrator(
        registry,
        history_loader=lambda chat_id: [],
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=42),
    )
    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="Search a few things",
            workspace_id=7,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    completion = next(event.payload for event in events if event.event == "final")
    assert completion.message == ["Done."]
    assert not [event for event in events if event.event == "error"]


def test_auto_approve_permissions_execute_approval_gated_tool():
    def send(context, arguments):
        return ToolExecutionOutput(content="sent")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="communication.email.send",
            description="Send email",
            arguments_model=LookupArguments,
            handler=send,
            requires_approval=True,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="call_1", name="communication.email.send", arguments={"query": "hi"}
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Sent.", finish_reason="stop"),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        history_loader=lambda chat_id: [],
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1),
        permissions_loader=lambda workspace_id: AgentPermissions(mode="auto_approve"),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="send it",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == ["proposed", "running", "succeeded"]
    assert all(state.requires_approval is False for state in tool_states)


@pytest.mark.parametrize("loader_fails", [False, True])
def test_require_approval_permissions_gate_read_only_tool(loader_fails):
    def load_permissions(_workspace_id):
        if loader_fails:
            raise RuntimeError("injected loader failed")
        return AgentPermissions(mode="require_approval")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search documents",
            arguments_model=LookupArguments,
            handler=lambda context, arguments: ToolExecutionOutput(content="found"),
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_1", name="documents.search", arguments={"query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Waiting on approval.", finish_reason="stop"),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        history_loader=lambda chat_id: [],
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1),
        permissions_loader=load_permissions,
        approvals=ToolApprovalRegistry(),
        grants=SessionGrantRegistry(),
        approval_timeout_seconds=0,
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="search",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    # No decision arrives, so the approval times out and fails closed.
    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert tool_states[-1].error == "approval_timeout"
    assert [state.status for state in tool_states] == ["proposed", "awaiting_approval", "failed"]
    assert all(state.requires_approval is True for state in tool_states)


def test_always_allow_permissions_skip_approval_for_listed_tool():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="workspace.write_markdown",
            description="Write markdown",
            arguments_model=LookupArguments,
            handler=lambda context, arguments: ToolExecutionOutput(content="written"),
            requires_approval=True,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_1", name="workspace.write_markdown", arguments={"query": "x"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Done.", finish_reason="stop"),
        ]
    )
    orchestrator = ChatOrchestrator(
        registry,
        history_loader=lambda chat_id: [],
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1),
        permissions_loader=lambda workspace_id: AgentPermissions(
            mode="require_approval",
            always_allow=frozenset({"workspace.write_markdown"}),
        ),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="write",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == ["proposed", "running", "succeeded"]


def _resolver(approvals: ToolApprovalRegistry, decision: str, decisions_made: list):
    """Background thread: resolve each pending approval with `decision`."""
    import threading
    import time

    def resolve_loop():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for pending in approvals.pending():
                approvals.resolve(
                    pending.run_id, pending.call_id, decision, workspace_id=pending.workspace_id
                )
                decisions_made.append((pending.tool_name, decision))
                return
            time.sleep(0.01)

    thread = threading.Thread(target=resolve_loop, daemon=True)
    thread.start()
    return thread


def _approval_orchestrator(registry, *, approvals, grants=None, persister=None):
    return ChatOrchestrator(
        registry,
        history_loader=lambda chat_id: [],
        history_writer=lambda **kwargs: SimpleNamespace(chat_session_id=1),
        permissions_loader=lambda user_id: AgentPermissions(mode="require_approval"),
        approvals=approvals,
        grants=grants or SessionGrantRegistry(),
        approval_timeout_seconds=5.0,
        always_allow_persister=persister or (lambda user_id, tool_name: None),
    )


def _gated_registry(calls):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="documents.search",
            description="Search documents",
            arguments_model=LookupArguments,
            handler=lambda context, arguments: (
                calls.append(arguments.query),
                ToolExecutionOutput(content="found"),
            )[1],
        )
    )
    return registry


def test_approval_approve_resumes_and_executes():
    calls = []
    approvals = ToolApprovalRegistry()
    registry = _gated_registry(calls)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_1", name="documents.search", arguments={"query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Found it.", finish_reason="stop"),
        ]
    )
    orchestrator = _approval_orchestrator(registry, approvals=approvals)
    decisions = []
    _resolver(approvals, "approve", decisions)

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="search",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == [
        "proposed",
        "awaiting_approval",
        "running",
        "succeeded",
    ]
    assert calls == ["q"]
    assert decisions == [("documents.search", "approve")]


def test_per_call_terminal_approval_ignores_standing_grant():
    calls = []
    approvals = ToolApprovalRegistry()
    grants = SessionGrantRegistry()
    grants.grant("chat:7", "terminal.run")
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="terminal.run",
            description="Run a protected host command",
            arguments_model=LookupArguments,
            handler=lambda context, arguments: (
                calls.append(arguments.query),
                ToolExecutionOutput(content="ran"),
            )[1],
            requires_approval=True,
            requires_per_call_approval=True,
        )
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[ToolCall(id="call_1", name="terminal.run", arguments={"query": "pwd"})],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Done.", finish_reason="stop"),
        ]
    )
    orchestrator = _approval_orchestrator(
        registry,
        approvals=approvals,
        grants=grants,
    )
    _resolver(approvals, "approve", [])

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="run it",
            workspace_id=1,
            chat_id=7,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == [
        "proposed",
        "awaiting_approval",
        "running",
        "succeeded",
    ]
    assert all(state.requires_per_call_approval is True for state in tool_states)
    assert calls == ["pwd"]


def test_approval_deny_blocks_and_tells_model():
    calls = []
    approvals = ToolApprovalRegistry()
    registry = _gated_registry(calls)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_1", name="documents.search", arguments={"query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Understood.", finish_reason="stop"),
        ]
    )
    orchestrator = _approval_orchestrator(registry, approvals=approvals)
    _resolver(approvals, "deny", [])

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="search",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == [
        "proposed",
        "awaiting_approval",
        "failed",
    ]
    assert calls == []
    tool_message = backend.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "denied" in tool_message["content"]
    assert "Do NOT retry" in tool_message["content"]


def test_approval_session_grant_skips_second_ask():
    calls = []
    approvals = ToolApprovalRegistry()
    grants = SessionGrantRegistry()
    registry = _gated_registry(calls)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_1", name="documents.search", arguments={"query": "a"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_2", name="documents.search", arguments={"query": "b"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Both done.", finish_reason="stop"),
        ]
    )
    orchestrator = _approval_orchestrator(registry, approvals=approvals, grants=grants)
    _resolver(approvals, "session", [])

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="search twice",
            workspace_id=1,
            chat_id=7,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == [
        "proposed",
        "awaiting_approval",
        "running",
        "succeeded",
        # second call: session grant, no awaiting_approval round-trip
        "proposed",
        "running",
        "succeeded",
    ]
    assert calls == ["a", "b"]
    assert "documents.search" in grants.granted("workspace:1:chat:7")


def test_approval_always_persists_to_settings():
    calls = []
    persisted = []
    approvals = ToolApprovalRegistry()
    registry = _gated_registry(calls)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_1", name="documents.search", arguments={"query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Done.", finish_reason="stop"),
        ]
    )
    orchestrator = _approval_orchestrator(
        registry,
        approvals=approvals,
        persister=lambda user_id, tool_name: persisted.append((user_id, tool_name)),
    )
    _resolver(approvals, "always", [])

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="search",
            workspace_id=9,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert tool_states[-1].status == "succeeded"
    assert calls == ["q"]
    assert persisted == [(9, "documents.search")]


def test_non_interactive_runs_deny_gated_tools_immediately():
    calls = []
    registry = _gated_registry(calls)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="call_1", name="documents.search", arguments={"query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            ModelTurn(text="Skipped.", finish_reason="stop"),
        ]
    )
    orchestrator = _approval_orchestrator(registry, approvals=ToolApprovalRegistry())

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="search",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
            interactive=False,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == ["proposed", "failed"]
    assert calls == []
    tool_message = backend.requests[1]["messages"][-1]
    assert "unattended" in tool_message["content"]


def test_reused_model_call_id_needs_a_new_invocation_approval():
    calls = []
    approvals = ToolApprovalRegistry()
    orchestrator = _approval_orchestrator(_gated_registry(calls), approvals=approvals)
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[ToolCall(id="same", name="documents.search", arguments={"query": "a"})]
            ),
            ModelTurn(
                tool_calls=[ToolCall(id="same", name="documents.search", arguments={"query": "b"})]
            ),
            ModelTurn(text="done"),
        ]
    )
    stream = orchestrator.stream(
        backend=backend,
        prompt="search",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
    )
    approval_ids = []
    for event in stream:
        if event.event == "tool_call" and event.payload.status == "awaiting_approval":
            pending = approvals.pending()[0]
            approval_ids.append(pending.call_id)
            if len(approval_ids) == 2:
                assert not approvals.resolve(pending.run_id, "same", "approve", workspace_id=1)
            approvals.resolve(
                pending.run_id,
                pending.call_id,
                "approve" if len(approval_ids) == 1 else "deny",
                workspace_id=1,
            )
    assert len(set(approval_ids)) == 2
    assert calls == ["a"]


def test_denial_survives_output_budget_and_prevents_repeat_prompts():
    calls = []
    approvals = ToolApprovalRegistry()
    orchestrator = _approval_orchestrator(_gated_registry(calls), approvals=approvals)
    orchestrator.max_tool_result_chars_total = 0
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[ToolCall(id="a", name="documents.search", arguments={"query": "a"})]
            ),
            ModelTurn(
                tool_calls=[ToolCall(id="b", name="documents.search", arguments={"query": "b"})]
            ),
            ModelTurn(text="done"),
        ]
    )
    prompts = 0
    for event in orchestrator.stream(
        backend=backend,
        prompt="search",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
    ):
        if event.event == "tool_call" and event.payload.status == "awaiting_approval":
            prompts += 1
            pending = approvals.pending()[0]
            approvals.resolve(pending.run_id, pending.call_id, "deny", workspace_id=1)
    assert prompts == 1
    assert calls == []
    assert "NOT consented" in backend.requests[-1]["messages"][-1]["content"]


def test_failed_standing_grant_is_reported_as_once_only():
    calls = []
    approvals = ToolApprovalRegistry()

    def broken_persister(workspace_id, tool_name):
        raise RuntimeError("database unavailable")

    orchestrator = _approval_orchestrator(
        _gated_registry(calls),
        approvals=approvals,
        persister=broken_persister,
    )
    backend = ScriptedBackend(
        [
            ModelTurn(
                tool_calls=[ToolCall(id="a", name="documents.search", arguments={"query": "a"})]
            ),
            ModelTurn(text="done"),
        ]
    )
    events = []
    for event in orchestrator.stream(
        backend=backend,
        prompt="search",
        workspace_id=1,
        chat_id=7,
        config=ModelRequestConfig(),
        system_prompt=None,
    ):
        events.append(event)
        if event.event == "tool_call" and event.payload.status == "awaiting_approval":
            pending = approvals.pending()[0]
            approvals.resolve(pending.run_id, pending.call_id, "always", workspace_id=1)
    final = next(event.payload for event in events if event.event == "final")
    assert calls == ["a"]
    assert "Approved once" in final.tool_calls[0].result_summary
    assert orchestrator.grants.granted("workspace:1:chat:7") == frozenset()
