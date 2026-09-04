from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agents.models.chat_result import WorkArtifact
from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolCall,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.agent_permissions import AgentPermissions
from app.services.chat_orchestrator import ChatOrchestrator, RunControlRegistry
from app.services.goal_runtime import GoalRuntimeRegistry
from app.services.tool_approvals import SessionGrantRegistry, ToolApprovalRegistry
from app.services.tool_registry import ToolRegistry, build_default_tool_registry


class LookupArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


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
        if turn.text:
            yield ModelEvent.text_delta(turn.text)
        yield ModelEvent.turn_complete(turn)


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


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///tmp/secret", "not-a-url"])
def test_artifact_urls_reject_unsafe_schemes(url):
    with pytest.raises(ValueError, match="HTTP"):
        WorkArtifact.from_url(url, kind="text", mime_type="text/plain")


def test_tool_result_reenters_model_context_and_turn_persists_once():
    calls = []

    def lookup(context, arguments):
        calls.append((context.user_id, arguments.query))
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
            user_id=7,
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
            user_id=7,
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
            ModelTurn(text='{"tasks":[{"title":"Finish work"}]}'),
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
            user_id=1,
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
    assert any(
        message["role"] == "user" and "Continue the objective" in message["content"]
        for message in backend.requests[-1]["messages"]
    )


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
        user_id=1,
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
        user_id=1,
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
        user_id=1,
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
            assert orchestrator.approvals.resolve(run_id, event.payload.id, "approve")
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
    controls.start("run", user_id=1)
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
            self.resolve(pending.run_id, pending.call_id, "approve")
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
            user_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    assert executed == ["first"]
    assert approvals.requested == 1
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
            user_id=1,
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
            user_id=1,
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
        user_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )

    started = next(stream)
    assert controls.cancel(started.payload["run_id"])
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
        user_id=1,
        chat_id=None,
        config=ModelRequestConfig(),
        system_prompt=None,
    )

    started = next(stream)
    run_id = started.payload["run_id"]
    assert controls.cancel(run_id)
    assert len(writes) == 1
    assert writes[0]["status"] == "cancelled"
    assert writes[0]["run_id"] == run_id

    # Mirrors the frontend aborting its SSE request immediately after the
    # cancellation endpoint acknowledges durable cancellation.
    stream.close()

    assert len(writes) == 1
    assert not controls.cancel(run_id)


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
            user_id=1,
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
            user_id=1,
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
        user_id=1,
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
        user_id=1,
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
        user_id=1,
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
        user_id=1,
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
            user_id=1,
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
        permissions_loader=lambda user_id: AgentPermissions(mode="auto_approve"),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="send it",
            user_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == ["proposed", "running", "succeeded"]
    assert all(state.requires_approval is False for state in tool_states)


def test_require_approval_permissions_gate_read_only_tool():
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
        permissions_loader=lambda user_id: AgentPermissions(mode="require_approval"),
        approvals=ToolApprovalRegistry(),
        grants=SessionGrantRegistry(),
        approval_timeout_seconds=0.05,
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="search",
            user_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    # No decision arrives, so the approval times out and fails closed.
    tool_states = [event.payload for event in events if event.event == "tool_call"]
    assert [state.status for state in tool_states] == [
        "proposed",
        "awaiting_approval",
        "failed",
    ]
    assert tool_states[-1].error == "approval_denied"


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
        permissions_loader=lambda user_id: AgentPermissions(
            mode="require_approval",
            always_allow=frozenset({"workspace.write_markdown"}),
        ),
    )

    events = list(
        orchestrator.stream(
            backend=backend,
            prompt="write",
            user_id=1,
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
                approvals.resolve(pending.run_id, pending.call_id, decision)
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
            user_id=1,
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
            user_id=1,
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
            user_id=1,
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
            user_id=1,
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
    assert "documents.search" in grants.granted("chat:7")


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
            user_id=9,
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
            user_id=1,
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
