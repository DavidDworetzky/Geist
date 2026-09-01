from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agents.models.chat_result import WorkArtifact
from agents.models.tool_calling import (
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolCall,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.chat_orchestrator import ChatOrchestrator, RunControlRegistry
from app.services.tool_intent_router import ToolIntentRouter
from app.services.tool_registry import ToolRegistry


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


class FailingIntentRouter(ToolIntentRouter):
    def classify(self, backend, messages):
        del backend, messages
        raise RuntimeError("classifier unavailable")


def test_classifier_failure_falls_back_to_direct_answer_without_tools():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="public.search",
            description="Public search",
            arguments_model=LookupArguments,
            handler=lambda _context, _arguments: ToolExecutionOutput(content="unused"),
            semantic_tags=frozenset({"public_retrieval"}),
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
        )
    )

    assert backend.requests[0]["tools"] == []
    assert next(event.payload for event in events if event.event == "final").message == [
        "Fallback answer"
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
        )
    )

    assert backend.requests[0]["tools"] == []
    assert backend.requests[1]["tools"] == expected_tools
    assert [event.payload["text"] for event in events if event.event == "delta"] == [
        "Direct response"
    ]


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
    assert backend.requests[1]["messages"][-1] == {
        "role": "tool",
        "content": '{"answer": "2023-tax-return.pdf"}',
        "tool_call_id": "call_1",
        "name": "documents.search",
    }
    assert [event.payload.status for event in events if event.event == "tool_call"] == [
        "proposed",
        "running",
        "succeeded",
    ]
    assert len(writes) == 1
    assert writes[0]["transcript"][1]["tool_calls"][0]["id"] == "call_1"
    completion = next(event.payload for event in events if event.event == "final")
    assert completion.chat_id == 42
    assert completion.tool_calls[0].id == "call_1"
    assert completion.message == ["I found 2023-tax-return.pdf."]


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


def test_backend_without_native_tools_receives_empty_registry():
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

    list(
        orchestrator.stream(
            backend=backend,
            prompt="news",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )

    assert backend.requests[0]["tools"] == []


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
    assert "aggregate budget exhausted" in tool_message["content"]


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
