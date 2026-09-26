import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from agents.local_agent import LocalAgent
from agents.models.tool_calling import (
    ChatMessage,
    ModelRequestConfig,
    ToolCall,
    ToolDefinition,
    ToolExecutionOutput,
)
from agents.online_agent import NativeProviderError, OnlineAgent
from app.services.chat_orchestrator import ChatOrchestrator
from app.services.tool_registry import ToolRegistry


class SearchArguments(BaseModel):
    query: str


def context():
    return AgentContext(
        AgentSettings(
            name="test",
            version="1",
            description="test",
            max_tokens=128,
            n=1,
            temperature=0.2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
    )


def tool_definition():
    return ToolDefinition(
        name="web.search",
        description="Search current news",
        arguments_model=SearchArguments,
        handler=lambda tool_context, arguments: ToolExecutionOutput(content="unused"),
    )


class FakeStreamResponse:
    status_code = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def iter_lines(self):
        return iter(self.lines)


class FailedStreamResponse(FakeStreamResponse):
    status_code = 503
    text = "temporarily unavailable"

    def __init__(self):
        super().__init__([])

    def read(self):
        return self.text.encode()


class OpenAIClient:
    def __init__(self, lines):
        self.lines = lines
        self.request = None

    def stream(self, method, url, **kwargs):
        self.request = {"method": method, "url": url, **kwargs}
        return FakeStreamResponse(self.lines)

    def close(self):
        pass


class SequencedOpenAIClient(OpenAIClient):
    def __init__(self, responses):
        super().__init__([])
        self.responses = iter(responses)
        self.requests = []

    def stream(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return next(self.responses)


def test_opus_reasoning_survives_tool_continuation_history_and_checkpoint():
    model = "anthropic/claude-opus-5.5"
    details = [
        {
            "type": "reasoning.text",
            "text": "Private ",
            "index": 0,
            "id": "reasoning-1",
            "format": "anthropic-claude-v1",
            "signature": None,
        },
        {"type": "reasoning.text", "text": "thought", "index": 0},
        {"type": "reasoning.text", "text": "", "signature": "signed-state", "index": 0},
        {
            "type": "reasoning.encrypted",
            "data": "opaque-state",
            "index": 1,
            "id": "reasoning-2",
            "format": "anthropic-claude-v1",
        },
    ]
    provider_name = OnlineAgent._provider_tool_name("web.search")
    chunks = [
        {"reasoning_details": [details[0]]},
        {"reasoning_details": details[1:3]},
        {
            "reasoning_details": [details[3]],
            "tool_calls": [
                {
                    "index": 0,
                    "id": "provider-call",
                    "function": {"name": provider_name, "arguments": '{"query":"news"}'},
                }
            ],
        },
    ]
    first = FakeStreamResponse(
        [
            *("data: " + json.dumps({"choices": [{"delta": delta}]}) for delta in chunks),
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
    )
    answer = FakeStreamResponse(
        [
            'data: {"choices":[{"delta":{"content":"Found it"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    agent = OnlineAgent(context(), "https://openrouter.ai/api/v1", model, api_key="test")
    agent.client.close()
    client = SequencedOpenAIClient([first, answer, answer])
    agent.client = client
    writes = []
    calls = []

    def lookup(tool_context, arguments):
        calls.append(arguments.query)
        return ToolExecutionOutput(content="Search result")

    definition = tool_definition()
    definition.handler = lookup
    registry = ToolRegistry()
    registry.register(definition)

    def write_history(**kwargs):
        writes.append(kwargs)
        return SimpleNamespace(chat_session_id=42)

    orchestrator = ChatOrchestrator(registry, history_writer=write_history)
    events = list(
        orchestrator.stream(
            backend=agent,
            prompt="Find news",
            workspace_id=1,
            chat_id=None,
            config=ModelRequestConfig(),
            system_prompt=None,
        )
    )
    assert calls == ["news"]
    continuation = client.requests[1]["json"]["messages"]
    assistant = next(message for message in continuation if message["role"] == "assistant")
    assert assistant["reasoning_details"] == details
    assert "reasoning_model" not in assistant
    assert "reasoning_endpoint" not in assistant
    assert assistant["content"] is None
    assert continuation[-1]["content"] == "Search result"
    assert continuation[-1]["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert continuation[-1]["tool_call_id"] != "provider-call"
    assert [e.payload["text"] for e in events if e.event == "delta"] == ["Found it"]
    assert next(e.payload for e in events if e.event == "final").message == ["Found it"]

    saved = json.loads(json.dumps(writes[0]["transcript"]))
    restored = orchestrator._history_messages([{"transcript": saved, "status": "completed"}])
    checkpoint = orchestrator._checkpoint_messages(restored)
    restored = [orchestrator._checkpoint_message(item) for item in checkpoint]
    restored.append(ChatMessage(role="user", content="Continue"))
    list(agent.stream_model_turn(restored, [definition], ModelRequestConfig()))
    replayed = next(m for m in client.requests[2]["json"]["messages"] if m["role"] == "assistant")
    assert replayed["reasoning_details"] == details


@pytest.mark.parametrize("target_model", [None, "other-model", "anthropic/claude-opus-5.5"])
@pytest.mark.parametrize(
    "target_endpoint",
    [
        None,
        "https://backup.example/v1/chat/completions",
        "https://openrouter.ai/api/v1/chat/completions",
    ],
)
def test_reasoning_replay_is_source_scoped_and_does_not_mutate_history(target_model, target_endpoint):
    model = "anthropic/claude-opus-5.5"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    saved = {
        "role": "assistant",
        "content": None,
        "reasoning_model": model,
        "reasoning_endpoint": endpoint,
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
    }
    message = ChatMessage.from_dict(saved)
    payload = message.to_openai(reasoning_model=target_model, reasoning_endpoint=target_endpoint)
    if target_model == model and target_endpoint == endpoint:
        assert payload["reasoning_details"] == saved["reasoning_details"]
        payload["reasoning_details"][0]["data"] = "modified"
    else:
        assert "reasoning_details" not in payload
    serialized = message.to_dict()
    serialized["reasoning_details"][0]["data"] = "modified"
    assert message.reasoning_details == saved["reasoning_details"]
    message.reasoning_details[0]["data"] = "changed in memory"
    assert saved["reasoning_details"][0]["data"] == "opaque"


def test_legacy_and_non_assistant_messages_do_not_send_reasoning():
    legacy = ChatMessage.from_dict({"role": "assistant", "content": "Old answer"})
    assert legacy.to_openai(reasoning_model="opus") == legacy.to_dict()
    for role in ("user", "tool", "system"):
        message = ChatMessage(
            role=role,
            content="text",
            reasoning_model="opus",
            reasoning_details=[{"type": "reasoning.text", "text": "hidden"}],
        )
        assert "reasoning_details" not in message.to_openai(reasoning_model="opus")
        assert "reasoning_details" not in message.to_dict()


def test_reasoning_from_interrupted_attempt_is_not_replayed():
    class InterruptedResponse(FakeStreamResponse):
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"discard"}]}}]}'
            raise httpx.ReadError("connection dropped")

    success = FakeStreamResponse(
        [
            'data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    agent = OnlineAgent(
        context(),
        "https://openrouter.ai/api/v1",
        "anthropic/claude-opus-5.5",
        api_key="test",
        max_retries=2,
    )
    agent.client.close()
    client = SequencedOpenAIClient([InterruptedResponse([]), success])
    agent.client = client
    events = list(
        agent.stream_model_turn(
            [ChatMessage(role="user", content="Hello")], [], ModelRequestConfig()
        )
    )
    assert len(client.requests) == 2
    assert events[-1].turn.reasoning_details == []
    assert events[-1].turn.reasoning_model is None
    assert events[-1].turn.reasoning_endpoint is None


def _run_streamed_tool_call(
    second_id=None, *, model="gpt-test", endpoint="https://api.openai.com/v1"
):
    provider_name = OnlineAgent._provider_tool_name("web.search")
    second_tool_call = {
        "index": 0,
        "function": {"arguments": ' news"}'},
    }
    if second_id is not None:
        second_tool_call["id"] = second_id
    lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": provider_name,
                                        "arguments": '{"query":"today',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [second_tool_call]},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        "data: [DONE]",
    ]
    agent = OnlineAgent(context(), endpoint, model, api_key="key")
    agent.client.close()
    agent.client = OpenAIClient(lines)

    events = list(
        agent.stream_model_turn(
            [ChatMessage(role="user", content="What is the news today?")],
            [tool_definition()],
            ModelRequestConfig(max_tokens=256),
        )
    )
    return events[-1].turn, agent, provider_name


def test_openai_stream_reassembles_tool_arguments_and_sends_schema():
    turn, agent, provider_name = _run_streamed_tool_call()

    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "web.search"
    assert turn.tool_calls[0].arguments == {"query": "today news"}
    assert turn.finish_reason == "tool_calls"
    assert agent.client.request["url"] == "https://api.openai.com/v1/chat/completions"
    payload = agent.client.request["json"]
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == provider_name
    assert provider_name != "web.search"
    assert payload["tools"][0]["function"]["parameters"]["required"] == ["query"]


def test_union_alpha_stream_filters_defaults_and_preserves_automatic_tool_calling():
    turn, agent, provider_name = _run_streamed_tool_call(
        model="stealth/union-alpha", endpoint="https://openrouter.ai/api/v1"
    )

    assert turn.tool_calls[0].name == "web.search"
    assert turn.tool_calls[0].arguments == {"query": "today news"}
    assert turn.finish_reason == "tool_calls"
    assert agent.client.request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    payload = agent.client.request["json"]
    assert payload["model"] == "stealth/union-alpha"
    assert payload["stream"] is True
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == provider_name
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0
    assert not {"n", "frequency_penalty", "presence_penalty", "stop", "reasoning"} & payload.keys()


def test_openai_stream_accepts_repeated_tool_call_id():
    turn, _agent, _provider_name = _run_streamed_tool_call("call_1")

    assert turn.tool_calls[0].id == "call_1"


def test_openai_stream_keeps_first_conflicting_tool_call_id(caplog):
    with caplog.at_level("WARNING", logger="agents.online_agent"):
        turn, _agent, _provider_name = _run_streamed_tool_call("call_2")

    assert turn.tool_calls[0].id == "call_1"
    assert "Ignoring conflicting tool-call ID" in caplog.text


def test_native_turn_retries_before_any_text_is_emitted():
    success = FakeStreamResponse(
        [
            'data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    client = SequencedOpenAIClient([FailedStreamResponse(), success])
    agent = OnlineAgent(
        context(),
        "https://api.openai.com/v1",
        "gpt-test",
        api_key="key",
        max_retries=2,
    )
    agent.client.close()
    agent.client = client

    events = list(
        agent.stream_model_turn(
            [ChatMessage(role="user", content="hello")],
            [tool_definition()],
            ModelRequestConfig(),
        )
    )

    assert len(client.requests) == 2
    assert events[-1].turn.text == "recovered"


def test_native_turn_fails_over_before_any_text_is_emitted():
    primary_client = SequencedOpenAIClient([FailedStreamResponse()])
    backup_client = OpenAIClient(
        [
            'data: {"choices":[{"delta":{"content":"backup"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    agent = OnlineAgent(
        context(),
        "https://api.openai.com/v1",
        "gpt-test",
        api_key="key",
        max_retries=1,
        backup_providers=[
            {
                "base_url": "https://backup.example/v1",
                "model": "backup-model",
                "api_key": "backup-key",
                "priority": 1,
                "supports_native_tool_calling": True,
            }
        ],
    )
    agent.client.close()
    agent.client = primary_client

    with patch("agents.online_agent.httpx.Client", return_value=backup_client):
        events = list(
            agent.stream_model_turn(
                [ChatMessage(role="user", content="hello")],
                [tool_definition()],
                ModelRequestConfig(),
            )
        )

    assert events[-1].turn.text == "backup"
    assert backup_client.request["url"] == "https://backup.example/v1/chat/completions"
    assert backup_client.request["json"]["model"] == "backup-model"


def test_native_turn_does_not_retry_after_emitting_text():
    class InterruptedResponse(FakeStreamResponse):
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'
            raise httpx.ReadError("connection dropped")

    client = SequencedOpenAIClient([InterruptedResponse([]), FailedStreamResponse()])
    agent = OnlineAgent(
        context(),
        "https://api.openai.com/v1",
        "gpt-test",
        api_key="key",
        max_retries=2,
    )
    agent.client.close()
    agent.client = client

    with pytest.raises(httpx.ReadError, match="connection dropped"):
        list(
            agent.stream_model_turn(
                [ChatMessage(role="user", content="hello")],
                [tool_definition()],
                ModelRequestConfig(),
            )
        )

    assert len(client.requests) == 1


def test_native_turn_does_not_retry_non_transient_http_errors():
    unauthorized = FailedStreamResponse()
    unauthorized.status_code = 401
    client = SequencedOpenAIClient([unauthorized, FailedStreamResponse()])
    agent = OnlineAgent(
        context(),
        "https://api.openai.com/v1",
        "gpt-test",
        api_key="key",
        max_retries=2,
    )
    agent.client.close()
    agent.client = client

    with pytest.raises(NativeProviderError, match="status 401"):
        list(
            agent.stream_model_turn(
                [ChatMessage(role="user", content="hello")],
                [tool_definition()],
                ModelRequestConfig(),
            )
        )

    assert len(client.requests) == 1


class AnthropicResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": OnlineAgent._provider_tool_name("web.search"),
                    "input": {"query": "today news"},
                }
            ],
            "stop_reason": "tool_use",
        }


class AnthropicClient:
    def __init__(self):
        self.request = None

    def post(self, url, **kwargs):
        self.request = {"url": url, **kwargs}
        return AnthropicResponse()

    def close(self):
        pass


def test_anthropic_tool_use_is_normalized():
    agent = OnlineAgent(context(), "https://api.anthropic.com/v1", "claude-test", api_key="key")
    agent.client.close()
    agent.client = AnthropicClient()

    events = list(
        agent.stream_model_turn(
            [
                ChatMessage(role="system", content="Be useful"),
                ChatMessage(role="user", content="News?"),
            ],
            [tool_definition()],
            ModelRequestConfig(),
        )
    )

    turn = events[-1].turn
    assert turn.tool_calls[0].name == "web.search"
    assert turn.tool_calls[0].arguments == {"query": "today news"}
    assert agent.client.request["url"] == "https://api.anthropic.com/v1/messages"
    assert agent.client.request["headers"]["x-api-key"] == "key"
    assert agent.client.request["json"]["system"] == "Be useful"
    assert agent.client.request["json"]["tools"][0]["name"] == OnlineAgent._provider_tool_name(
        "web.search"
    )


def test_provider_name_codec_applies_to_followup_assistant_calls():
    tool = tool_definition()
    internal_to_provider, provider_to_internal = OnlineAgent._provider_tool_name_maps([tool])
    provider_name = internal_to_provider[tool.name]

    message = ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(id="call_1", name=tool.name, arguments={"query": "news"})],
    )

    assert provider_to_internal[provider_name] == "web.search"
    assert (
        message.to_openai(internal_to_provider)["tool_calls"][0]["function"]["name"]
        == provider_name
    )


def test_provider_name_codec_applies_to_historical_tool_results():
    result_message = ChatMessage(
        role="tool",
        name="retired.tool",
        tool_call_id="call_old",
        content="old result",
    )
    internal_to_provider, _provider_to_internal = OnlineAgent._provider_tool_name_maps(
        [], [result_message]
    )

    provider_message = result_message.to_openai(internal_to_provider)

    assert internal_to_provider["retired.tool"] == OnlineAgent._provider_tool_name("retired.tool")
    assert "name" not in provider_message
    assert provider_message == {
        "role": "tool",
        "content": "old result",
        "tool_call_id": "call_old",
    }


def test_native_tool_capability_is_known_provider_or_explicit_override():
    custom = OnlineAgent(context(), "https://example.test/v1", "custom", api_key="key")
    forced = OnlineAgent(
        context(),
        "https://example.test/v1",
        "custom",
        api_key="key",
        supports_native_tool_calling=True,
    )
    openai = OnlineAgent(context(), "https://api.openai.com/v1", "gpt-test", api_key="key")

    try:
        assert custom.supports_native_tool_calling is False
        assert forced.supports_native_tool_calling is True
        assert openai.supports_native_tool_calling is True
    finally:
        custom.client.close()
        forced.client.close()
        openai.client.close()


class LocalRunner:
    def __init__(self):
        self.messages = None
        self.generation_config = None

    def complete(self, system_prompt, user_prompt, generation_config):
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "local answer"},
        ]

    def complete_messages(self, messages, generation_config):
        self.messages = messages
        self.generation_config = generation_config
        return [
            {"role": "user", "content": messages[-1]["content"]},
            {"role": "assistant", "content": "local answer"},
        ]


class UnsupportedNativeStreamLocalRunner(LocalRunner):
    def __init__(self):
        super().__init__()
        self.stream_called = False

    def stream_model_turn(self, messages, tools, config):
        self.stream_called = True
        yield from ()


def local_agent_without_loading_model():
    agent = LocalAgent.__new__(LocalAgent)
    agent.runner_type = "test"
    agent.runner = LocalRunner()
    return agent


def test_local_runner_fails_closed_when_given_tool_schemas():
    agent = local_agent_without_loading_model()
    with pytest.raises(ValueError, match="does not support native tool calling"):
        list(
            agent.stream_model_turn(
                [ChatMessage(role="user", content="news")],
                [tool_definition()],
                ModelRequestConfig(),
            )
        )


def test_local_runner_fails_closed_before_calling_unsupported_native_stream():
    agent = LocalAgent.__new__(LocalAgent)
    agent.runner_type = "test"
    agent.runner = UnsupportedNativeStreamLocalRunner()

    with pytest.raises(ValueError, match="does not support native tool calling"):
        list(
            agent.stream_model_turn(
                [ChatMessage(role="user", content="news")],
                [tool_definition()],
                ModelRequestConfig(),
            )
        )

    assert agent.runner.stream_called is False


def test_local_runner_can_complete_persistence_free_without_tools():
    agent = local_agent_without_loading_model()
    events = list(
        agent.stream_model_turn(
            [ChatMessage(role="user", content="hello")],
            [],
            ModelRequestConfig(),
        )
    )
    assert events[0].text == "local answer"
    assert events[-1].turn.text == "local answer"


def test_local_runner_preserves_structured_conversation_roles():
    agent = local_agent_without_loading_model()
    messages = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="Remember cobalt."),
        ChatMessage(role="assistant", content="I will remember cobalt."),
        ChatMessage(role="user", content="What should you remember?"),
    ]

    list(agent.stream_model_turn(messages, [], ModelRequestConfig()))

    assert agent.runner.messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Remember cobalt."},
        {"role": "assistant", "content": "I will remember cobalt."},
        {"role": "user", "content": "What should you remember?"},
    ]


def test_local_runner_forwards_all_stop_sequences():
    agent = local_agent_without_loading_model()

    list(
        agent.stream_model_turn(
            [ChatMessage(role="user", content="hello")],
            [],
            ModelRequestConfig(stop=["STOP", "END"]),
        )
    )

    assert agent.runner.generation_config.stop == ["STOP", "END"]
