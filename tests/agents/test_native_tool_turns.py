import json
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


def test_openai_stream_reassembles_tool_arguments_and_sends_schema():
    provider_name = OnlineAgent._provider_tool_name("web.search")
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
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": ' news"}'}}]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        "data: [DONE]",
    ]
    agent = OnlineAgent(context(), "https://api.openai.com/v1", "gpt-test", api_key="key")
    agent.client.close()
    agent.client = OpenAIClient(lines)

    events = list(
        agent.stream_model_turn(
            [ChatMessage(role="user", content="What is the news today?")],
            [tool_definition()],
            ModelRequestConfig(max_tokens=256),
        )
    )

    turn = events[-1].turn
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

    def complete(self, system_prompt, user_prompt, generation_config):
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "local answer"},
        ]

    def complete_messages(self, messages, generation_config):
        self.messages = messages
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
