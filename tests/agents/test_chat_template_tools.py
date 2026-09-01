from __future__ import annotations

import pytest
from pydantic import BaseModel

from agents.architectures.chat_template_tools import (
    build_tool_payload,
    parse_tool_response,
    tokenizer_supports_tools,
)
from agents.models.tool_calling import ChatMessage, ToolCall, ToolDefinition, ToolExecutionOutput


class SearchArguments(BaseModel):
    query: str
    max_results: int = 3


def search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="web.search",
        description="Search the web",
        arguments_model=SearchArguments,
        handler=lambda _context, _arguments: ToolExecutionOutput(content="unused"),
    )


class ToolTokenizer:
    chat_template = "uses tools"

    def apply_chat_template(self, messages, **kwargs):
        del messages
        tools = kwargs.get("tools") or []
        return str(tools)


class IgnoringTokenizer(ToolTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "no catalog"


def test_capability_probe_requires_template_to_render_catalog():
    assert tokenizer_supports_tools(ToolTokenizer()) is True
    assert tokenizer_supports_tools(IgnoringTokenizer()) is False


def test_payload_maps_dotted_names_and_preserves_tool_round_trip():
    prior_call = ToolCall(id="call-1", name="web.search", arguments={"query": "news"})
    payload = build_tool_payload(
        [
            ChatMessage(role="user", content="Find news"),
            ChatMessage(role="assistant", content=None, tool_calls=[prior_call]),
            ChatMessage(
                role="tool",
                content='{"results": []}',
                tool_call_id="call-1",
                name="web.search",
            ),
        ],
        [search_tool()],
    )

    provider_name = payload.tools[0]["function"]["name"]
    assert provider_name != "web.search"
    assert payload.provider_to_internal[provider_name] == "web.search"
    assert payload.messages[1]["tool_calls"][0]["function"]["name"] == provider_name
    assert payload.messages[1]["tool_calls"][0]["function"]["arguments"] == {"query": "news"}
    assert payload.messages[2] == {
        "role": "tool",
        "content": '{"results": []}',
        "tool_call_id": "call-1",
    }


def test_qwen_tool_markup_parses_to_internal_call():
    payload = build_tool_payload([ChatMessage(role="user", content="news")], [search_tool()])
    provider_name = next(iter(payload.provider_to_internal))

    turn = parse_tool_response(
        (
            "<tool_call>\n"
            f'{{"name":"{provider_name}","arguments":{{"query":"celebrity news",'
            '"max_results":3}}\n'
            "</tool_call>"
        ),
        provider_to_internal=payload.provider_to_internal,
    )

    assert turn.finish_reason == "tool_calls"
    assert turn.text == ""
    assert turn.tool_calls[0].name == "web.search"
    assert turn.tool_calls[0].arguments == {"query": "celebrity news", "max_results": 3}


def test_bare_tool_json_accepts_parameters_alias():
    payload = build_tool_payload([ChatMessage(role="user", content="news")], [search_tool()])
    provider_name = next(iter(payload.provider_to_internal))

    turn = parse_tool_response(
        (f'{{"name":"{provider_name}","parameters":' '{"query":"celebrity news","max_results":3}}'),
        provider_to_internal=payload.provider_to_internal,
    )

    assert turn.finish_reason == "tool_calls"
    assert turn.text == ""
    assert turn.tool_calls[0].name == "web.search"
    assert turn.tool_calls[0].arguments == {"query": "celebrity news", "max_results": 3}


@pytest.mark.parametrize(
    "response,match",
    [
        ("<tool_call>{bad}</tool_call>", "invalid tool-call JSON"),
        (
            '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
            "unknown tool",
        ),
        ("<tool_call>{}", "incomplete tool-call markup"),
        (
            '<tool_call>{"name":"safe","arguments":[]}</tool_call>',
            "non-object tool arguments",
        ),
    ],
)
def test_malformed_tool_calls_fail_closed(response, match):
    with pytest.raises(ValueError, match=match):
        parse_tool_response(response, provider_to_internal={"safe": "web.search"})
