from __future__ import annotations

import pytest
from pydantic import BaseModel

from agents.architectures.chat_template_tools import (
    ToolResponseStream,
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


def test_unknown_bare_json_descriptor_remains_text():
    response = '{"name":"summarize","parameters":{"text":"hello"}}'

    turn = parse_tool_response(response, provider_to_internal={"safe": "web.search"})

    assert turn.finish_reason == "stop"
    assert turn.text == response
    assert turn.tool_calls == []


def test_bare_tool_json_prefers_arguments_over_parameters():
    turn = parse_tool_response(
        '{"name":"safe","arguments":{"query":"new"},"parameters":{"query":"old"}}',
        provider_to_internal={"safe": "web.search"},
    )

    assert turn.tool_calls[0].arguments == {"query": "new"}


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


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 13, 10000])
@pytest.mark.parametrize(
    "response",
    [
        "  Ordinary prose with spaces and\nnewlines.  ",
        "```python\ndef compare(a, b):\n    return a < b\n```",
        "Literal partial tags: <tool_ and </tool_ end.",
        "Ends with a prefix <tool_call",
        '<tool_call>{"name":"safe","arguments":{}}</tool_call>',
        'Before <tool_call>{"name":"safe","arguments":{}}</tool_call> after.',
        '<tool_call>{"name":"safe","arguments":{}}</tool_call>\n'
        '<tool_call>{"name":"safe","parameters":{}}</tool_call> final text.',
        '  {"name":"safe","parameters":{"query":"hello"}}  ',
        '{"name":"ordinary data","value":3}',
        '{"value":3} followed by plain text',
        "   ",
    ],
)
def test_incremental_tool_text_matches_complete_parser(response, chunk_size):
    names = {"safe": "web.search"}
    parser = ToolResponseStream(names)
    deltas = [
        parser.feed(response[i : i + chunk_size]) for i in range(0, len(response), chunk_size)
    ]
    tail, turn = parser.finish()
    expected = parse_tool_response(response, provider_to_internal=names)

    assert "".join(deltas) + tail == expected.text == turn.text
    assert [(call.name, call.arguments) for call in turn.tool_calls] == [
        (call.name, call.arguments) for call in expected.tool_calls
    ]
    assert turn.finish_reason == expected.finish_reason


def test_incremental_tool_text_streams_before_and_after_a_call():
    parser = ToolResponseStream({"safe": "web.search"})
    assert parser.feed("First sentence. ") == "First sentence."
    assert parser.feed("<tool_") == ""
    assert parser.feed('call>{"name":"safe","arguments":{"query":"private payload"}}') == ""
    assert parser.feed("</tool_") == ""
    assert parser.feed("call> Second sentence.") == "  Second sentence."
    tail, turn = parser.finish()
    assert tail == ""
    assert turn.text == "First sentence.  Second sentence."
    assert turn.tool_calls[0].arguments == {"query": "private payload"}


@pytest.mark.parametrize(
    "response",
    [
        '<tool_call>{"name":"safe","arguments":{}}',
        '<tool_call>{"name":"missing","arguments":{}}</tool_call>',
        '<tool_call>{"name":"safe","arguments":[]}</tool_call>',
        "<tool_call>{bad}</tool_call>",
        "</tool_call>",
        '<tool_call>{"name":"safe","arguments":{}}</tool_call></tool_call>',
    ],
)
def test_incremental_malformed_tools_fail_without_exposing_payload(response):
    parser = ToolResponseStream({"safe": "web.search"})
    visible = []
    with pytest.raises(ValueError):
        for char in response:
            visible.append(parser.feed(char))
        parser.finish()
    assert "".join(visible) == ""


def test_leading_json_is_held_until_tool_or_text_is_known():
    parser = ToolResponseStream({"safe": "web.search"})
    assert parser.feed(" \n{") == ""
    assert parser.feed('"name":"safe","arguments":{}}') == ""
    tail, turn = parser.finish()
    assert tail == ""
    assert turn.tool_calls[0].name == "web.search"


def test_tool_markup_lookahead_does_not_grow_with_plain_text():
    parser = ToolResponseStream({"safe": "web.search"})
    for _ in range(1000):
        assert parser.feed("plain text ")
        assert len(parser._pending) < len("</tool_call>")
    tail, turn = parser.finish()
    assert tail == ""
    assert turn.text == ("plain text " * 1000).strip()


def function_call(parameters: str, name: str = "safe") -> str:
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def xml_tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "safe",
                "parameters": {
                    "type": "object",
                    "$defs": {"Text": {"type": "string"}},
                    "properties": {
                        "query": {"$ref": "#/$defs/Text"},
                        "max_results": {"type": "integer"},
                        "include_images": {"type": "boolean"},
                        "options": {"type": "object"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "recency": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                },
            },
        }
    ]


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 13, 10000])
def test_qwen_xml_call_streams_prose_and_preserves_typed_arguments(chunk_size):
    response = "Searching now. " + function_call(
        "<parameter=query>\nrecent celebrity headlines\n</parameter>\n"
        "<parameter=max_results>\n3\n</parameter>\n"
        "<parameter=include_images>false</parameter>\n"
        '<parameter=options>{"region":"us"}</parameter>\n'
        '<parameter=tags>["news","recent"]</parameter>\n'
        "<parameter=recency>null</parameter>\n"
    )
    parser = ToolResponseStream({"safe": "web.search"}, xml_tool_schema())
    visible = [
        parser.feed(response[i : i + chunk_size]) for i in range(0, len(response), chunk_size)
    ]
    tail, turn = parser.finish()
    assert "".join(visible) + tail == turn.text == "Searching now."
    assert turn.finish_reason == "tool_calls"
    assert turn.tool_calls[0].name == "web.search"
    assert turn.tool_calls[0].arguments == {
        "query": "recent celebrity headlines",
        "max_results": 3,
        "include_images": False,
        "options": {"region": "us"},
        "tags": ["news", "recent"],
        "recency": None,
    }


@pytest.mark.parametrize(
    "value",
    ["123", "true", "null", '{"x":1}', '"quoted"', "", " \n  indented\ntext & <literal>  \n "],
)
def test_qwen_xml_string_parameters_are_not_coerced_or_stripped(value):
    response = function_call(f"<parameter=query>\n{value}\n</parameter>\n")
    turn = parse_tool_response(
        response, provider_to_internal={"safe": "web.search"}, tools=xml_tool_schema()
    )
    assert turn.tool_calls[0].arguments == {"query": value}


@pytest.mark.parametrize(
    "response",
    [
        function_call("<parameter=query>x</parameter>\n<parameter=query>y</parameter>"),
        function_call("<parameter=query>x"),
        function_call("garbage<parameter=query>x</parameter>"),
        function_call("<parameter=query>x</parameter>garbage"),
        function_call("<parameter=max_results>not a number</parameter>"),
        function_call("<parameter=options>{bad}</parameter>"),
        function_call("<parameter=query>x</parameter>", name="unknown"),
        "<tool_call><function=safe></tool_call>",
        "<tool_call><function=safe></function><function=safe></function></tool_call>",
    ],
)
def test_malformed_xml_call_fails_closed_without_exposing_arguments(response):
    parser = ToolResponseStream({"safe": "web.search"}, xml_tool_schema())
    visible = []
    with pytest.raises(ValueError):
        for char in response:
            visible.append(parser.feed(char))
        parser.finish()
    assert "".join(visible) == ""


def test_json_and_xml_tool_calls_can_coexist_in_one_turn():
    response = function_call("<parameter=query>news</parameter>") + (
        '<tool_call>{"name":"safe","arguments":{"query":"more news"}}</tool_call>'
    )
    turn = parse_tool_response(
        response, provider_to_internal={"safe": "web.search"}, tools=xml_tool_schema()
    )
    assert [call.arguments["query"] for call in turn.tool_calls] == ["news", "more news"]
