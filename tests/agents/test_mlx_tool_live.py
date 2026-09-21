"""Opt-in model-native tool protocol qualification using installed MLX weights."""

import os
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel

from agents.architectures.chat_template_tools import build_tool_payload, parse_tool_response
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.models.tool_calling import (
    ChatMessage,
    ModelRequestConfig,
    ToolCall,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.local_models import get_local_model_manager


class LookupArguments(BaseModel):
    query: str


def lookup_tool() -> ToolDefinition:
    return ToolDefinition(
        name="lookup.fact",
        description="Look up a code by its query. Use the returned code in your answer.",
        arguments_model=LookupArguments,
        handler=lambda _context, _arguments: ToolExecutionOutput(content="KITE-739"),
    )


@pytest.fixture(scope="module")
def runner() -> Iterator[MLXLlamaRunner]:
    if os.getenv("GEIST_RUN_MLX_TOOL_SMOKE") != "1":
        pytest.skip("set GEIST_RUN_MLX_TOOL_SMOKE=1 to qualify an installed MLX model's tools")
    manager = get_local_model_manager()
    artifact = manager.get_artifact(os.getenv("GEIST_MLX_TOOL_ARTIFACT_ID", "qwen3.8-27b-4bit-mlx"))
    status = manager.status(artifact.id)
    assert status["supported"] is True
    assert status["status"] == "installed", "Install the model before running this smoke test"
    instance = MLXLlamaRunner()
    instance.load(artifact.model_id, {"weights_dir": status["path"], "implementation": "mlx_lm"})
    try:
        yield instance
    finally:
        instance.cleanup()


@pytest.mark.live_model
def test_installed_tokenizer_tool_history_round_trips_through_parser(runner):
    expected = ToolCall(id="prior-call", name="lookup.fact", arguments={"query": "123"})
    payload = build_tool_payload(
        [
            ChatMessage(role="user", content="Look this up"),
            ChatMessage(role="assistant", tool_calls=[expected]),
        ],
        [lookup_tool()],
    )
    rendered = runner.llama.tokenizer.apply_chat_template(
        payload.messages,
        tools=payload.tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # The last block is the actual assistant call, not the template's example.
    calls = re.findall(r"<tool_call>.*?</tool_call>", rendered, re.DOTALL)
    assert calls, "The installed template does not use a qualified tagged tool protocol"
    turn = parse_tool_response(
        calls[-1], provider_to_internal=payload.provider_to_internal, tools=payload.tools
    )
    assert [(call.name, call.arguments) for call in turn.tool_calls] == [
        (expected.name, expected.arguments)
    ]


@pytest.mark.live_model
def test_real_mlx_calls_a_tool_and_uses_its_result(runner):
    tool = lookup_tool()
    messages = [
        ChatMessage(
            role="system", content="Use the available tool when asked. Do not invent tool results."
        ),
        ChatMessage(
            role="user",
            content="Call lookup.fact with query 123 as a string, then report only the code it returns.",
        ),
    ]
    config = ModelRequestConfig(max_tokens=192, temperature=0.0)
    first = list(runner.stream_model_turn(messages, [tool], config))[-1].turn
    assert first is not None
    assert len(first.tool_calls) == 1
    call = first.tool_calls[0]
    assert call.name == tool.name
    assert LookupArguments.model_validate(call.arguments).query == "123"
    messages.extend(
        [
            ChatMessage(role="assistant", content=first.text, tool_calls=first.tool_calls),
            ChatMessage(role="tool", content="KITE-739", name=call.name, tool_call_id=call.id),
        ]
    )
    second = list(runner.stream_model_turn(messages, [tool], config))[-1].turn
    assert second is not None
    assert not second.tool_calls
    assert "KITE-739" in second.text


@pytest.mark.live_model
def test_real_mlx_stream_resumes_and_closes_across_workers_then_generates_again(runner):
    messages = [ChatMessage(role="user", content="Write a long explanation of how rain forms.")]
    stream = runner.stream_model_turn(
        messages, [], ModelRequestConfig(max_tokens=64, temperature=0.0)
    )
    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        assert first.submit(next, stream).result(timeout=60).kind == "text_delta"
        assert second.submit(next, stream).result(timeout=60).kind == "text_delta"
        second.submit(stream.close).result(timeout=60)
    following = list(
        runner.stream_model_turn(
            [ChatMessage(role="user", content="Say hello in one sentence.")],
            [],
            ModelRequestConfig(max_tokens=24, temperature=0.0),
        )
    )
    assert following[-1].kind == "turn_complete"
    assert following[-1].turn.text
