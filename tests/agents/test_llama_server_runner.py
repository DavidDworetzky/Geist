"""OpenAI-wire contract tests for the managed llama-server runner."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.architectures.base_runner import GenerationConfig
from agents.architectures.llama_server_process import LlamaServerConnection
from agents.architectures.llama_server_runner import LlamaServerRunner
from agents.models.tool_calling import ChatMessage, ModelRequestConfig


class StreamResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self.lines)


def _loaded_runner(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUFtest")
    artifact = SimpleNamespace(id="artifact-id", model_id="test/model")
    model_manager = MagicMock()
    model_manager.require_installed.return_value = (artifact, model_path)
    server_manager = MagicMock()
    server_manager.start.return_value = LlamaServerConnection(
        "http://127.0.0.1:43123",
        "private-key",
        "cpu",
        "test/model",
        str(model_path),
    )
    client = MagicMock()
    with patch("agents.architectures.llama_server_runner.httpx.Client", return_value=client):
        runner = LlamaServerRunner(
            model_manager=model_manager,
            server_manager=server_manager,
        )
        runner.load("test/model")
    return runner, client, model_manager, server_manager


def test_load_resolves_managed_artifact_and_starts_private_server(tmp_path):
    runner, _client, model_manager, server_manager = _loaded_runner(tmp_path)

    model_manager.require_installed.assert_called_once_with("test/model")
    server_manager.start.assert_called_once()
    assert runner.headers["Authorization"] == "Bearer private-key"


def test_complete_messages_adapts_openai_response(tmp_path):
    runner, client, _model_manager, _server_manager = _loaded_runner(tmp_path)
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}]
    }
    client.post.return_value = response

    result = runner.complete_messages(
        [{"role": "user", "content": "hi"}],
        GenerationConfig(max_tokens=20, temperature=0.2),
    )

    assert result[-1] == {"role": "assistant", "content": "hello"}
    payload = client.post.call_args.kwargs["json"]
    assert payload["model"] == "test/model"
    assert payload["stream"] is False


def test_complete_messages_captures_native_llama_timings(tmp_path):
    runner, client, _model_manager, _server_manager = _loaded_runner(tmp_path)
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {
            "prompt_tokens": 44,
            "completion_tokens": 12,
            "total_tokens": 56,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
        "timings": {
            "cache_n": 20,
            "prompt_n": 24,
            "prompt_ms": 120.0,
            "prompt_per_second": 200.0,
            "predicted_n": 12,
            "predicted_ms": 600.0,
            "predicted_per_second": 20.0,
        },
    }
    client.post.return_value = response

    with patch(
        "agents.architectures.llama_server_runner.time.perf_counter",
        side_effect=[5.0, 5.8],
    ):
        result = runner.complete_messages_with_stats(
            [{"role": "user", "content": "hi"}],
            GenerationConfig(max_tokens=20),
        )

    assert result.generation_stats is not None
    assert result.generation_stats.backend == "llama.cpp"
    assert result.generation_stats.prompt_tokens == 44
    assert result.generation_stats.cached_prompt_tokens == 20
    assert result.generation_stats.completion_tokens == 12
    assert result.generation_stats.prompt_seconds == 0.12
    assert result.generation_stats.generation_seconds == 0.6
    assert result.generation_stats.prompt_tps == 200.0
    assert result.generation_stats.generation_tps == 20.0
    assert result.generation_stats.completion_tps == pytest.approx(15.0)


def test_stream_normalizes_text_and_tool_call_deltas(tmp_path):
    runner, client, _model_manager, _server_manager = _loaded_runner(tmp_path)
    chunks = [
        {"choices": [{"delta": {"content": "Use "}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "lookup", "arguments": '{"id":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "7}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"]
    client.stream.return_value = StreamResponse(lines)

    events = list(
        runner.stream_model_turn(
            [ChatMessage(role="user", content="find it")],
            [],
            ModelRequestConfig(max_tokens=40),
        )
    )

    assert events[0].kind == "text_delta"
    assert events[0].text == "Use "
    turn = events[-1].turn
    assert turn is not None
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"id": 7}


def test_stream_captures_terminal_usage_and_timings(tmp_path):
    runner, client, _model_manager, _server_manager = _loaded_runner(tmp_path)
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            "timings": {
                "cache_n": 0,
                "prompt_n": 8,
                "prompt_ms": 40.0,
                "prompt_per_second": 200.0,
                "predicted_n": 4,
                "predicted_ms": 200.0,
                "predicted_per_second": 20.0,
            },
        },
    ]
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"]
    client.stream.return_value = StreamResponse(lines)

    with patch(
        "agents.architectures.llama_server_runner.time.perf_counter",
        side_effect=[10.0, 10.1, 10.5],
    ):
        events = list(
            runner.stream_model_turn(
                [ChatMessage(role="user", content="hi")],
                [],
                ModelRequestConfig(max_tokens=4),
            )
        )

    stats = events[-1].turn.generation_stats
    assert stats is not None
    assert stats.prompt_tokens == 8
    assert stats.completion_tokens == 4
    assert stats.time_to_first_token == pytest.approx(0.1)
    assert stats.total_seconds == pytest.approx(0.5)
