"""OpenAI-wire contract tests for the managed llama-server runner."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": "7}"}}
                        ]
                    },
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
