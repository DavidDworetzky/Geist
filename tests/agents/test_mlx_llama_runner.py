"""Selection tests for the switchable MLX runner."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from agents.architectures.base_runner import GenerationConfig
from agents.architectures.llama.mlx_lm_backend import MLXLMBackend
from agents.architectures.mlx_llama_runner import MLXLlamaRunner


def _backend_module(name, class_name, backend_class):
    module = ModuleType(name)
    setattr(module, class_name, backend_class)
    return module


def test_manual_is_default_and_receives_requested_path(monkeypatch):
    monkeypatch.delenv("GEIST_MLX_IMPLEMENTATION", raising=False)
    backend = MagicMock()
    backend_class = MagicMock(return_value=backend)
    module_name = "agents.architectures.llama.llama_mlx"
    fake_module = _backend_module(module_name, "LlamaMLX", backend_class)
    with patch.dict(sys.modules, {module_name: fake_module}):
        runner = MLXLlamaRunner()
        runner.load(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            {"weights_dir": "/models/llama"},
        )

    assert runner.implementation == "manual"
    backend_class.assert_called_once_with(
        max_new_tokens=16,
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        weights_dir="/models/llama",
    )


def test_mlx_lm_can_be_selected_by_environment(monkeypatch):
    monkeypatch.setenv("GEIST_MLX_IMPLEMENTATION", "mlx-lm")
    backend = MagicMock()
    backend_class = MagicMock(return_value=backend)
    module_name = "agents.architectures.llama.mlx_lm_backend"
    fake_module = _backend_module(module_name, "MLXLMBackend", backend_class)
    with patch.dict(sys.modules, {module_name: fake_module}):
        runner = MLXLlamaRunner()
        runner.load("model-id", {"weights_dir": "/models/llama"})

    assert runner.implementation == "mlx_lm"
    backend_class.assert_called_once_with(
        max_new_tokens=16,
        model_id="model-id",
        weights_dir="/models/llama",
    )


def test_device_config_overrides_environment(monkeypatch):
    monkeypatch.setenv("GEIST_MLX_IMPLEMENTATION", "mlx_lm")
    backend_class = MagicMock(return_value=MagicMock())
    module_name = "agents.architectures.llama.llama_mlx"
    fake_module = _backend_module(module_name, "LlamaMLX", backend_class)
    with patch.dict(sys.modules, {module_name: fake_module}):
        runner = MLXLlamaRunner()
        runner.load("model-id", {"implementation": "manual"})
    assert runner.implementation == "manual"


def test_unknown_implementation_is_rejected():
    runner = MLXLlamaRunner()
    with pytest.raises(ValueError, match="Unknown MLX implementation"):
        runner.load("model-id", {"implementation": "other"})


def test_generation_config_and_response_contract():
    runner = MLXLlamaRunner()
    runner.llama = MagicMock()
    runner.llama.complete.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    config = GenerationConfig(max_tokens=8, temperature=0.2, top_p=0.9)

    result = runner.complete("system", "hello", config)

    assert result[-1]["content"] == "hi"
    assert runner.llama.max_new_tokens == 8
    assert runner.llama.temperature == 0.2
    assert runner.llama.top_p == 0.9
    runner.llama.complete.assert_called_once_with(
        system_prompt="system",
        user_prompt="hello",
    )


def test_structured_messages_reach_mlx_backend_unchanged():
    runner = MLXLlamaRunner()
    runner.llama = MagicMock()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Remember cobalt."},
        {"role": "assistant", "content": "I will remember cobalt."},
        {"role": "user", "content": "What should you remember?"},
    ]
    runner.llama.complete_messages.return_value = [
        {"role": "user", "content": messages[-1]["content"]},
        {"role": "assistant", "content": "cobalt"},
    ]

    result = runner.complete_messages(messages, GenerationConfig(max_tokens=12))

    assert result[-1]["content"] == "cobalt"
    runner.llama.complete_messages.assert_called_once_with(messages)


def test_mlx_lm_prompt_uses_native_roles_for_conversation_history():
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.tokenizer = MagicMock()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Remember cobalt."},
        {"role": "assistant", "content": "I will remember cobalt."},
        {"role": "user", "content": "What should you remember?"},
    ]
    backend.tokenizer.apply_chat_template.return_value = "rendered prompt"

    prompt = backend._build_messages_prompt(messages)

    assert prompt == "rendered prompt"
    backend.tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
