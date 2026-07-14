"""Selection tests for the switchable MLX runner."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from agents.architectures.base_runner import GenerationConfig
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
