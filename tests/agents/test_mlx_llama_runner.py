"""Selection tests for the switchable MLX runner."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.architectures.base_runner import GenerationConfig
from agents.architectures.llama.mlx_lm_backend import (
    MLXLMBackend,
    _first_stop_index,
    _is_qwen3_model,
    _normalize_stops,
    _stop_prefix_length,
)
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


def test_qwen3_8_defaults_to_mlx_lm_with_managed_weights(monkeypatch):
    monkeypatch.delenv("GEIST_MLX_IMPLEMENTATION", raising=False)
    monkeypatch.delenv("LOCAL_WEIGHTS_DIR", raising=False)
    backend_class = MagicMock(return_value=MagicMock())
    module_name = "agents.architectures.llama.mlx_lm_backend"
    fake_module = _backend_module(module_name, "MLXLMBackend", backend_class)

    with (
        patch.dict(sys.modules, {module_name: fake_module}),
        patch.object(
            MLXLlamaRunner,
            "_resolve_weights_dir",
            return_value="/models/qwen3.8",
        ),
    ):
        runner = MLXLlamaRunner()
        runner.load("Qwen/Qwen3.8-27B")

    assert runner.implementation == "mlx_lm"
    backend_class.assert_called_once_with(
        max_new_tokens=16,
        model_id="Qwen/Qwen3.8-27B",
        weights_dir="/models/qwen3.8",
    )


def test_qwen_rejects_explicit_manual_implementation():
    runner = MLXLlamaRunner()

    with pytest.raises(ValueError, match="manual MLX implementation only supports Llama"):
        runner.load(
            "Qwen/Qwen3.8-27B",
            {"implementation": "manual", "weights_dir": "/models/qwen3.8"},
        )


def test_mlx_lm_forwards_qwen_chat_template_controls(monkeypatch):
    monkeypatch.delenv("GEIST_MLX_IMPLEMENTATION", raising=False)
    backend_class = MagicMock(return_value=MagicMock())
    module_name = "agents.architectures.llama.mlx_lm_backend"
    fake_module = _backend_module(module_name, "MLXLMBackend", backend_class)

    with patch.dict(sys.modules, {module_name: fake_module}):
        runner = MLXLlamaRunner()
        runner.load(
            "mlx-community/Qwen3.8-27B-4bit",
            {
                "weights_dir": "/models/qwen3.8",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": False,
                },
            },
        )

    backend_class.assert_called_once_with(
        max_new_tokens=16,
        model_id="mlx-community/Qwen3.8-27B-4bit",
        weights_dir="/models/qwen3.8",
        chat_template_kwargs={
            "enable_thinking": True,
            "preserve_thinking": False,
        },
    )


def test_device_config_overrides_environment(monkeypatch):
    monkeypatch.setenv("GEIST_MLX_IMPLEMENTATION", "mlx_lm")
    backend_class = MagicMock(return_value=MagicMock())
    module_name = "agents.architectures.llama.llama_mlx"
    fake_module = _backend_module(module_name, "LlamaMLX", backend_class)
    with patch.dict(sys.modules, {module_name: fake_module}):
        runner = MLXLlamaRunner()
        runner.load(
            "model-id",
            {"implementation": "manual", "weights_dir": "/models/llama"},
        )
    assert runner.implementation == "manual"


def test_unknown_implementation_is_rejected():
    runner = MLXLlamaRunner()
    with pytest.raises(ValueError, match="Unknown MLX implementation"):
        runner.load("model-id", {"implementation": "other"})


def test_managed_snapshot_is_required_when_no_explicit_weights_path():
    manager = MagicMock()
    artifact = MagicMock(id="managed-mlx", backend="mlx_llama")
    manager.require_installed.return_value = (artifact, "/models/managed/snapshot")
    module_name = "app.services.local_models"
    fake_module = ModuleType(module_name)
    fake_module.get_local_model_manager = MagicMock(return_value=manager)

    with patch.dict(sys.modules, {module_name: fake_module}):
        resolved = MLXLlamaRunner._resolve_weights_dir(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            {"artifact_id": "managed-mlx"},
        )

    assert resolved == "/models/managed/snapshot"
    manager.require_installed.assert_called_once_with("managed-mlx")


def test_mlx_runner_rejects_managed_gguf_artifact():
    manager = MagicMock()
    artifact = MagicMock(id="managed-gguf", backend="llama_server")
    manager.require_installed.return_value = (artifact, "/models/model.gguf")
    module_name = "app.services.local_models"
    fake_module = ModuleType(module_name)
    fake_module.get_local_model_manager = MagicMock(return_value=manager)

    with (
        patch.dict(sys.modules, {module_name: fake_module}),
        pytest.raises(ValueError, match="not compatible with the MLX runner"),
    ):
        MLXLlamaRunner._resolve_weights_dir("model-id", {"artifact_id": "managed-gguf"})


def test_generation_config_and_response_contract():
    runner = MLXLlamaRunner()
    runner.llama = MagicMock()
    runner.llama.complete.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    config = GenerationConfig(
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.5,
        stop=["STOP", "END"],
    )

    result = runner.complete("system", "hello", config)

    assert result[-1]["content"] == "hi"
    assert runner.llama.max_new_tokens == 8
    assert runner.llama.temperature == 0.2
    assert runner.llama.top_p == 0.9
    assert runner.llama.frequency_penalty == 0.3
    assert runner.llama.presence_penalty == 0.5
    assert runner.llama.stop == ["STOP", "END"]
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
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.chat_template_kwargs = {}
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
        enable_thinking=False,
    )


def test_mlx_lm_prompt_allows_qwen_thinking_override():
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.tokenizer = MagicMock()
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.chat_template_kwargs = {
        "enable_thinking": True,
        "preserve_thinking": False,
    }
    backend.tokenizer.apply_chat_template.return_value = "rendered prompt"
    messages = [{"role": "user", "content": "hello"}]

    backend._build_messages_prompt(messages)

    backend.tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=False,
    )


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen3-8B", True),
        ("Qwen/Qwen3.8-27B", True),
        ("mlx-community/Qwen3.8-27B-4bit", True),
        ("Qwen/Qwen2.5-7B-Instruct", False),
        ("meta-llama/Llama-3.1-8B-Instruct", False),
    ],
)
def test_qwen3_model_detection(model_id, expected):
    assert _is_qwen3_model(model_id) is expected


def test_stop_sequence_helpers_handle_chunk_boundaries_and_duplicates():
    stops = _normalize_stops(["<END>", "<END>", "STOP", ""])

    assert stops == ("<END>", "STOP")
    assert _first_stop_index("answer<END>ignored", stops) == 6
    assert _first_stop_index("answer", stops) is None
    assert _stop_prefix_length("answer<EN", stops) == 3
    assert _stop_prefix_length("answer", stops) == 0


def test_qwen3_stream_applies_sampler_penalties_and_cross_chunk_stop():
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model = object()
    backend.tokenizer = MagicMock()
    backend.tokenizer.apply_chat_template.return_value = "rendered prompt"
    backend.chat_template_kwargs = {}
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.max_new_tokens = 100
    backend.temperature = 1.0
    backend.top_p = 0.95
    backend.frequency_penalty = 0.2
    backend.presence_penalty = 0.5
    backend.stop = "<END>"

    responses = [
        SimpleNamespace(
            text="answer<EN",
            prompt_tokens=10,
            prompt_tps=20.0,
            generation_tokens=2,
            generation_tps=5.0,
            peak_memory=1.0,
        ),
        SimpleNamespace(
            text="D>ignored",
            prompt_tokens=10,
            prompt_tps=20.0,
            generation_tokens=3,
            generation_tps=5.0,
            peak_memory=1.0,
        ),
    ]
    stream_generate = MagicMock(return_value=iter(responses))
    make_sampler = MagicMock(return_value="sampler")
    logits_processors = [object(), object()]
    make_logits_processors = MagicMock(return_value=logits_processors)
    mlx_lm_module = _backend_module("mlx_lm", "stream_generate", stream_generate)
    sample_utils_module = _backend_module(
        "mlx_lm.sample_utils",
        "make_sampler",
        make_sampler,
    )
    sample_utils_module.make_logits_processors = make_logits_processors

    with (
        patch.dict(
            sys.modules,
            {
                "mlx_lm": mlx_lm_module,
                "mlx_lm.sample_utils": sample_utils_module,
            },
        ),
    ):
        chunks = list(backend.stream_messages([{"role": "user", "content": "hello"}]))

    assert chunks == ["answer"]
    make_sampler.assert_called_once_with(temp=1.0, top_p=0.95, top_k=20)
    make_logits_processors.assert_called_once_with(
        presence_penalty=0.5,
        frequency_penalty=0.2,
    )
    stream_generate.assert_called_once_with(
        backend.model,
        backend.tokenizer,
        "rendered prompt",
        max_tokens=100,
        sampler="sampler",
        logits_processors=logits_processors,
    )


def test_mlx_lm_uses_a_thread_local_generation_stream():
    mlx_core = MagicMock()
    generation_module = MagicMock()
    thread_local_stream = MagicMock()
    mlx_lm_module = ModuleType("mlx_lm")
    mlx_lm_module.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    mlx_core.new_thread_local_stream.return_value = thread_local_stream

    with (
        patch.dict(sys.modules, {"mlx_lm": mlx_lm_module}),
        patch(
            "agents.architectures.llama.mlx_lm_backend.importlib.import_module",
            side_effect=lambda name: (mlx_core if name == "mlx.core" else generation_module),
        ),
    ):
        MLXLMBackend(max_new_tokens=8, model_id="Qwen/Qwen3.8-27B")

    assert generation_module.generation_stream is thread_local_stream
