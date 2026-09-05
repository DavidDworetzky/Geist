"""Selection tests for the switchable MLX runner."""

import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.architectures.base_runner import GenerationConfig
from agents.architectures.chat_template_tools import provider_tool_name
from agents.architectures.llama.mlx_lm_backend import (
    MLXLMBackend,
    _first_stop_index,
    _is_qwen3_model,
    _normalize_stops,
    _prefill_step_size,
    _stop_prefix_length,
)
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.models.tool_calling import (
    ChatMessage,
    ModelRequestConfig,
    ToolDefinition,
    ToolExecutionOutput,
)


def _search_tool():
    return ToolDefinition(
        name="web.search",
        description="Search the web",
        arguments_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        handler=lambda _context, _arguments: ToolExecutionOutput(content="unused"),
    )


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


def test_large_llama_defaults_to_manual_implementation(monkeypatch):
    monkeypatch.delenv("GEIST_MLX_IMPLEMENTATION", raising=False)
    backend = MagicMock()
    backend_class = MagicMock(return_value=backend)
    module_name = "agents.architectures.llama.llama_mlx"
    fake_module = _backend_module(module_name, "LlamaMLX", backend_class)

    with patch.dict(sys.modules, {module_name: fake_module}):
        runner = MLXLlamaRunner()
        runner.load(
            "mlx-community/Llama-3.1-70B-Instruct-4bit",
            {"weights_dir": "/models/llama-70b"},
        )

    assert runner.implementation == "manual"
    backend_class.assert_called_once_with(
        max_new_tokens=16,
        model_id="mlx-community/Llama-3.1-70B-Instruct-4bit",
        weights_dir="/models/llama-70b",
    )


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("llama", True),
        ("qwen3_5", False),
        ("mllama", False),
    ],
)
def test_manual_support_uses_installed_model_architecture(tmp_path, model_type, expected):
    (tmp_path / "config.json").write_text(
        f'{{"model_type": "{model_type}"}}',
        encoding="utf-8",
    )

    assert (
        MLXLlamaRunner._manual_implementation_supports(
            "custom/Llama-named-checkpoint",
            str(tmp_path),
        )
        is expected
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

    assert runner.model_id is None
    assert runner.implementation is None
    assert runner.weights_dir is None


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
            "mlx-community/Llama-3.1-8B-Instruct-4bit",
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
    runner.implementation = "mlx_lm"
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


def test_manual_backend_warns_once_for_unsupported_generation_controls(caplog):
    runner = MLXLlamaRunner()
    runner.llama = MagicMock()
    runner.implementation = "manual"
    config = GenerationConfig(
        frequency_penalty=0.3,
        presence_penalty=0.5,
        stop=["STOP", "END"],
    )

    runner._apply_generation_config(config)
    runner._apply_generation_config(config)

    warnings = [
        record for record in caplog.records if "manual MLX implementation ignores" in record.message
    ]
    assert len(warnings) == 1


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


def test_mlx_runner_streams_backend_segments_without_buffering():
    runner = MLXLlamaRunner()
    runner.llama = MagicMock()
    runner.llama.stream_model_turn = None
    runner.llama.stream_messages.return_value = iter(("co", "balt"))

    events = list(
        runner.stream_model_turn(
            [ChatMessage(role="user", content="Name the code word.")],
            [],
            ModelRequestConfig(max_tokens=8, temperature=0.0),
        )
    )

    assert [event.kind for event in events] == [
        "text_delta",
        "text_delta",
        "turn_complete",
    ]
    assert [event.text for event in events[:2]] == ["co", "balt"]
    assert events[-1].turn.text == "cobalt"


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


def test_mlx_lm_prompt_keeps_structural_template_options():
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.tokenizer = MagicMock()
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.chat_template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
    }
    backend.tokenizer.apply_chat_template.return_value = "rendered prompt"
    messages = [{"role": "user", "content": "hello"}]

    backend._build_messages_prompt(messages)

    backend.tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def test_mlx_lm_native_turn_preserves_tool_history_and_parses_call():
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.supports_native_tool_calling = True
    safe_name = provider_tool_name("web.search")
    backend.stream_messages = MagicMock(
        return_value=iter(
            [
                "<tool_",
                f'call>{{"name":"{safe_name}","arguments":',
                '{"query":"celebrity news"}}</tool_call>',
            ]
        )
    )
    messages = [
        ChatMessage(role="user", content="Find news"),
        ChatMessage(
            role="assistant",
            content="Earlier answer",
        ),
        ChatMessage(role="user", content="Search now"),
    ]

    events = list(
        backend.stream_model_turn(
            messages,
            [_search_tool()],
            ModelRequestConfig(),
        )
    )

    rendered_messages, rendered_tools = backend.stream_messages.call_args.args
    assert rendered_messages[1] == {"role": "assistant", "content": "Earlier answer"}
    assert rendered_tools[0]["function"]["name"] == safe_name
    turn = events[-1].turn
    assert turn is not None
    assert turn.tool_calls[0].name == "web.search"
    assert turn.tool_calls[0].arguments == {"query": "celebrity news"}


@pytest.mark.parametrize("cancel", [False, True])
def test_mlx_lm_plain_turn_streams_lazily_and_closes(cancel):
    backend = MLXLMBackend.__new__(MLXLMBackend)
    produced = []
    closed = []

    def responses(*args):
        try:
            for segment in ("co", "balt"):
                produced.append(segment)
                yield segment
        finally:
            closed.append(True)

    backend.stream_messages = responses
    events = backend.stream_model_turn(
        [ChatMessage(role="user", content="Name the code word.")], [], ModelRequestConfig()
    )
    first = next(events)
    assert first.kind == "text_delta"
    assert produced == ["co"]
    if cancel:
        events.close()
        assert produced == ["co"]
    else:
        remaining = list(events)
        assert [event.kind for event in remaining] == ["text_delta", "turn_complete"]
        assert remaining[-1].turn.text == "cobalt"
    assert closed == [True]


def test_manual_mlx_stays_tool_disabled():
    runner = MLXLlamaRunner()
    runner.llama = MagicMock()
    runner.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    runner.implementation = "manual"
    runner.supports_native_tool_calling = False

    with pytest.raises(ValueError, match="does not support native tool calling"):
        list(
            runner.stream_model_turn(
                [ChatMessage(role="user", content="search")],
                [_search_tool()],
                ModelRequestConfig(),
            )
        )


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen3-8B", True),
        ("Qwen/Qwen3.8-27B", True),
        ("mlx-community/Qwen3.8-27B-4bit", True),
        ("Qwen/Qwen3-32B", True),
        ("Qwen/Qwen2.5-7B-Instruct", False),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", False),
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


def test_qwen3_stream_warns_for_unsupported_penalties_and_cross_chunk_stop(caplog):
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

    def response_stream():
        for response in responses:
            yield response.text

    stream = response_stream()
    backend._stream_prompt = MagicMock(return_value=stream)
    make_sampler = MagicMock(return_value="sampler")
    logits_processors = [object(), object()]
    make_logits_processors = MagicMock(return_value=logits_processors)
    mlx_lm_module = ModuleType("mlx_lm")
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
    make_logits_processors.assert_called_once_with()
    assert "presence/frequency penalties" in caplog.text
    backend._stream_prompt.assert_called_once_with("rendered prompt", "sampler", logits_processors)
    assert stream.gi_frame is None


def test_non_qwen_stream_omits_top_k_and_empty_logits_processors():
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model = object()
    backend.tokenizer = MagicMock()
    backend.tokenizer.apply_chat_template.return_value = "rendered prompt"
    backend.chat_template_kwargs = {}
    backend.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    backend.max_new_tokens = 8
    backend.temperature = 0.7
    backend.top_p = 0.9
    backend.frequency_penalty = 0.0
    backend.presence_penalty = 0.0
    backend.stop = None

    backend._stream_prompt = MagicMock(return_value=iter(()))
    make_sampler = MagicMock(return_value="sampler")
    make_logits_processors = MagicMock(return_value=[])
    mlx_lm_module = ModuleType("mlx_lm")
    sample_utils_module = _backend_module(
        "mlx_lm.sample_utils",
        "make_sampler",
        make_sampler,
    )
    sample_utils_module.make_logits_processors = make_logits_processors

    with patch.dict(
        sys.modules,
        {
            "mlx_lm": mlx_lm_module,
            "mlx_lm.sample_utils": sample_utils_module,
        },
    ):
        assert list(backend.stream_messages([{"role": "user", "content": "hello"}])) == []

    make_sampler.assert_called_once_with(temp=0.7, top_p=0.9)
    make_logits_processors.assert_called_once_with()
    backend._stream_prompt.assert_called_once_with("rendered prompt", "sampler", None)


def test_mlx_lm_uses_a_thread_local_generation_stream():
    mlx_core = MagicMock()
    generation_module = MagicMock()
    thread_local_stream = MagicMock()
    mlx_lm_module = ModuleType("mlx_lm")
    tokenizer = MagicMock()
    tokenizer.chat_template = "{{ tools }}"
    tokenizer.apply_chat_template.side_effect = lambda _messages, **kwargs: str(
        kwargs.get("tools") or []
    )
    mlx_lm_module.load = MagicMock(return_value=(MagicMock(), tokenizer))
    mlx_core.new_thread_local_stream.return_value = thread_local_stream

    with (
        patch.dict(sys.modules, {"mlx_lm": mlx_lm_module}),
        patch(
            "agents.architectures.llama.mlx_lm_backend.importlib.import_module",
            side_effect=lambda name: (mlx_core if name == "mlx.core" else generation_module),
        ),
    ):
        backend = MLXLMBackend(max_new_tokens=8, model_id="Qwen/Qwen3.8-27B")

    assert generation_module.generation_stream is thread_local_stream
    assert backend.supports_native_tool_calling is True


def test_mlx_prefill_step_size_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("GEIST_MLX_PREFILL_STEP_SIZE", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        _prefill_step_size()


def test_mlx_lm_reuses_an_exact_conversation_prefix(monkeypatch):
    monkeypatch.setenv("GEIST_MLX_DFLASH", "off")
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model = MagicMock()
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.tokenizer = MagicMock()
    backend.tokenizer.bos_token = None
    backend.tokenizer.apply_chat_template.side_effect = ("first", "second")
    backend.tokenizer.encode.side_effect = ([1, 2], [1, 2, 3, 4])
    backend.max_new_tokens = 2
    backend.temperature = 0.0
    backend.top_p = 1.0
    backend.chat_template_kwargs = {}
    backend.frequency_penalty = 0.0
    backend.presence_penalty = 0.0
    backend.stop = None
    backend.prefill_step_size = 2048
    backend._generation_lock = threading.RLock()
    backend._prompt_cache = None
    backend._cached_tokens = ()
    prompt_cache = MagicMock()

    mlx_core = ModuleType("mlx.core")
    mlx_core.array = MagicMock(side_effect=lambda tokens: tuple(tokens))
    mlx_module = ModuleType("mlx")
    mlx_module.core = mlx_core
    mlx_lm_module = ModuleType("mlx_lm")
    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = MagicMock(return_value=prompt_cache)
    sampler_module = ModuleType("mlx_lm.sample_utils")
    sampler_module.make_sampler = MagicMock(return_value="sampler")
    sampler_module.make_logits_processors = MagicMock(return_value=[])
    calls = []

    def stream_generate(*args, **kwargs):
        calls.append((args, kwargs))
        token = 9 if len(calls) == 1 else 10
        yield SimpleNamespace(
            text="A" if len(calls) == 1 else "B",
            token=token,
            prompt_tps=100.0,
            generation_tokens=1,
            generation_tps=20.0,
            peak_memory=1.0,
        )

    mlx_lm_module.stream_generate = stream_generate
    modules = {
        "mlx": mlx_module,
        "mlx.core": mlx_core,
        "mlx_lm": mlx_lm_module,
        "mlx_lm.models.cache": cache_module,
        "mlx_lm.sample_utils": sampler_module,
    }

    with patch.dict(sys.modules, modules):
        assert "".join(backend.stream_messages([{"role": "user", "content": "x"}])) == "A"
        backend.tokenizer.encode.side_effect = ([1, 2, 9, 3, 4],)
        assert "".join(backend.stream_messages([{"role": "user", "content": "y"}])) == "B"

    assert calls[0][0][2] == (1, 2)
    assert calls[1][0][2] == (3, 4)
    assert calls[1][1]["prompt_cache"] is prompt_cache
    assert backend.last_stats["cached_prompt_tokens"] == 3
    cache_module.make_prompt_cache.assert_called_once_with(backend.model)


def test_mlx_lm_discards_cache_when_the_prefix_changes(monkeypatch):
    monkeypatch.setenv("GEIST_MLX_DFLASH", "off")
    backend = MLXLMBackend.__new__(MLXLMBackend)
    backend.model = MagicMock()
    backend.model_id = "Qwen/Qwen3.8-27B"
    backend.tokenizer = MagicMock()
    backend.tokenizer.bos_token = None
    backend.tokenizer.apply_chat_template.return_value = "changed"
    backend.tokenizer.encode.return_value = [7, 8]
    backend.max_new_tokens = 1
    backend.temperature = 0.0
    backend.top_p = 1.0
    backend.chat_template_kwargs = {}
    backend.frequency_penalty = 0.0
    backend.presence_penalty = 0.0
    backend.stop = None
    backend.prefill_step_size = 2048
    backend._generation_lock = threading.RLock()
    backend._prompt_cache = MagicMock(name="old_cache")
    backend._cached_tokens = (1, 2, 3)
    fresh_cache = MagicMock(name="fresh_cache")

    mlx_core = ModuleType("mlx.core")
    mlx_core.array = MagicMock(side_effect=lambda tokens: tuple(tokens))
    mlx_module = ModuleType("mlx")
    mlx_module.core = mlx_core
    mlx_lm_module = ModuleType("mlx_lm")
    mlx_lm_module.stream_generate = MagicMock(
        return_value=iter(
            [
                SimpleNamespace(
                    text="C",
                    token=11,
                    prompt_tps=50.0,
                    generation_tokens=1,
                    generation_tps=10.0,
                    peak_memory=1.0,
                )
            ]
        )
    )
    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = MagicMock(return_value=fresh_cache)
    sampler_module = ModuleType("mlx_lm.sample_utils")
    sampler_module.make_sampler = MagicMock(return_value="sampler")
    sampler_module.make_logits_processors = MagicMock(return_value=[])

    with patch.dict(
        sys.modules,
        {
            "mlx": mlx_module,
            "mlx.core": mlx_core,
            "mlx_lm": mlx_lm_module,
            "mlx_lm.models.cache": cache_module,
            "mlx_lm.sample_utils": sampler_module,
        },
    ):
        assert "".join(backend.stream_messages([{"role": "user", "content": "z"}])) == "C"

    assert backend.last_stats["cached_prompt_tokens"] == 0
    assert backend._prompt_cache is fresh_cache
    cache_module.make_prompt_cache.assert_called_once_with(backend.model)
