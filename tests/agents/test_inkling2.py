"""Focused tests for the in-repo Inkling 2 Mini MLX implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agents.architectures.chat_template_tools import (
    parse_tool_response,
    provider_tool_name,
    strip_inkling_response_markup,
)
from agents.architectures.llama.mlx_lm_backend import (
    MLXLMBackend,
    _InklingStreamFilter,
    _is_inkling2_model,
    _load_model_and_tokenizer,
    _read_local_model_type,
)
from agents.model_catalog import get_model_spec, infer_model_spec
from agents.models.tool_calling import ChatMessage, ModelRequestConfig, ToolDefinition
from app.services.local_models import CURATED_LOCAL_ARTIFACTS


def _tiny_inkling2_config(*, core_start: int = 1) -> dict:
    """Build a minimal 4-layer (looped x 2) Inkling 2 config."""
    return {
        "model_type": "inkling2_loop",
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "vocab_size": 128,
            "unpadded_vocab_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 16,
            "sliding_window_size": 16,
            "local_layer_ids": [0, 1, 2],
            "d_rel": 8,
            "rel_extent": 16,
            "log_scaling_n_floor": 16,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 10.0,
            "rel_l2_clip_threshold": 10.0,
            "sconv_kernel_size": 3,
            "n_layers_per_attn_res_block": 2,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "intermediate_size": 64,
            "mlp_gate_softcap": 30.0,
            "mlp_up_softcap": 30.0,
            "route_scale": 1.0,
            "final_logit_softcapping": 30.0,
            "loop_config": {
                "core_start": core_start,
                "core_end": 3,
                "num_passes": 2,
                "gate_rank": 16,
            },
        },
        "vision_config": {
            "hidden_size": 64,
        },
    }


def _install_stub_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install minimal stub mlx_lm modules so backend unit tests run without MLX."""
    state = SimpleNamespace(
        load_calls=[],
        load_model_calls=[],
        load_tokenizer_calls=[],
    )
    mlx_lm_mod = ModuleType("mlx_lm")
    mlx_lm_utils = ModuleType("mlx_lm.utils")
    mlx_lm_sample = ModuleType("mlx_lm.sample_utils")

    def fake_load(path_or_id):
        state.load_calls.append(path_or_id)
        return SimpleNamespace(model_type="llama"), SimpleNamespace()

    def fake_load_model(model_path, *, lazy=False, get_model_classes=None):
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        classes = get_model_classes(config) if get_model_classes else None
        state.load_model_calls.append((model_path, lazy, classes))
        return SimpleNamespace(model_type=config["model_type"]), config

    def fake_load_tokenizer(model_path, *, eos_token_ids=None):
        state.load_tokenizer_calls.append((model_path, eos_token_ids))
        return SimpleNamespace(eos_token_ids=eos_token_ids)

    mlx_lm_mod.load = fake_load
    mlx_lm_utils.load_model = fake_load_model
    mlx_lm_utils.load_tokenizer = fake_load_tokenizer
    mlx_lm_sample.make_sampler = lambda **kwargs: kwargs
    mlx_lm_sample.make_logits_processors = lambda: []

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", mlx_lm_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)
    return state


def test_catalog_and_curated_artifacts_include_inkling2_mini() -> None:
    spec = get_model_spec("thinkingmachines/Inkling-2-Mini")
    assert spec is not None
    assert spec.family == "inkling"
    assert spec.backend == "mlx_llama"
    assert spec.context_window == 1_048_576
    assert spec.supports_function_calling is True
    assert spec.supports_reasoning is True

    inferred = infer_model_spec("mlx-community/Inkling-2-Mini-4bit")
    assert inferred is not None
    assert inferred.id == "thinkingmachines/Inkling-2-Mini"

    by_id = {item.id: item for item in CURATED_LOCAL_ARTIFACTS}
    official = by_id.get("inkling-2-mini-mlx")
    quantized = by_id.get("inkling-2-mini-4bit-mlx")
    assert official is not None
    assert official.repo_id == "thinkingmachines/Inkling-2-Mini"
    assert official.backend == "mlx_llama"
    assert "model*.safetensors" in official.allow_patterns
    assert quantized is not None
    assert quantized.repo_id == "mlx-community/Inkling-2-Mini-4bit"


def test_inkling2_response_markup_and_tool_parsing() -> None:
    raw = (
        "<|message_model|><|content_thinking|>Let me think.<|end_message|>"
        "<|message_model|><|content_text|>The answer is 42.<|end_message|>"
        "<|content_model_end_sampling|>"
    )
    assert strip_inkling_response_markup(raw) == "The answer is 42."

    tool_raw = (
        "<|message_model|><|content_thinking|>Calling weather.<|end_message|>"
        '<|message_model|><|content_invoke_tool_json|>{"name": "get_weather", '
        '"args": {"city": "Austin"}}<|end_message|><|content_model_end_sampling|>'
    )
    turn = parse_tool_response(
        tool_raw,
        provider_to_internal={"get_weather": "weather.get"},
    )
    assert turn.text == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "weather.get"
    assert turn.tool_calls[0].arguments == {"city": "Austin"}

    # Mixed text + tool call in the same Inkling response
    mixed_raw = (
        "<|message_model|><|content_thinking|>Need weather.<|end_message|>"
        "<|message_model|><|content_text|>Checking Austin now.<|end_message|>"
        '<|message_model|><|content_invoke_tool_json|>{"name": "get_weather", '
        '"args": {"city": "Austin"}}<|end_message|><|content_model_end_sampling|>'
    )
    mixed_turn = parse_tool_response(
        mixed_raw,
        provider_to_internal={"get_weather": "weather.get"},
    )
    assert mixed_turn.text == "Checking Austin now."
    assert len(mixed_turn.tool_calls) == 1
    assert mixed_turn.tool_calls[0].name == "weather.get"
    assert mixed_turn.tool_calls[0].arguments == {"city": "Austin"}

    # Streaming filter handles split control markers cleanly
    stream_filter = _InklingStreamFilter()
    chunks = [
        "<|message_",
        "model|><|content_think",
        "ing|>hidden reasoning<|end_mes",
        "sage|><|message_model|><|con",
        "tent_text|>Hello ",
        "world!<|end_",
        "message|><|content_model_end_sampling|>",
    ]
    emitted = "".join(stream_filter.feed(chunk) for chunk in chunks) + stream_filter.flush()
    assert emitted == "Hello world!"

    # Streaming filter preserves plain text before any marker and suppresses tool JSON blocks
    mixed_filter = _InklingStreamFilter()
    mixed_chunks = [
        "Prefix ",
        "<|message_model|><|content_invoke_tool_json|>",
        '{"name":"get_weather","args":{"city":"Austin"}}',
        "<|end_message|><|message_model|><|content_text|>Done.<|end_message|>",
    ]
    mixed_emitted = (
        "".join(mixed_filter.feed(chunk) for chunk in mixed_chunks) + mixed_filter.flush()
    )
    assert mixed_emitted == "Prefix Done."


def test_read_local_model_type_and_loader_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_stub_mlx_lm(monkeypatch)
    fake_inkling_pkg = ModuleType("agents.architectures.inkling2")
    fake_inkling_pkg.Model = type("FakeInklingModel", (), {})
    fake_inkling_pkg.ModelArgs = type("FakeInklingArgs", (), {})
    monkeypatch.setitem(sys.modules, "agents.architectures.inkling2", fake_inkling_pkg)

    # Corrupt UTF-8 config falls back to None without raising
    bad_dir = tmp_path / "bad_utf8"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_bytes(b"\xff\xfe\xfd")
    assert _read_local_model_type(str(bad_dir)) is None

    # Local inkling2_loop snapshot routes through load_model + load_tokenizer
    inkling_dir = tmp_path / "inkling_snap"
    inkling_dir.mkdir()
    (inkling_dir / "config.json").write_text(
        json.dumps({"model_type": "inkling2_loop", "eos_token_id": 200006}),
        encoding="utf-8",
    )
    assert _read_local_model_type(str(inkling_dir)) == "inkling2_loop"
    model, tokenizer = _load_model_and_tokenizer(str(inkling_dir))
    assert model.model_type == "inkling2_loop"
    assert tokenizer.eos_token_ids == 200006
    assert len(state.load_model_calls) == 1
    assert state.load_model_calls[0][2] == (fake_inkling_pkg.Model, fake_inkling_pkg.ModelArgs)
    assert state.load_calls == []

    # Non-Inkling model falls back to stock mlx_lm.load
    other_dir = tmp_path / "llama_snap"
    other_dir.mkdir()
    (other_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    _load_model_and_tokenizer(str(other_dir))
    assert state.load_calls == [str(other_dir)]


def test_mlx_lm_backend_inkling2_defaults_streaming_and_tool_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_mlx_lm(monkeypatch)
    assert _is_inkling2_model("thinkingmachines/Inkling-2-Mini") is True
    assert _is_inkling2_model("mlx-community/Inkling-2-Mini-4bit") is True
    assert _is_inkling2_model("Qwen/Qwen3-4B") is False

    class DummyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            tools = kwargs.get("tools")
            return "with_tools" if tools else "prompt"

    dummy_model = SimpleNamespace(model_type="inkling2_loop")

    monkeypatch.setattr(
        "agents.architectures.llama.mlx_lm_backend._configure_thread_local_generation_stream",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.architectures.llama.mlx_lm_backend._load_model_and_tokenizer",
        lambda path_or_id: (dummy_model, DummyTokenizer()),
    )

    backend = MLXLMBackend(
        max_new_tokens=32,
        model_id="thinkingmachines/Inkling-2-Mini",
    )
    assert backend._template_options()["reasoning_effort"] == "none"

    # stream_messages applies _InklingStreamFilter when tools is empty,
    # and preserves raw markup when tools are provided.
    raw_chunks = [
        "<|message_model|><|content_thinking|>hidden<|end_message|>",
        "<|message_model|><|content_text|>Hello ",
        "world!<|end_message|>",
        '<|message_model|><|content_invoke_tool_json|>{"name":"weather_get","args":{"city":"Austin"}}<|end_message|>',
    ]
    monkeypatch.setattr(
        backend,
        "_stream_prompt",
        lambda prompt, sampler, logits_processors=None: iter(raw_chunks),
    )

    plain_stream = "".join(backend.stream_messages([{"role": "user", "content": "Hi"}]))
    assert plain_stream == "Hello world!"

    tool_stream = "".join(
        backend.stream_messages(
            [{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "weather_get"}}],
        )
    )
    assert "<|content_invoke_tool_json|>" in tool_stream

    # stream_model_turn streams text_delta incrementally and closes the underlying generator
    closed = {"flag": False}

    provider_name = provider_tool_name("weather.get")

    def closing_stream(messages, tools=None):
        try:
            yield "<|message_model|><|content_thinking|>plan<|end_message|>"
            yield "<|message_model|><|content_text|>Checking "
            yield "Austin.<|end_message|>"
            yield (
                f'<|message_model|><|content_invoke_tool_json|>{{"name":"{provider_name}",'
                '"args":{"city":"Austin"}}<|end_message|>'
            )
        finally:
            closed["flag"] = True

    monkeypatch.setattr(backend, "stream_messages", closing_stream)
    backend.supports_native_tool_calling = True
    tool = ToolDefinition(
        name="weather.get",
        description="Get weather",
        handler=lambda **kw: kw,
        arguments_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    events = list(
        backend.stream_model_turn(
            [ChatMessage(role="user", content="Weather in Austin?")],
            [tool],
            ModelRequestConfig(max_tokens=64),
        )
    )
    assert closed["flag"] is True
    text_deltas = [ev.text for ev in events if ev.kind == "text_delta"]
    assert len(text_deltas) >= 2  # streamed incrementally, not buffered into one chunk
    assert "".join(text_deltas) == "Checking Austin."
    final = events[-1]
    assert final.kind == "turn_complete"
    assert final.turn is not None
    assert final.turn.text == "Checking Austin."
    assert len(final.turn.tool_calls) == 1
    assert final.turn.tool_calls[0].name == "weather.get"
    assert final.turn.tool_calls[0].arguments == {"city": "Austin"}

    # complete_messages strips Inkling markup cleanly
    monkeypatch.setattr(
        backend,
        "stream_messages",
        lambda messages, tools=None: iter(
            [
                "<|message_model|><|content_thinking|>hidden<|end_message|>",
                "<|message_model|><|content_text|>Hello world!<|end_message|>",
            ]
        ),
    )
    result = backend.complete_messages([{"role": "user", "content": "Hi"}])
    assert result == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello world!"},
    ]


def test_inkling2_tiny_model_forward_decode_and_taps() -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from agents.architectures.inkling2 import Model, ModelArgs

    args = ModelArgs.from_dict(_tiny_inkling2_config())
    assert args.vision is None

    model = Model(args)
    cache = model.make_cache()
    assert len(cache) == 6

    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 4, 128)

    next_token = mx.argmax(logits[:, -1:, :], axis=-1).astype(mx.int32)
    step_hidden, taps = model.model.language_model(
        next_token,
        cache=cache,
        return_taps={1, 4, 6},
    )
    mx.eval(step_hidden, *taps.values())
    assert step_hidden.shape == (1, 1, 64)
    assert sorted(taps.keys()) == [1, 4, 6]
    for tensor in taps.values():
        assert tensor.shape == (1, 1, 64)

    # Loop with core_start == 0 and num_passes > 1 fails with a clear ValueError
    no_prelude_model = Model(ModelArgs.from_dict(_tiny_inkling2_config(core_start=0)))
    with pytest.raises(ValueError, match="prelude layers"):
        no_prelude_model(prompt, cache=no_prelude_model.make_cache())


def test_inkling2_nvfp4_global_scale_to_quantized_and_sanitize() -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from agents.architectures.inkling2 import Model, ModelArgs
    from agents.architectures.inkling2.moe import QuantizedSwitchLinear, SwitchLinear

    mx.random.seed(7)
    w = mx.random.normal((4, 32, 64)).astype(mx.bfloat16)
    wq, scales = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
    global_scale = mx.array([2.0, 0.5, 1.5, 3.0], dtype=mx.float32)

    qsl = QuantizedSwitchLinear(
        input_dims=64,
        output_dims=32,
        num_experts=4,
        group_size=16,
        bits=4,
        mode="nvfp4",
    )
    qsl.weight = wq
    qsl.scales = scales
    qsl.global_scale = global_scale

    x = mx.random.normal((1, 2, 1, 64)).astype(mx.bfloat16)
    indices = mx.array([[[0, 2], [1, 3]]], dtype=mx.uint32)
    out = qsl(x, indices)
    mx.eval(out)

    # Compare against explicit dequantization + global_scale / (448 * 6)
    w_deq = mx.dequantize(wq, scales, group_size=16, bits=4, mode="nvfp4", dtype=mx.float32)
    w_scaled = w_deq * (global_scale[:, None, None] / (448.0 * 6.0))
    ref_list = []
    for t in range(2):
        tok_out = []
        for k in range(2):
            e = int(indices[0, t, k].item())
            tok_out.append(x[0, t, 0].astype(mx.float32) @ w_scaled[e].T)
        ref_list.append(mx.stack(tok_out, axis=0))
    ref = mx.stack(ref_list, axis=0)[None, ...]
    mx.eval(ref)
    max_diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item())
    assert max_diff < 5e-2

    # SwitchLinear.to_quantized() quantizes the source weights (not random weights)
    sl = SwitchLinear(input_dims=64, output_dims=32, num_experts=4, bias=False)
    sl.weight = w
    ql_from_sl = sl.to_quantized(group_size=32, bits=8, mode="mxfp8")
    out_sl = sl(x, indices)
    out_ql = ql_from_sl(x, indices)
    mx.eval(out_sl, out_ql)
    mean_diff = float(mx.mean(mx.abs(out_sl.astype(mx.float32) - out_ql.astype(mx.float32))).item())
    assert mean_diff < 0.25

    # Test sanitize drops vision weights and remaps prefixes/conv1d/scales
    args = ModelArgs.from_dict(_tiny_inkling2_config())
    model = Model(args)
    conv_w = mx.zeros((64, 1, 3), dtype=mx.bfloat16)
    sanitized = model.sanitize(
        {
            "model.visual.backbone.ln_pre.weight": mx.zeros((8,)),
            "model.llm.layers.0.attn_sconv.weight": conv_w,
            "model.llm.layers.0.mlp.experts.w13_weight": wq.view(mx.uint8),
            "model.llm.layers.0.mlp.experts.w13_weight_scale": scales,
            "model.llm.layers.0.mlp.experts.w13_weight_amax": mx.array(
                [0.5, 1.0, 2.0, 4.0], dtype=mx.float32
            ),
            "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator": mx.array(
                [448.0, 448.0, 448.0, 448.0], dtype=mx.float32
            ),
        }
    )
    assert not any("vision_tower" in k for k in sanitized)
    assert sanitized[
        "model.language_model.layers.0.post_attention_convolution.conv1d.weight"
    ].shape == (64, 3, 1)
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj.weight" in sanitized
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj.scales" in sanitized
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj.global_scale" in sanitized
    assert isinstance(
        model.model.language_model.layers[0].mlp.experts.gate_up_proj,
        QuantizedSwitchLinear,
    )

    # Incomplete cross-pass query pair raises ValueError naming the missing tensor
    with pytest.raises(ValueError, match="cross_pass_kv_queries"):
        model.sanitize({"model.llm.cross_pass_queries": mx.zeros((2, 64))})
