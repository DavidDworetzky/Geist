"""Native Metal checks for speculative state recovery and small-row matmul."""

import itertools
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace

import pytest


if sys.platform != "darwin":
    pytest.skip("Native MLX validation requires Apple Silicon", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from agents.architectures.llama.dflash_backend import (
    DFlashDecoder,
    sampled_accept,
    target_probabilities,
)
from agents.architectures.llama.dflash_model import (
    CandidateSelector,
    DFlashConfig,
    DFlashDraftModel,
)
from agents.architectures.llama.qwen_compiled_verifier import CompiledQwenTarget
from agents.architectures.llama.qwen_expanded_mlp import install_expanded_mlp
from agents.architectures.llama.qwen_kernel_lab import install_lab, lab_matmul
from agents.architectures.llama.qwen_small_m import (
    install_small_m,
    make_dequant_table,
    pack_small_m_weights,
    small_m_matmul,
    tune_small_m,
)
from agents.architectures.llama.qwen_speculative import QwenSpeculativeTarget
from agents.architectures.llama.speculation_policy import SpeculationPolicy
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.models.tool_calling import ModelEvent, ModelRequestConfig


@pytest.fixture
def small_target():
    mx.random.seed(4)
    model = TextModel(
        TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
        )
    )
    model.eval()
    return model, QwenSpeculativeTarget(model, (0, 2))


@pytest.mark.parametrize("keep", [1, 2, 4, 7, 8])
def test_partial_verify_recovers_recurrent_and_attention_state(small_target, keep):
    model, target = small_target
    prefix = mx.array([[1, 2, 3]])
    block = mx.array([[4, 5, 6, 7, 8, 9, 10, 11]])
    expected_cache, actual_cache = model.make_cache(), model.make_cache()
    mx.eval(model(prefix, cache=expected_cache), model(prefix, cache=actual_cache))
    mx.eval(model(block[:, :keep], cache=expected_cache))
    verified, _ = target.forward(block, actual_cache, capture=True)
    mx.eval(verified)
    target.rollback(actual_cache, keep)
    for expected, actual in zip(expected_cache, actual_cache, strict=False):
        for name in ("offset", "lengths", "left_padding"):
            assert getattr(expected, name, None) == getattr(actual, name, None)
        for a, b in zip(expected.state, actual.state, strict=False):
            assert bool(mx.allclose(a, b, atol=3e-5, rtol=3e-4).item())
    next_id = mx.array([[12]])
    expected = model(next_id, cache=expected_cache)
    actual = model(next_id, cache=actual_cache)
    assert bool(mx.allclose(expected, actual, atol=3e-5, rtol=3e-4).item())


@pytest.mark.parametrize("keep", [1, 8, 17, 33])
def test_compiled_verifier_restores_state_across_different_widths(small_target, keep):
    model, ordinary = small_target
    compiled = CompiledQwenTarget(model, (0, 2))
    expected_cache, actual_cache = model.make_cache(), model.make_cache()
    prefix = mx.array([[1, 2, 3]])
    mx.eval(model(prefix, cache=expected_cache), model(prefix, cache=actual_cache))
    block = mx.arange(4, 44).reshape(1, -1)
    expected, _ = ordinary.forward(block, expected_cache, capture=True)
    actual, _ = compiled.forward(block, actual_cache, capture=True)
    assert bool(mx.allclose(expected, actual, atol=3e-5, rtol=3e-4).item())
    ordinary.rollback(expected_cache, keep)
    compiled.rollback(actual_cache, keep)
    for expected, actual in zip(expected_cache, actual_cache, strict=True):
        for a, b in zip(expected.state, actual.state, strict=True):
            assert bool(mx.allclose(a, b, atol=3e-5, rtol=3e-4).item())
    next_block = mx.array([[44, 45, 46]])
    expected, _ = ordinary.forward(next_block, expected_cache, capture=True)
    actual, _ = compiled.forward(next_block, actual_cache, capture=True)
    assert bool(mx.allclose(expected, actual, atol=3e-5, rtol=3e-4).item())


def _macos_before_15() -> bool:
    major = platform.mac_ver()[0].split(".")[0]
    return major.isdigit() and int(major) < 15


@pytest.mark.parametrize(
    "variant",
    ["pad", "fp16_native", "relaxed", "fast", "half_acc", "chunked_half_acc", "affine_fold"],
)
def test_laboratory_projections_have_finite_bounded_errors(variant):
    if variant == "relaxed" and _macos_before_15():
        pytest.skip("Metal math mode `relaxed` requires macOS 15")
    mx.random.seed(42)
    layer = nn.Linear(512, 4096, bias=False)
    layer.set_dtype(mx.bfloat16)
    layer = layer.to_quantized(group_size=64, bits=4)
    x = mx.random.normal((8, 512)).astype(mx.bfloat16)
    expected, actual = layer(x), lab_matmul(layer, x, variant)
    assert bool(mx.all(mx.isfinite(actual)).item())
    error = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    assert error.item() <= 0.02 * mx.max(mx.abs(expected.astype(mx.float32))).item()


def test_lab_install_is_idempotent_and_preserves_single_row():
    model = nn.Sequential(nn.Linear(512, 4096, bias=False))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    x = mx.ones((1, 512), dtype=mx.bfloat16)
    expected = model(x)
    assert len(install_lab(model, "pad")) == 1
    assert install_lab(model, "pad") == []
    assert bool(mx.array_equal(expected, model(x)).item())


@pytest.mark.parametrize("variant", ["pad", "fp16_native", "affine_fold"])
def test_lab_direct_call_preserves_unsupported_quantization(variant, monkeypatch):
    layer = nn.Linear(512, 4096, bias=False)
    layer.set_dtype(mx.bfloat16)
    layer = layer.to_quantized(group_size=32, bits=4)
    x = mx.ones((8, 512), dtype=mx.bfloat16)
    monkeypatch.setattr(
        "agents.architectures.llama.qwen_kernel_lab.experimental_kernel",
        lambda *args: pytest.fail("Unsupported grouping must not construct a kernel"),
    )
    assert bool(mx.array_equal(layer(x), lab_matmul(layer, x, variant)).item())


@pytest.mark.parametrize("source", ["absent", "match match"])
def test_lab_source_rewrites_reject_missing_or_ambiguous_targets(source):
    from agents.architectures.llama.qwen_kernel_lab import _replace_once

    with pytest.raises(ValueError, match="source drift"):
        _replace_once(source, "match", "replacement")


@pytest.mark.parametrize("split_k", [0, 3, 32])
def test_lab_rejects_invalid_split_before_constructing_kernel(split_k, monkeypatch):
    layer = nn.Linear(512, 4096, bias=False)
    layer.set_dtype(mx.bfloat16)
    layer = layer.to_quantized(group_size=64, bits=4)
    monkeypatch.setattr(
        "agents.architectures.llama.qwen_kernel_lab.experimental_kernel",
        lambda *args: pytest.fail("Invalid split must not construct a kernel"),
    )
    with pytest.raises(ValueError, match="split-K"):
        lab_matmul(layer, mx.ones((8, 512), dtype=mx.bfloat16), "fast", split_k=split_k)


def test_benchmark_kernel_conflict_fails_before_loading_weights(monkeypatch, capsys):
    from scripts import benchmark_mlx_dflash as benchmark

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--weights-dir",
            "unused",
            "--drafter-dir",
            "unused",
            "--small-m",
            "--lab-variant",
            "pad",
        ],
    )
    monkeypatch.setattr(
        benchmark, "load", lambda *args: pytest.fail("Must validate before loading")
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert "Select either --small-m or --lab-variant" in capsys.readouterr().err


@pytest.mark.parametrize("rows", [1, 8, 64, 257])
def test_expanded_weights_are_bounded_and_preserve_fallback(rows, monkeypatch):
    model = nn.Module()
    model.mlp = nn.Module()
    model.mlp.gate_proj = nn.Linear(512, 4096, bias=True)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    x = mx.random.normal((rows, 512)).astype(mx.bfloat16)
    original = model.mlp.gate_proj
    expected = original(x)
    with pytest.raises(ValueError, match="needs"):
        install_expanded_mlp(model, budget_gb=0.0001)
    assert model.mlp.gate_proj is original
    wrappers = install_expanded_mlp(model, budget_gb=0.01)
    assert len(wrappers) == 1
    monkeypatch.setattr(mx, "get_active_memory", lambda: 10**15)
    assert install_expanded_mlp(model, budget_gb=0.01) == []
    actual = model.mlp.gate_proj(x)
    assert bool(mx.allclose(expected, actual, atol=0.01, rtol=0.01).item())
    if rows != 64:
        assert bool(mx.array_equal(expected, actual).item())
    wrappers[0].enabled = False
    assert bool(mx.array_equal(expected, model.mlp.gate_proj(x)).item())


@pytest.mark.parametrize("count", [19, 33, 65, 100])
@pytest.mark.parametrize("reject_copy", [False, True])
def test_long_copy_proposals_are_verified_and_cache_is_recoverable(
    small_target, count, reject_copy
):
    model, _ = small_target
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8),
        make_cache=lambda: [],
        project_ctx=lambda h: h,
        append_ctx=lambda *_: None,
        select_block=lambda *_, cap, **kwargs: (mx.zeros((cap,), dtype=mx.int32), None, None),
    )
    decoder = DFlashDecoder(model, SimpleNamespace(eos_token_ids=set()), drafter, copy_window=64)
    prompt = [0] * 16 + ([1] * 64 if reject_copy else [0] * 64) + [0] * 7
    assert list(decoder.generate(prompt, max_tokens=count)) == [0] * count
    assert decoder.last_stats["copy_rounds"] > 0
    if reject_copy:
        assert decoder.last_stats["copy_accepted"] < decoder.last_stats["copy_proposed"]
    expected_cache = model.make_cache()
    mx.eval(model(mx.array([prompt + [0] * (count - 1)]), cache=expected_cache))
    for expected, actual in zip(expected_cache, decoder._cached_state[0], strict=True):
        for a, b in zip(expected.state, actual.state, strict=True):
            assert bool(mx.allclose(a, b, atol=3e-5, rtol=3e-4).item())
    assert list(decoder.generate(prompt + [0] * count + [9], max_tokens=3)) == [0, 0, 0]
    assert decoder.last_stats["cached_prompt_tokens"] == len(prompt) + count - 1


def test_hidden_capture_preserves_target_forward(small_target):
    model, target = small_target
    ids = mx.array([[4, 7, 8, 3]])
    expected = model(ids, cache=model.make_cache())
    actual, hidden = target.forward(ids, model.make_cache(), capture=True)
    assert hidden.shape == (1, 4, 256)
    assert bool(mx.array_equal(expected, actual).item())


@pytest.mark.parametrize("count", [1, 2, 8])
@pytest.mark.parametrize("eos", [False, True])
def test_installed_stream_generate_tail_and_cache_contract(small_target, count, eos):
    from mlx_lm import stream_generate
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    model, _ = small_target
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    tokenizer = TokenizerWrapper(
        SimpleNamespace(
            decode=lambda tokens: "x" * len(tokens),
            clean_up_tokenization_spaces=False,
            eos_token_id=0 if eos else 127,
            get_vocab=lambda: {},
            chat_template=None,
        )
    )
    cache = model.make_cache()
    responses = list(
        stream_generate(model, tokenizer, [1, 2, 3], max_tokens=count, prompt_cache=cache)
    )
    expected = 1 if eos else count
    assert [response.token for response in responses] == [0] * expected
    assert "".join(response.text for response in responses) == ("" if eos else "x" * count)
    assert responses[-1].finish_reason == ("stop" if eos else "length")
    assert all(item.offset == 3 + expected for item in cache if hasattr(item, "offset"))


def test_rollback_rejects_incomplete_attention_trim(small_target, monkeypatch):
    model, target = small_target
    cache = model.make_cache()
    logits, _ = target.forward(mx.array([[1, 2, 3]]), cache, capture=True)
    mx.eval(logits)
    attention = next(item for item in cache if hasattr(item, "offset"))
    monkeypatch.setattr(attention, "trim", lambda count: count - 1)
    with pytest.raises(RuntimeError, match="roll back"):
        target.rollback(cache, 1)


def test_speculative_capture_rejects_padded_recurrent_cache(small_target):
    from agents.architectures.llama.qwen_speculative import captured_delta

    model, _ = small_target
    layer = next(layer for layer in model.model.layers if layer.is_linear)
    cache = model.make_cache()[model.model.ssm_idx]
    cache.left_padding = mx.array([0])
    with pytest.raises(ValueError, match="single sequence"):
        captured_delta(layer.linear_attn, mx.zeros((1, 2, 128)), None, cache)


@pytest.mark.parametrize("rows", [6, 7, 8])
@pytest.mark.parametrize("bits", [4, 8])
def test_small_m_kernel_matches_quantized_projection(rows, bits):
    mx.random.seed(42)
    layer = nn.Linear(512, 4096, bias=False)
    layer.set_dtype(mx.bfloat16)
    quantized = layer.to_quantized(group_size=64, bits=bits)
    x = mx.random.normal((rows, 512)).astype(mx.bfloat16)
    expected, actual = quantized(x), small_m_matmul(quantized, x)
    error = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    scale = mx.max(mx.abs(expected.astype(mx.float32)))
    assert error.item() <= 0.02 * scale.item()


def test_small_m_install_is_idempotent_and_preserves_single_row():
    model = nn.Sequential(nn.Linear(512, 4096, bias=False))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    x = mx.ones((1, 512), dtype=mx.bfloat16)
    expected = model(x)
    wrappers = install_small_m(model)
    assert len(wrappers) == 1
    assert install_small_m(model) == []
    assert bool(mx.array_equal(expected, model(x)).item())


def test_speculative_sampling_rejects_impossible_target_token():
    accepted, replacement = sampled_accept(
        mx.array([[-float("inf"), 0.0], [-float("inf"), 0.0]]),
        mx.array([0]),
        mx.array([[0]]),
        mx.array([[1.0]]),
        1.0,
        1.0,
    )
    assert accepted == 0
    assert replacement == 1


def test_speculative_sampling_accepts_identical_point_distributions():
    accepted, replacement = sampled_accept(
        mx.array([[0.0, -float("inf")], [-float("inf"), 0.0]]),
        mx.array([0]),
        mx.array([[0]]),
        mx.array([[1.0]]),
        1.0,
        1.0,
    )
    assert accepted == 1
    assert replacement == 1


def test_sampled_accept_recovers_target_mass_outside_draft_candidates():
    mx.random.seed(23)
    p = mx.array([0.2, 0.5, 0.3])
    q = mx.array([[0.8, 0.2]])
    logits = mx.stack([mx.log(p), mx.log(p)])
    proposals = mx.random.categorical(mx.log(q[0]), shape=(1024,)).tolist()
    counts = [0, 0, 0]
    for proposal in proposals:
        accepted, replacement = sampled_accept(
            logits, mx.array([proposal]), mx.array([[0, 1]]), q, 1.0, 1.0
        )
        counts[proposal if accepted else replacement] += 1
    assert counts[2] > 0
    for count, probability in zip(counts, [0.2, 0.5, 0.3], strict=True):
        assert abs(count / len(proposals) - probability) < 0.06


@pytest.mark.parametrize("count", [1, 2, 7, 8, 9, 19])
@pytest.mark.parametrize("block_size", [8, 16])
def test_output_limit_inside_accepted_block_retains_exact_prefix(small_target, count, block_size):
    model, _ = small_target
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8),
        make_cache=lambda: [],
        project_ctx=lambda h: h,
        append_ctx=lambda *_: None,
        select_block=lambda *_, cap, **kwargs: (mx.zeros((cap,), dtype=mx.int32), None, None),
    )
    decoder = DFlashDecoder(
        model,
        SimpleNamespace(eos_token_ids=set()),
        drafter,
        max_draft=block_size - 1,
        experimental_block_size=block_size,
    )
    prompt = [1, 2, 3]
    assert list(decoder.generate(prompt, max_tokens=count)) == [0] * count
    assert decoder._cached_tokens == tuple(prompt + [0] * (count - 1))
    expected = model.make_cache()
    mx.eval(model(mx.array([list(decoder._cached_tokens)]), cache=expected))
    actual = decoder._cached_state[0]
    for a, b in zip(expected, actual, strict=True):
        for x, y in zip(a.state, b.state, strict=True):
            assert bool(mx.allclose(x, y, atol=3e-5, rtol=3e-4).item())
    followup = prompt + [0] * count + [4, 5]
    assert list(decoder.generate(followup, max_tokens=3)) == [0, 0, 0]
    assert decoder.last_stats["cached_prompt_tokens"] == len(prompt) + count - 1


def test_generator_close_invalidates_cache(small_target):
    model, _ = small_target
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8),
        make_cache=lambda: [],
    )
    decoder = DFlashDecoder(model, SimpleNamespace(eos_token_ids=set()), drafter)
    generator = decoder.generate([1, 2], max_tokens=10)
    next(generator)
    generator.close()
    assert decoder._cached_state is None
    assert decoder._cached_tokens == ()
    assert decoder.target.records == {}


@pytest.mark.parametrize("cancel", [False, True])
def test_copy_block_eos_or_cancellation_preserves_emitted_prefix(small_target, cancel):
    model, _ = small_target
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8),
        make_cache=lambda: [],
        project_ctx=lambda h: h,
        append_ctx=lambda *_: None,
    )
    decoder = DFlashDecoder(model, SimpleNamespace(eos_token_ids={1}), drafter, copy_window=64)
    forward = decoder.target.forward

    def with_eos(ids, cache, **kwargs):
        logits, hidden = forward(ids, cache, **kwargs)
        if kwargs.get("capture"):
            logits = logits.at[0, 8, 1].add(1)
        return logits, hidden

    decoder.target.forward = with_eos
    prompt = [0] * 80
    stream = decoder.generate(prompt, max_tokens=100)
    if cancel:
        assert [next(stream), next(stream), next(stream)] == [0, 0, 0]
        stream.close()
        assert decoder._cached_state is None
        assert decoder._cached_tokens == ()
    else:
        assert list(stream) == [0] * 9 + [1]
        assert decoder._cached_tokens == tuple(prompt + [0] * 9)
        expected_cache = model.make_cache()
        mx.eval(model(mx.array([list(decoder._cached_tokens)]), cache=expected_cache))
        for expected, actual in zip(expected_cache, decoder._cached_state[0], strict=True):
            for a, b in zip(expected.state, actual.state, strict=True):
                assert bool(mx.allclose(a, b, atol=3e-5, rtol=3e-4).item())
    assert decoder.last_stats["copy_rounds"] == 1
    assert decoder.target.records == {}


def test_copy_proposals_reject_sampling_before_generation(small_target):
    model, _ = small_target
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8)
    )
    decoder = DFlashDecoder(model, SimpleNamespace(eos_token_ids=set()), drafter, copy_window=64)
    with pytest.raises(ValueError, match="greedy"):
        list(decoder.generate([1, 2, 3], max_tokens=10, temperature=0.7))


@pytest.mark.parametrize("count", [1, 3, 5, 16, 41])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_adaptive_transition_preserves_target_and_drafter_cache(
    small_target, count, temperature, monkeypatch
):
    model, _ = small_target
    config = DFlashConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=256,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        max_position_embeddings=4096,
        block_size=8,
        target_layer_ids=(0, 2),
        num_target_layers=4,
        mask_token_id=127,
        selector_rank=8,
        selector_top_k=8,
        conv_kernel_size=2,
        conv_group_size=16,
        layer_types=("sliding_attention", "sliding_attention"),
        sliding_window=16,
    )
    drafter = DFlashDraftModel(config).bind(model)
    drafter.eval()

    def force_fallback(policy, emitted, seconds):
        policy.fallback = True

    monkeypatch.setattr(SpeculationPolicy, "observe_round", force_fallback)
    monkeypatch.setattr("agents.architectures.llama.dflash_backend._DEFERRED_CONTEXT_LIMIT", 8)
    decoder = DFlashDecoder(
        model, SimpleNamespace(eos_token_ids=set()), drafter, prefill_step_size=7, adaptive=True
    )
    prompt = list(range(1, 38))
    output = list(decoder.generate(prompt, max_tokens=count, temperature=temperature))
    expected_cache = model.make_cache()
    mx.eval(model(mx.array([prompt + output[:-1]]), cache=expected_cache))
    actual_cache, draft_cache, previous = decoder._cached_state
    for expected, actual in zip(expected_cache, actual_cache, strict=True):
        for a, b in zip(expected.state, actual.state, strict=True):
            assert bool(mx.allclose(a, b, atol=3e-5, rtol=3e-4).item())
    assert all(
        item.offset + previous.shape[1] == len(decoder._cached_tokens) for item in draft_cache
    )
    if temperature == 0:
        reference_cache = model.make_cache()
        logits = model(mx.array([prompt]), cache=reference_cache)
        expected = [int(mx.argmax(logits[0, -1]).item())]
        for _ in range(count - 1):
            logits = model(mx.array([[expected[-1]]]), cache=reference_cache)
            expected.append(int(mx.argmax(logits[0, -1]).item()))
        assert output == expected
    if count > 12:
        assert decoder.last_stats["fallback_at_token"] is not None
        assert decoder.last_stats["direct_tokens"] > 4
    followup = prompt + output + [9, 8]
    assert len(list(decoder.generate(followup, max_tokens=3))) == 3
    assert decoder.last_stats["cached_prompt_tokens"] == len(prompt) + count - 1


@pytest.mark.parametrize("cancel", [False, True])
def test_adaptive_native_step_respects_eos_and_close(small_target, cancel):
    model, _ = small_target
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8),
        make_cache=lambda: [],
    )
    tokenizer = SimpleNamespace(eos_token_ids=set())
    decoder = DFlashDecoder(model, tokenizer, drafter, adaptive=True, profile=True)
    stream = decoder.generate([1, 2], max_tokens=20)
    assert next(stream) == 0
    assert next(stream) == 0
    if cancel:
        stream.close()
        assert decoder._cached_state is None
    else:
        tokenizer.eos_token_ids.add(0)
        assert list(stream) == []
        assert decoder._cached_tokens == (1, 2, 0)
    assert decoder.target.records == {}
    assert decoder.last_stats["profile_seconds"]["native"] > 0


@pytest.mark.parametrize("profile", [False, True])
def test_profiled_fallback_fences_hidden_state_without_recalibration(
    small_target, monkeypatch, profile
):
    model, _ = small_target
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    initialize = SpeculationPolicy.__init__

    def fallback_only(policy):
        initialize(policy)
        policy.native_seconds = [1.0] * 4
        policy.fallback = True

    monkeypatch.setattr(SpeculationPolicy, "__init__", fallback_only)
    monkeypatch.setattr(
        SpeculationPolicy,
        "observe_native",
        lambda *args: pytest.fail("fallback must not recalibrate"),
    )
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8),
        make_cache=lambda: [],
    )
    decoder = DFlashDecoder(
        model, SimpleNamespace(eos_token_ids=set()), drafter, adaptive=True, profile=profile
    )
    evaluate = mx.eval
    fences = []

    def record_eval(*args, **kwargs):
        if len(args) == 2 and getattr(args[0], "shape", None) == (1, 1, 256):
            fences.append(True)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", record_eval)
    assert list(decoder.generate([1, 2], max_tokens=4)) == [0] * 4
    assert len(fences) == (3 if profile else 0)


def test_real_dflash_stream_survives_consumer_worker_changes(small_target):
    model, _ = small_target
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0, 2), mask_token_id=127, block_size=8),
        make_cache=lambda: [],
        project_ctx=lambda h: h,
        append_ctx=lambda *_: None,
        select_block=lambda *_, cap, **kwargs: (mx.zeros((cap,), dtype=mx.int32), None, None),
    )
    runner = MLXLlamaRunner()

    def load():
        target = TextModel(model.args)
        target.eval()
        target.lm_head.weight = mx.zeros_like(target.lm_head.weight)
        return DFlashDecoder(target, SimpleNamespace(eos_token_ids=set()), drafter)

    with runner._request_lock:
        decoder = runner._on_worker(load)

    def events(*args):
        with closing(decoder.generate([1, 2], max_tokens=4)) as tokens:
            for token in tokens:
                yield ModelEvent.text_delta(str(token))

    runner._stream_model_turn = events
    stream = runner.stream_model_turn([], [], ModelRequestConfig())
    try:
        with (
            ThreadPoolExecutor(max_workers=1) as first,
            ThreadPoolExecutor(max_workers=1) as second,
        ):
            assert first.submit(next, stream).result(timeout=10).text == "0"
            assert second.submit(next, stream).result(timeout=10).text == "0"
            second.submit(stream.close).result(timeout=10)
        assert decoder._cached_state is None
        assert len(list(runner.stream_model_turn([], [], ModelRequestConfig()))) == 4
    finally:
        stream.close()
        runner.cleanup()


@pytest.mark.parametrize("top_k", [0, 1, 2, 3])
def test_nucleus_temperature_order_matches_mlx_lm(top_k, monkeypatch):
    from mlx_lm import sample_utils

    logits = mx.array([[1.0, 1.5, -1.0, 0.0]])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    captured = []

    def capture(logits, temp):
        captured.append(mx.softmax(logits / temp, axis=-1))
        return mx.argmax(logits, axis=-1)

    monkeypatch.setattr(sample_utils, "categorical_sampling", capture)
    sample_utils.make_sampler(temp=0.7, top_p=0.8, top_k=top_k)(logprobs)
    expected = captured[0]
    assert bool(mx.array_equal(target_probabilities(logits, 0.7, 0.8, top_k), expected).item())


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("strategy", ["local", "prefix", "sequence"])
@pytest.mark.parametrize("copying", [False, True])
def test_chunked_prefill_and_rotating_draft_cache_match_greedy_target(
    small_target, profile, strategy, copying, monkeypatch
):
    model, _ = small_target
    config = DFlashConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=256,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        max_position_embeddings=4096,
        block_size=8,
        target_layer_ids=(0, 2),
        num_target_layers=4,
        mask_token_id=127,
        selector_rank=8,
        selector_top_k=8,
        conv_kernel_size=2,
        conv_group_size=16,
        layer_types=("sliding_attention", "sliding_attention"),
        sliding_window=16,
    )
    drafter = DFlashDraftModel(config).bind(model)
    drafter.greedy_strategy = strategy
    drafter.eval()
    prompt = list(range(1, 38))
    if copying:
        monkeypatch.setattr(
            "agents.architectures.llama.ngram_draft.NgramDraft.propose",
            lambda self, limit, minimum: [0] * limit,
        )
    decoder = DFlashDecoder(
        model,
        SimpleNamespace(eos_token_ids=set()),
        drafter,
        prefill_step_size=7,
        profile=profile,
        copy_window=64 if copying else 0,
    )
    actual = list(decoder.generate(prompt, max_tokens=19))
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    expected = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(18):
        logits = model(mx.array([[expected[-1]]]), cache=cache)
        expected.append(int(mx.argmax(logits[0, -1]).item()))
    assert actual == expected
    assert decoder.last_stats["verify_rounds"] > 1
    assert (decoder.last_stats["copy_rounds"] > 0) is copying
    assert ("profile_seconds" in decoder.last_stats) is profile


@pytest.mark.parametrize(
    "split_k,column_tiles", [(1, 1), (2, 1), (4, 1), (8, 1), (2, 2), (4, 4), (16, 1)]
)
def test_kernel_layout_variants_are_numerically_bounded(split_k, column_tiles):
    mx.random.seed(17)
    layer = nn.Linear(1024, 4104, bias=True)
    layer.set_dtype(mx.bfloat16)
    layer = layer.to_quantized(group_size=64, bits=4)
    x = mx.random.normal((2, 4, 1024)).astype(mx.bfloat16)
    expected = layer(x).astype(mx.float32)
    actual = small_m_matmul(layer, x, split_k, column_tiles=column_tiles).astype(mx.float32)
    assert mx.max(mx.abs(expected - actual)).item() <= 0.02 * mx.max(mx.abs(expected)).item()


@pytest.mark.parametrize("rows", [6, 8, 9, 16, 24, 32])
@pytest.mark.parametrize("bits", [4, 8])
def test_packed_wide_kernel_preserves_unpacked_values(rows, bits):
    mx.random.seed(31)
    layer = nn.Linear(1024, 4104, bias=False)
    layer.set_dtype(mx.bfloat16)
    layer = layer.to_quantized(group_size=64, bits=bits)
    x = mx.random.normal((1, rows, 1024)).astype(mx.bfloat16)
    unpacked = small_m_matmul(layer, x)
    packed = small_m_matmul(layer, x, packed_weights=pack_small_m_weights(layer))
    assert bool(mx.array_equal(unpacked, packed).item())
    native = layer(x).astype(mx.float32)
    assert (
        mx.max(mx.abs(packed.astype(mx.float32) - native)).item()
        < 0.02 * mx.max(mx.abs(native)).item()
    )


@pytest.mark.parametrize("rows,bits", [(6, 4), (8, 4), (16, 4), (32, 4), (8, 8), (16, 8)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_direct_fragments_preserve_staged_mma(rows, bits, shuffle):
    mx.random.seed(47)
    layer = nn.Linear(1024, 4104, bias=False)
    layer.set_dtype(mx.bfloat16)
    layer = layer.to_quantized(group_size=64, bits=bits)
    x = mx.random.normal((1, rows, 1024)).astype(mx.bfloat16)
    staged = small_m_matmul(layer, x)
    direct = small_m_matmul(layer, x, direct_fragment=True, shuffle_fragments=shuffle)
    packed = small_m_matmul(
        layer,
        x,
        packed_weights=pack_small_m_weights(layer),
        direct_fragment=True,
        shuffle_fragments=shuffle,
    )
    assert bool(mx.array_equal(staged, direct).item())
    assert bool(mx.array_equal(direct, packed).item())


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_batch,split_k", [(1, 5), (2, 5), (4, 5), (1, 10), (2, 10)])
def test_group_batch_preserves_staged_mma(bits, group_batch, split_k):
    mx.random.seed(81)
    layer = nn.QuantizedLinear.from_linear(
        nn.Linear(5120, 128, bias=False), group_size=64, bits=bits
    )
    layer.scales = layer.scales.astype(mx.bfloat16)
    layer.biases = layer.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, 8, 5120)).astype(mx.bfloat16)
    expected = small_m_matmul(layer, x, split_k)
    actual = small_m_matmul(layer, x, split_k, group_batch=group_batch)
    assert bool(mx.array_equal(expected, actual).item())


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("stage_stride", [9, 10, 12, 16])
def test_padding_preserves_staged_mma(bits, stage_stride):
    mx.random.seed(83)
    layer = nn.QuantizedLinear.from_linear(
        nn.Linear(5120, 128, bias=False), group_size=64, bits=bits
    )
    layer.scales = layer.scales.astype(mx.bfloat16)
    layer.biases = layer.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, 8, 5120)).astype(mx.bfloat16)
    expected = small_m_matmul(layer, x)
    actual = small_m_matmul(layer, x, stage_stride=stage_stride)
    assert bool(mx.array_equal(expected, actual).item())


@pytest.mark.parametrize("rows", [6, 8, 16])
def test_lookup_preserves_dequantized_values(rows):
    mx.random.seed(87)
    layer = nn.QuantizedLinear.from_linear(nn.Linear(5120, 128, bias=False), group_size=64, bits=4)
    layer.scales = layer.scales.astype(mx.bfloat16)
    layer.biases = layer.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, rows, 5120)).astype(mx.bfloat16)
    expected = small_m_matmul(layer, x)
    actual = small_m_matmul(layer, x, dequant_table=make_dequant_table(layer))
    assert bool(mx.array_equal(expected, actual).item())


@pytest.mark.parametrize("objective", ["prefix", "sequence"])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_selector_lookahead_matches_exhaustive_search(objective, width):
    selector = CandidateSelector(8, 32, 4, 3)
    scores = mx.random.normal((width, 3, 3), key=mx.random.key(43))
    ids = mx.arange(width * 3).reshape(width, 3)
    probabilities = mx.softmax(scores, axis=-1).tolist()

    def value(path):
        previous, product, total = 0, 1.0, 0.0
        for slot, selected in enumerate(path):
            product *= probabilities[slot][previous][selected]
            total += product
            previous = selected
        return total if objective == "prefix" else product

    expected = max(itertools.product(range(3), repeat=width), key=value)
    actual = selector.walk_lookahead(scores, ids, objective=objective).tolist()
    assert actual == [slot * 3 + selected for slot, selected in enumerate(expected)]


def test_tuner_measures_serial_chain_and_preserves_decode():
    model = nn.Sequential(nn.Linear(512, 4096, bias=False))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=64)
    wrappers = install_small_m(model)
    single = mx.ones((1, 1, 512), dtype=mx.bfloat16)
    expected = wrappers[0].linear(single)
    results = tune_small_m(wrappers)
    assert len(results) == 1
    assert results[0]["dependent_chain"] is True
    assert results[0]["split_k"] in (0, 2, 4, 8)
    assert bool(mx.array_equal(model(single), expected).item())
