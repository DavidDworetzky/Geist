"""Unit tests for the generic Transformers causal-LM runner."""

import os
from collections import UserDict
from unittest.mock import MagicMock, patch

import pytest
import torch

from agents.architectures.base_runner import GenerationConfig
from agents.architectures.transformers_runner import TransformersRunner


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.chat_template = "{{ messages }}"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.decode.return_value = "fast response"
    return tokenizer


def _model():
    model = MagicMock()
    model.to.return_value = model
    model.device = torch.device("cpu")
    model.hf_device_map = None
    model.generate.return_value = torch.tensor([[1, 2, 3, 8, 9]])
    return model


@patch("agents.architectures.transformers_runner.importlib.util.find_spec", return_value=None)
@patch("agents.architectures.transformers_runner.AutoModelForCausalLM")
@patch("agents.architectures.transformers_runner.AutoTokenizer")
@patch("agents.architectures.transformers_runner.AutoConfig")
def test_load_and_generate_uses_direct_suffix_decode(
    config_cls, tokenizer_cls, model_cls, _find_spec
):
    config_cls.from_pretrained.return_value = MagicMock(
        architectures=["Qwen2ForCausalLM"], max_position_embeddings=64
    )
    tokenizer = _tokenizer()
    model = _model()
    tokenizer_cls.from_pretrained.return_value = tokenizer
    model_cls.from_pretrained.return_value = model

    runner = TransformersRunner()
    runner.load("Qwen/Qwen2.5-3B-Instruct", {"device": "cpu"})
    result = runner.complete(
        "Be helpful", "hello", GenerationConfig(max_tokens=100, temperature=0.0)
    )

    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once()
    model.generate.assert_called_once()
    assert model.generate.call_args.kwargs["do_sample"] is False
    assert model.generate.call_args.kwargs["use_cache"] is True
    assert model.generate.call_args.kwargs["max_new_tokens"] == 61
    tokenizer.decode.assert_called_once()
    decoded_ids = tokenizer.decode.call_args.args[0]
    assert decoded_ids.tolist() == [8, 9]
    assert result[1]["content"] == "fast response"


def test_complete_messages_preserves_structured_conversation_roles():
    runner = TransformersRunner()
    runner.model_id = "test/model"
    runner.model = _model()
    runner.tokenizer = _tokenizer()
    runner.config = MagicMock(max_position_embeddings=64)
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Remember cobalt."},
        {"role": "assistant", "content": "I will remember cobalt."},
        {"role": "user", "content": "What should you remember?"},
    ]

    runner.complete_messages(
        messages,
        GenerationConfig(max_tokens=8, temperature=0.0),
    )

    applied_messages = runner.tokenizer.apply_chat_template.call_args.args[0]
    assert applied_messages == messages


@patch("agents.architectures.transformers_runner.importlib.util.find_spec", return_value=None)
@patch("agents.architectures.transformers_runner.AutoModelForCausalLM")
@patch("agents.architectures.transformers_runner.AutoTokenizer")
@patch("agents.architectures.transformers_runner.AutoConfig")
def test_default_llama_uses_existing_legacy_weights_directory(
    config_cls, tokenizer_cls, model_cls, _find_spec, tmp_path, monkeypatch
):
    weights_dir = tmp_path / "app" / "model_weights" / "llama_3_1"
    weights_dir.mkdir(parents=True)
    (weights_dir / "config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    config_cls.from_pretrained.return_value = MagicMock(architectures=["LlamaForCausalLM"])
    tokenizer_cls.from_pretrained.return_value = _tokenizer()
    model_cls.from_pretrained.return_value = _model()

    TransformersRunner().load(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        {"device": "cpu"},
    )

    expected_source = os.path.join("app", "model_weights", "llama_3_1")
    config_cls.from_pretrained.assert_called_once_with(
        expected_source,
        trust_remote_code=False,
    )


@patch("agents.architectures.transformers_runner.importlib.util.find_spec", return_value=None)
@patch("agents.architectures.transformers_runner.AutoModelForCausalLM")
@patch("agents.architectures.transformers_runner.AutoTokenizer")
@patch("agents.architectures.transformers_runner.AutoConfig")
def test_revision_is_shared_by_config_tokenizer_and_model(
    config_cls, tokenizer_cls, model_cls, _find_spec
):
    config_cls.from_pretrained.return_value = MagicMock(architectures=["Phi3ForCausalLM"])
    tokenizer_cls.from_pretrained.return_value = _tokenizer()
    model_cls.from_pretrained.return_value = _model()

    TransformersRunner().load(
        "microsoft/Phi-4-mini-instruct",
        {"device": "cpu", "revision": "immutable-sha"},
    )

    for loader in (config_cls, tokenizer_cls, model_cls):
        assert loader.from_pretrained.call_args.kwargs["revision"] == "immutable-sha"


def test_mapping_like_batch_encoding_and_system_role_fallback():
    runner = TransformersRunner()
    runner.model_id = "google/gemma-example"
    runner.model = _model()
    runner.tokenizer = _tokenizer()
    runner.config = MagicMock(max_position_embeddings=64)
    encoded = UserDict(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )
    runner.tokenizer.apply_chat_template.side_effect = [
        ValueError("system role unsupported"),
        encoded,
    ]

    runner.complete("Be concise", "hello", GenerationConfig(max_tokens=4, temperature=0.0))

    fallback_messages = runner.tokenizer.apply_chat_template.call_args.args[0]
    assert fallback_messages == [{"role": "user", "content": "Be concise\n\nhello"}]
    kwargs = runner.model.generate.call_args.kwargs
    assert {"input_ids", "attention_mask"}.issubset(kwargs)


def test_multiple_stop_sequences_use_the_earliest_match():
    runner = TransformersRunner()
    runner.model_id = "test/model"
    runner.model = _model()
    runner.tokenizer = _tokenizer()
    runner.tokenizer.decode.return_value = "first END trailing STOP ignored"
    runner.config = MagicMock(max_position_embeddings=64)

    result = runner.complete(
        "", "hello", GenerationConfig(max_tokens=8, temperature=0.0, stop=["STOP", "END"])
    )

    assert result[1]["content"] == "first"


@patch("agents.architectures.transformers_runner.importlib.util.find_spec")
@patch("agents.architectures.transformers_runner.AutoModelForCausalLM")
@patch("agents.architectures.transformers_runner.AutoTokenizer")
@patch("agents.architectures.transformers_runner.AutoConfig")
def test_cuda_uses_accelerate_without_duplicate_device_copy(
    config_cls, tokenizer_cls, model_cls, find_spec
):
    find_spec.return_value = MagicMock()
    config_cls.from_pretrained.return_value = MagicMock(architectures=["Qwen2ForCausalLM"])
    tokenizer_cls.from_pretrained.return_value = _tokenizer()
    model = _model()
    model.hf_device_map = {"": 0}
    model_cls.from_pretrained.return_value = model

    with patch.object(TransformersRunner, "_select_device", return_value=torch.device("cuda")):
        TransformersRunner().load("Qwen/Qwen2.5-3B-Instruct")

    kwargs = model_cls.from_pretrained.call_args.kwargs
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["device_map"] == "auto"
    model.to.assert_not_called()


@patch("agents.architectures.transformers_runner.importlib.util.find_spec", return_value=None)
@patch("agents.architectures.transformers_runner.AutoModelForCausalLM")
@patch("agents.architectures.transformers_runner.AutoTokenizer")
@patch("agents.architectures.transformers_runner.AutoConfig")
def test_mps_defaults_to_stable_eager_attention(config_cls, tokenizer_cls, model_cls, _find_spec):
    config_cls.from_pretrained.return_value = MagicMock(architectures=["SmolLMForCausalLM"])
    tokenizer_cls.from_pretrained.return_value = _tokenizer()
    model_cls.from_pretrained.return_value = _model()

    TransformersRunner().load(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        {"device": "mps"},
    )

    assert model_cls.from_pretrained.call_args.kwargs["attn_implementation"] == "eager"


@patch("agents.architectures.transformers_runner.metadata.version", return_value="4.48.0")
def test_minimum_transformers_version_fails_early(_version):
    runner = TransformersRunner()
    with pytest.raises(RuntimeError, match="transformers>=4.57.0"):
        runner.load("allenai/Olmo-3-7B-Instruct", {"device": "cpu"})


def test_server_backed_model_fails_before_loading():
    runner = TransformersRunner()
    with pytest.raises(ValueError, match="server-backed"):
        runner.load("kimi-k2.5")


@patch("agents.architectures.transformers_runner.importlib.util.find_spec", return_value=None)
@patch("agents.architectures.transformers_runner.AutoModelForCausalLM")
@patch("agents.architectures.transformers_runner.AutoTokenizer")
@patch("agents.architectures.transformers_runner.AutoConfig")
def test_explicit_override_can_load_server_backed_text_model(
    config_cls, tokenizer_cls, model_cls, _find_spec
):
    config_cls.from_pretrained.return_value = MagicMock(architectures=["Glm4MoeLiteForCausalLM"])
    tokenizer_cls.from_pretrained.return_value = _tokenizer()
    model_cls.from_pretrained.return_value = _model()

    TransformersRunner().load(
        "zai-org/GLM-4.7-Flash",
        {"device": "cpu", "allow_server_backed": True},
    )

    model_cls.from_pretrained.assert_called_once()


@patch("agents.architectures.transformers_runner.importlib.util.find_spec", return_value=None)
@patch("agents.architectures.transformers_runner.AutoTokenizer")
@patch("agents.architectures.transformers_runner.AutoConfig")
def test_multimodal_architecture_is_not_misrouted(config_cls, _tokenizer, _find_spec):
    config_cls.from_pretrained.return_value = MagicMock(
        architectures=["Mistral3ForConditionalGeneration"]
    )
    runner = TransformersRunner()
    with pytest.raises(ValueError, match="multimodal"):
        runner.load("some-org/multimodal-model", {"device": "cpu"})
