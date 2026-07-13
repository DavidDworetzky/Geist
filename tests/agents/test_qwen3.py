"""
Tests for Qwen 3 runner, model loading paths, inference, and factory integration.

Covers:
- Qwen3Runner loading strategies (hub, local pretrained, raw safetensors)
- Pipeline caching and cleanup
- Inference via complete() and generate()
- Chat template formatting
- Factory auto-detection of Qwen 3 models
- Factory weights_dir propagation
"""
import sys
import os
import json
import types
import pytest
from unittest.mock import Mock, MagicMock, patch, call

# ---------------------------------------------------------------------------
# Mock out MLX before any project imports — MLX is Apple-Silicon-only and the
# agents.architectures package transitively imports it via the llama runner.
#
# Requirements for the mock:
# 1. __spec__ must be a real ModuleSpec (importlib.util.find_spec checks it)
# 2. Submodules must be importable (from mlx.utils import tree_unflatten)
# 3. Attributes accessed at import time (mx.array, nn.Module, etc.) must not crash
# ---------------------------------------------------------------------------
import importlib
_MLX_SUBMODULES = ("mlx", "mlx.core", "mlx.core.random", "mlx.nn", "mlx.utils")
for _mod_name in _MLX_SUBMODULES:
    if _mod_name not in sys.modules:
        _mock = MagicMock()
        _mock.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None)
        _mock.__path__ = []
        _mock.__file__ = None
        _mock.__name__ = _mod_name
        _mock.__package__ = _mod_name
        sys.modules[_mod_name] = _mock

from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.architectures.registry import register_runner, get_runner, clear_registry
from agents.factory import AgentFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generation_config(**overrides):
    defaults = dict(max_tokens=64, temperature=0.7, top_p=0.9)
    defaults.update(overrides)
    return GenerationConfig(**defaults)


def _mock_tokenizer(has_chat_template=True):
    tok = MagicMock()
    tok.eos_token_id = 2
    if has_chat_template:
        tok.apply_chat_template = MagicMock(
            return_value="<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        del tok.apply_chat_template  # hasattr will return False
    return tok


def _mock_model():
    model = MagicMock()
    model.num_parameters.return_value = 8_000_000_000
    model.to.return_value = model
    return model


# ---------------------------------------------------------------------------
# Qwen3Runner: loading strategies
# ---------------------------------------------------------------------------

class TestQwen3RunnerLoadHub:
    """Loading from HuggingFace Hub (no local files)."""

    @patch("agents.architectures.vllm_runner.AutoModelForCausalLM")
    @patch("agents.architectures.vllm_runner.AutoTokenizer")
    @patch("agents.architectures.vllm_runner.os.path.exists", return_value=False)
    def test_load_from_hub(self, mock_exists, mock_tok_cls, mock_model_cls):
        from agents.architectures.qwen3_runner import Qwen3Runner

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        runner = Qwen3Runner()
        runner.load("Qwen/Qwen3-8B")

        # Should call from_pretrained with the model ID (not a local path)
        mock_tok_cls.from_pretrained.assert_called_once_with("Qwen/Qwen3-8B")
        mock_model_cls.from_pretrained.assert_called_once()
        assert runner.model is not None
        assert runner.tokenizer is not None

    @patch("agents.architectures.vllm_runner.login")
    @patch("agents.architectures.vllm_runner.AutoModelForCausalLM")
    @patch("agents.architectures.vllm_runner.AutoTokenizer")
    @patch("agents.architectures.vllm_runner.os.path.exists", return_value=False)
    def test_hub_login_with_token(self, mock_exists, mock_tok_cls, mock_model_cls, mock_login):
        from agents.architectures.qwen3_runner import Qwen3Runner

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        with patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "hf_test_token"}):
            runner = Qwen3Runner()
            runner.load("Qwen/Qwen3-4B")

        mock_login.assert_called_once_with(token="hf_test_token")

    @patch("agents.architectures.vllm_runner.login")
    @patch("agents.architectures.vllm_runner.AutoModelForCausalLM")
    @patch("agents.architectures.vllm_runner.AutoTokenizer")
    @patch("agents.architectures.vllm_runner.os.path.exists", return_value=False)
    def test_hub_no_login_without_token(self, mock_exists, mock_tok_cls, mock_model_cls, mock_login):
        from agents.architectures.qwen3_runner import Qwen3Runner

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        with patch.dict(os.environ, {}, clear=True):
            runner = Qwen3Runner()
            runner.load("Qwen/Qwen3-4B")

        mock_login.assert_not_called()


class TestQwen3RunnerLoadLocal:
    """Loading from a standard HuggingFace pretrained directory."""

    def _exists_side_effect(self, weights_dir):
        """Return True for config.json and model.safetensors.index.json."""
        hf_index_path = os.path.join(weights_dir, "model.safetensors.index.json")
        config_path = os.path.join(weights_dir, "config.json")
        def _side(path):
            return path in (config_path, hf_index_path)
        return _side

    @patch("agents.architectures.vllm_runner.AutoModelForCausalLM")
    @patch("agents.architectures.vllm_runner.AutoTokenizer")
    def test_load_from_local_pretrained(self, mock_tok_cls, mock_model_cls):
        from agents.architectures.qwen3_runner import Qwen3Runner

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        weights_dir = "app/model_weights/Qwen_Qwen3-8B"

        with patch("agents.architectures.vllm_runner.os.path.exists",
                    side_effect=self._exists_side_effect(weights_dir)):
            runner = Qwen3Runner()
            runner.load("Qwen/Qwen3-8B")

        # Should load from local weights_dir, not from hub
        mock_tok_cls.from_pretrained.assert_called_once_with(weights_dir)
        first_arg = mock_model_cls.from_pretrained.call_args[0][0]
        assert first_arg == weights_dir


class TestQwen3RunnerLoadSafetensors:
    """Loading from raw safetensors files (config.json + *.safetensors)."""

    @patch("agents.architectures.vllm_runner.safetensors.torch.load_file")
    @patch("agents.architectures.vllm_runner.AutoModelForCausalLM")
    @patch("agents.architectures.vllm_runner.AutoConfig")
    @patch("agents.architectures.vllm_runner.AutoTokenizer")
    @patch("agents.architectures.vllm_runner.glob.glob")
    def test_load_from_safetensors(self, mock_glob, mock_tok_cls, mock_config_cls,
                                    mock_model_cls, mock_load_file):
        from agents.architectures.qwen3_runner import Qwen3Runner

        weights_dir = "/data/qwen3-weights"
        st_files = [
            os.path.join(weights_dir, "model-00001-of-00004.safetensors"),
            os.path.join(weights_dir, "model-00002-of-00004.safetensors"),
        ]
        mock_glob.return_value = st_files

        # config.json exists but NO model.safetensors.index.json
        def _exists(path):
            return path == os.path.join(weights_dir, "config.json") or \
                   path == os.path.join(weights_dir, "tokenizer.json")

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_config = MagicMock()
        mock_config_cls.from_pretrained.return_value = mock_config
        mock_model = _mock_model()
        mock_model.load_state_dict.return_value = ([], [])  # (missing, unexpected)
        mock_model_cls.from_config.return_value = mock_model
        mock_load_file.return_value = {"layer.weight": MagicMock()}

        config_data = {"torch_dtype": "bfloat16", "model_type": "qwen3"}

        with patch("agents.architectures.vllm_runner.os.path.exists", side_effect=_exists), \
             patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = Mock(return_value=False)
            mock_open.return_value.read = Mock(return_value=json.dumps(config_data))
            # json.load needs a file object
            with patch("agents.architectures.vllm_runner.json.load", return_value=config_data):
                runner = Qwen3Runner()
                runner.load("Qwen/Qwen3-8B", device_config={"weights_dir": weights_dir})

        # Should use from_config (not from_pretrained) for raw safetensors
        mock_model_cls.from_config.assert_called_once_with(mock_config)
        mock_model_cls.from_pretrained.assert_not_called()

        # Should have called load_file for each safetensors file
        assert mock_load_file.call_count == 2
        mock_model.load_state_dict.assert_called_once()

    @patch("agents.architectures.vllm_runner.safetensors.torch.load_file")
    @patch("agents.architectures.vllm_runner.AutoModelForCausalLM")
    @patch("agents.architectures.vllm_runner.AutoConfig")
    @patch("agents.architectures.vllm_runner.AutoTokenizer")
    @patch("agents.architectures.vllm_runner.glob.glob")
    def test_safetensors_tokenizer_fallback_to_hub(self, mock_glob, mock_tok_cls,
                                                    mock_config_cls, mock_model_cls,
                                                    mock_load_file):
        from agents.architectures.qwen3_runner import Qwen3Runner

        weights_dir = "/data/weights"
        mock_glob.return_value = [os.path.join(weights_dir, "model.safetensors")]

        # Only config.json exists, NO tokenizer files
        def _exists(path):
            return path == os.path.join(weights_dir, "config.json")

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_config_cls.from_pretrained.return_value = MagicMock()
        mock_model = _mock_model()
        mock_model.load_state_dict.return_value = ([], [])
        mock_model_cls.from_config.return_value = mock_model
        mock_load_file.return_value = {}

        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with patch("agents.architectures.vllm_runner.os.path.exists", side_effect=_exists), \
             patch("builtins.open", return_value=mock_file), \
             patch("agents.architectures.vllm_runner.json.load", return_value={}):
            runner = Qwen3Runner()
            runner.load("Qwen/Qwen3-8B", device_config={"weights_dir": weights_dir})

        # Tokenizer should fall back to hub model ID
        mock_tok_cls.from_pretrained.assert_called_once_with("Qwen/Qwen3-8B")


# ---------------------------------------------------------------------------
# Qwen3Runner: device selection
# ---------------------------------------------------------------------------

class TestQwen3RunnerDevice:

    @patch("agents.architectures.vllm_runner.AutoModelForCausalLM")
    @patch("agents.architectures.vllm_runner.AutoTokenizer")
    @patch("agents.architectures.vllm_runner.os.path.exists", return_value=False)
    def test_explicit_device_config(self, mock_exists, mock_tok_cls, mock_model_cls):
        from agents.architectures.qwen3_runner import Qwen3Runner
        import torch

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        runner = Qwen3Runner()
        runner.load("Qwen/Qwen3-8B", device_config={"device": "cpu"})

        assert runner.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Qwen3Runner: inference (complete / generate)
# ---------------------------------------------------------------------------

class TestQwen3RunnerInference:

    def _create_loaded_runner(self):
        """Create a Qwen3Runner with mocked model/tokenizer already loaded."""
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.model = _mock_model()
        runner.tokenizer = _mock_tokenizer()
        runner.device = "cpu"
        runner.model_id = "Qwen/Qwen3-8B"
        return runner

    @patch("agents.architectures.vllm_runner.transformers.pipeline")
    def test_complete_with_chat_template(self, mock_pipeline_fn):
        runner = self._create_loaded_runner()
        config = _make_generation_config()

        prompt_text = runner.tokenizer.apply_chat_template.return_value
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": prompt_text + "Hello world!"}]
        mock_pipeline_fn.return_value = mock_pipe

        result = runner.complete("You are helpful.", "Say hello", config)

        # Should use tokenizer's chat template
        runner.tokenizer.apply_chat_template.assert_called_once()
        template_call = runner.tokenizer.apply_chat_template.call_args
        messages = template_call[0][0]
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Say hello"}

        # Result should be in strings_to_message_dict format
        assert isinstance(result, list)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Say hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hello world!"

    @patch("agents.architectures.vllm_runner.transformers.pipeline")
    def test_complete_without_system_prompt(self, mock_pipeline_fn):
        runner = self._create_loaded_runner()
        config = _make_generation_config()

        prompt_text = runner.tokenizer.apply_chat_template.return_value
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": prompt_text + "Response"}]
        mock_pipeline_fn.return_value = mock_pipe

        runner.complete("", "Just a user message", config)

        # No system message in the template call
        messages = runner.tokenizer.apply_chat_template.call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("agents.architectures.vllm_runner.transformers.pipeline")
    def test_complete_strips_chat_markers(self, mock_pipeline_fn):
        runner = self._create_loaded_runner()
        config = _make_generation_config()

        prompt_text = runner.tokenizer.apply_chat_template.return_value
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"generated_text": prompt_text + "Clean response<|im_end|>extra junk"}
        ]
        mock_pipeline_fn.return_value = mock_pipe

        result = runner.complete("", "test", config)
        assert result[1]["content"] == "Clean response"

    @patch("agents.architectures.vllm_runner.transformers.pipeline")
    def test_generate_delegates_to_complete(self, mock_pipeline_fn):
        runner = self._create_loaded_runner()
        config = _make_generation_config()

        prompt_text = runner.tokenizer.apply_chat_template.return_value
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": prompt_text + "Generated!"}]
        mock_pipeline_fn.return_value = mock_pipe

        result = runner.generate("prompt text", config)

        # generate() should produce same format as complete()
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Generated!"

    def test_generate_raises_when_not_loaded(self):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        config = _make_generation_config()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.generate("test", config)

    def test_complete_raises_when_not_loaded(self):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        config = _make_generation_config()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.complete("sys", "usr", config)

    @patch("agents.architectures.vllm_runner.transformers.pipeline")
    def test_complete_respects_temperature_zero(self, mock_pipeline_fn):
        """Temperature 0 should set do_sample=False and temperature=None."""
        runner = self._create_loaded_runner()
        config = _make_generation_config(temperature=0.0)

        prompt_text = runner.tokenizer.apply_chat_template.return_value
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": prompt_text + "deterministic"}]
        mock_pipeline_fn.return_value = mock_pipe

        runner.complete("", "test", config)

        pipe_call_kwargs = mock_pipe.call_args
        # The pipeline is called as pipeline(prompt, **kwargs)
        kwargs = pipe_call_kwargs[1] if pipe_call_kwargs[1] else {}
        assert kwargs.get("do_sample") is False
        assert kwargs.get("temperature") is None


# ---------------------------------------------------------------------------
# Qwen3Runner: pipeline caching
# ---------------------------------------------------------------------------

class TestQwen3RunnerPipelineCaching:

    @patch("agents.architectures.vllm_runner.transformers.pipeline")
    def test_pipeline_created_once(self, mock_pipeline_fn):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.model = _mock_model()
        runner.tokenizer = _mock_tokenizer()
        runner.device = "cpu"
        runner.model_id = "Qwen/Qwen3-8B"

        prompt_text = runner.tokenizer.apply_chat_template.return_value
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": prompt_text + "r1"}]
        mock_pipeline_fn.return_value = mock_pipe

        config = _make_generation_config()

        # Call complete twice
        runner.complete("", "first", config)
        runner.complete("", "second", config)

        # Pipeline constructor should only be called once
        mock_pipeline_fn.assert_called_once()
        # But the pipeline itself should be called twice
        assert mock_pipe.call_count == 2


# ---------------------------------------------------------------------------
# Qwen3Runner: cleanup
# ---------------------------------------------------------------------------

class TestQwen3RunnerCleanup:

    def test_cleanup_releases_resources(self):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.model = _mock_model()
        runner.tokenizer = _mock_tokenizer()
        runner._pipeline = MagicMock()

        runner.cleanup()

        assert runner.model is None
        assert runner.tokenizer is None
        assert runner._pipeline is None

    def test_cleanup_on_fresh_runner(self):
        """Cleanup on a never-loaded runner should not raise."""
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.cleanup()  # should be a no-op


# ---------------------------------------------------------------------------
# Qwen3Runner: ChatML fallback
# ---------------------------------------------------------------------------

class TestQwen3RunnerChatMLFallback:

    @patch("agents.architectures.vllm_runner.transformers.pipeline")
    def test_fallback_chatml_format(self, mock_pipeline_fn):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.model = _mock_model()
        runner.tokenizer = _mock_tokenizer(has_chat_template=False)
        runner.device = "cpu"
        runner.model_id = "Qwen/Qwen3-8B"

        mock_pipe = MagicMock()
        # The fallback will produce a ChatML-formatted prompt
        expected_prefix = (
            "<|im_start|>system\nBe helpful<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        mock_pipe.return_value = [{"generated_text": expected_prefix + "Hi there!"}]
        mock_pipeline_fn.return_value = mock_pipe

        config = _make_generation_config()
        result = runner.complete("Be helpful", "Hello", config)

        # Verify the pipeline was called with the ChatML-formatted prompt
        prompt_arg = mock_pipe.call_args[0][0]
        assert "<|im_start|>system" in prompt_arg
        assert "<|im_start|>user" in prompt_arg
        assert "<|im_start|>assistant" in prompt_arg

        assert result[1]["content"] == "Hi there!"


# ---------------------------------------------------------------------------
# Runner registry
# ---------------------------------------------------------------------------

class TestQwen3RunnerRegistry:

    def setup_method(self):
        clear_registry()

    def test_qwen3_runner_registered(self):
        """After register_all_runners, 'qwen3' should be available."""
        from agents.architectures.registry import register_all_runners
        register_all_runners()

        runner_cls = get_runner("qwen3")
        assert runner_cls is not None

        from agents.architectures.qwen3_runner import Qwen3Runner
        assert runner_cls is Qwen3Runner

    def test_qwen3_runner_is_base_runner(self):
        from agents.architectures.qwen3_runner import Qwen3Runner
        assert issubclass(Qwen3Runner, BaseRunner)


# ---------------------------------------------------------------------------
# Factory: auto-detection and weights_dir
# ---------------------------------------------------------------------------

class TestFactoryQwen3AutoDetection:

    def test_infer_runner_type_qwen3_models(self):
        assert AgentFactory._infer_runner_type("Qwen/Qwen3-8B") == "qwen3"
        assert AgentFactory._infer_runner_type("Qwen/Qwen3-4B") == "qwen3"
        assert AgentFactory._infer_runner_type("Qwen/Qwen3-1.7B") == "qwen3"
        assert AgentFactory._infer_runner_type("Qwen/Qwen3-0.6B") == "qwen3"
        assert AgentFactory._infer_runner_type("qwen3-custom") == "qwen3"

    def test_infer_runner_type_non_qwen(self):
        assert AgentFactory._infer_runner_type("meta-llama/Meta-Llama-3.1-8B-Instruct") == "mlx_llama"
        assert AgentFactory._infer_runner_type("gpt-4") == "mlx_llama"
        assert AgentFactory._infer_runner_type("Qwen/Qwen2.5-72B-Instruct") == "mlx_llama"

    def test_infer_runner_type_case_insensitive(self):
        assert AgentFactory._infer_runner_type("QWEN/QWEN3-8B") == "qwen3"
        assert AgentFactory._infer_runner_type("qwen/qwen3-8b") == "qwen3"

    def test_factory_auto_detects_qwen3(self):
        """create_agent with a Qwen3 model ID should use the qwen3 runner."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="Qwen/Qwen3-8B",
            )

            MockLocalAgent.assert_called_once()
            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "qwen3"
            assert kwargs["model_id"] == "Qwen/Qwen3-8B"

    def test_factory_explicit_runner_overrides_auto(self):
        """Explicit runner_type should take precedence over auto-detection."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="Qwen/Qwen3-8B",
                runner_type="mlx_llama",
            )

            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "mlx_llama"

    def test_factory_propagates_weights_dir(self):
        """weights_dir kwarg should be merged into device_config."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="Qwen/Qwen3-8B",
                weights_dir="/data/qwen3-weights",
            )

            _, kwargs = MockLocalAgent.call_args
            assert "device_config" in kwargs
            assert kwargs["device_config"]["weights_dir"] == "/data/qwen3-weights"
            # weights_dir should NOT leak as a separate kwarg
            assert "weights_dir" not in kwargs

    def test_factory_weights_dir_merges_with_existing_device_config(self):
        """weights_dir should merge into an existing device_config, not replace it."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="Qwen/Qwen3-8B",
                weights_dir="/data/weights",
                device_config={"device": "cuda"},
            )

            _, kwargs = MockLocalAgent.call_args
            dc = kwargs["device_config"]
            assert dc["weights_dir"] == "/data/weights"
            assert dc["device"] == "cuda"

    def test_factory_weights_dir_works_for_non_qwen(self):
        """weights_dir should be propagated for any runner, not just qwen3."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                weights_dir="/data/llama-weights",
            )

            _, kwargs = MockLocalAgent.call_args
            assert kwargs["device_config"]["weights_dir"] == "/data/llama-weights"


# ---------------------------------------------------------------------------
# Factory: create_from_config
# ---------------------------------------------------------------------------

class TestFactoryCreateFromConfig:

    def test_config_with_qwen3_model(self):
        context = Mock()
        context.settings = Mock()

        config = {
            "agent_type": "local",
            "model": "Qwen/Qwen3-8B",
            "weights_dir": "/data/qwen3",
        }

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_from_config(config, context)

            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "qwen3"
            assert kwargs["device_config"]["weights_dir"] == "/data/qwen3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRunnerConsolidation:
    """The qwen3 and vllm keys must resolve to the same shared implementation."""

    def test_qwen3_is_a_vllm_runner_subclass(self):
        from agents.architectures.qwen3_runner import Qwen3Runner
        from agents.architectures.vllm_runner import VLLMRunner

        assert issubclass(Qwen3Runner, VLLMRunner)
        # The old module-globals sync hack is gone; patching vllm_runner is
        # the single source of truth for both runner keys.
        assert not hasattr(Qwen3Runner, "_sync_shared_symbols")

    def test_both_keys_registered(self):
        from agents.architectures.qwen3_runner import Qwen3Runner
        from agents.architectures.registry import ensure_runners_registered, get_runner
        from agents.architectures.vllm_runner import VLLMRunner

        ensure_runners_registered()
        assert get_runner("vllm") is VLLMRunner
        assert get_runner("qwen3") is Qwen3Runner
