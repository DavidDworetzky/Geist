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
import json
import os
import platform
import sys
from queue import Queue
from unittest.mock import MagicMock, Mock, patch

import pytest


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

from agents.architectures import vllm_runner as vllm_runner_module
from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.architectures.registry import clear_registry, get_runner
from agents.factory import AgentFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generation_config(**overrides):
    defaults = {"max_tokens": 64, "temperature": 0.7, "top_p": 0.9}
    defaults.update(overrides)
    return GenerationConfig(**defaults)


def _mock_tokenizer(has_chat_template=True):
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.pad_token_id = 2
    tok.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
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

    def generate(**kwargs):
        kwargs["streamer"].on_finalized_text("Hello ")
        kwargs["streamer"].on_finalized_text("world!", stream_end=True)

    model.generate.side_effect = generate
    return model


class FakeTextIteratorStreamer:
    def __init__(self, *_args, **kwargs):
        self.chunks = Queue()
        self.timeout = kwargs.get("timeout")

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.chunks.put(text)
        if stream_end:
            self.chunks.put(None)

    def end(self):
        self.chunks.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.chunks.get(timeout=1)
        if chunk is None:
            raise StopIteration
        return chunk


@pytest.fixture(autouse=True)
def use_fake_text_streamer(monkeypatch):
    monkeypatch.setattr(vllm_runner_module, "TextIteratorStreamer", FakeTextIteratorStreamer)


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
    def test_hub_no_login_without_token(
        self, mock_exists, mock_tok_cls, mock_model_cls, mock_login
    ):
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

        with patch(
            "agents.architectures.vllm_runner.os.path.exists",
            side_effect=self._exists_side_effect(weights_dir),
        ):
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
    def test_load_from_safetensors(
        self, mock_glob, mock_tok_cls, mock_config_cls, mock_model_cls, mock_load_file
    ):
        from agents.architectures.qwen3_runner import Qwen3Runner

        weights_dir = "/data/qwen3-weights"
        st_files = [
            os.path.join(weights_dir, "model-00001-of-00004.safetensors"),
            os.path.join(weights_dir, "model-00002-of-00004.safetensors"),
        ]
        mock_glob.return_value = st_files

        # config.json exists but NO model.safetensors.index.json
        def _exists(path):
            return path == os.path.join(weights_dir, "config.json") or path == os.path.join(
                weights_dir, "tokenizer.json"
            )

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_config = MagicMock()
        mock_config_cls.from_pretrained.return_value = mock_config
        mock_model = _mock_model()
        mock_model.load_state_dict.return_value = ([], [])  # (missing, unexpected)
        mock_model_cls.from_config.return_value = mock_model
        mock_load_file.return_value = {"layer.weight": MagicMock()}

        config_data = {"torch_dtype": "bfloat16", "model_type": "qwen3"}

        with (
            patch("agents.architectures.vllm_runner.os.path.exists", side_effect=_exists),
            patch("builtins.open", create=True) as mock_open,
        ):
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
    def test_safetensors_tokenizer_fallback_to_hub(
        self, mock_glob, mock_tok_cls, mock_config_cls, mock_model_cls, mock_load_file
    ):
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

        with (
            patch("agents.architectures.vllm_runner.os.path.exists", side_effect=_exists),
            patch("builtins.open", return_value=mock_file),
            patch("agents.architectures.vllm_runner.json.load", return_value={}),
        ):
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
        import torch

        from agents.architectures.qwen3_runner import Qwen3Runner

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

    def test_complete_with_chat_template(self):
        runner = self._create_loaded_runner()
        config = _make_generation_config()

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

    def test_complete_without_system_prompt(self):
        runner = self._create_loaded_runner()
        config = _make_generation_config()

        runner.complete("", "Just a user message", config)

        # No system message in the template call
        messages = runner.tokenizer.apply_chat_template.call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_complete_hides_stop_markers(self):
        runner = self._create_loaded_runner()
        config = _make_generation_config(stop=["<|im_end|>"])

        def generate(**kwargs):
            kwargs["streamer"].on_finalized_text(
                "Clean response<|im_end|>extra junk",
                stream_end=True,
            )

        runner.model.generate.side_effect = generate

        result = runner.complete("", "test", config)
        assert result[1]["content"] == "Clean response"

    def test_generate_collects_the_stream(self):
        runner = self._create_loaded_runner()
        config = _make_generation_config()

        result = runner.generate("prompt text", config)

        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hello world!"
        streamer = runner.model.generate.call_args.kwargs["streamer"]
        assert streamer.timeout == vllm_runner_module.GENERATION_STREAM_TIMEOUT_SECONDS

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

    def test_complete_respects_temperature_zero(self):
        """Temperature 0 disables sampling-only generation arguments."""
        runner = self._create_loaded_runner()
        config = _make_generation_config(temperature=0.0)

        runner.complete("", "test", config)

        kwargs = runner.model.generate.call_args.kwargs
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs


# ---------------------------------------------------------------------------
# Qwen3Runner: model reuse
# ---------------------------------------------------------------------------


class TestQwen3RunnerPipelineCaching:
    def test_reuses_loaded_model_for_multiple_streams(self):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.model = _mock_model()
        runner.tokenizer = _mock_tokenizer()
        runner.device = "cpu"
        runner.model_id = "Qwen/Qwen3-8B"

        config = _make_generation_config()

        runner.complete("", "first", config)
        runner.complete("", "second", config)

        assert runner.model.generate.call_count == 2


# ---------------------------------------------------------------------------
# Qwen3Runner: cleanup
# ---------------------------------------------------------------------------


class TestQwen3RunnerCleanup:
    def test_cleanup_releases_resources(self):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.model = _mock_model()
        runner.tokenizer = _mock_tokenizer()

        runner.cleanup()

        assert runner.model is None
        assert runner.tokenizer is None

    def test_cleanup_on_fresh_runner(self):
        """Cleanup on a never-loaded runner should not raise."""
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.cleanup()  # should be a no-op


# ---------------------------------------------------------------------------
# Qwen3Runner: ChatML fallback
# ---------------------------------------------------------------------------


class TestQwen3RunnerChatMLFallback:
    def test_fallback_chatml_format(self):
        from agents.architectures.qwen3_runner import Qwen3Runner

        runner = Qwen3Runner()
        runner.model = _mock_model()
        runner.tokenizer = _mock_tokenizer(has_chat_template=False)
        runner.device = "cpu"
        runner.model_id = "Qwen/Qwen3-8B"

        expected_prefix = (
            "<|im_start|>system\nBe helpful<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        config = _make_generation_config()
        result = runner.complete("Be helpful", "Hello", config)

        prompt_arg = runner.tokenizer.call_args.args[0]
        assert prompt_arg == expected_prefix

        assert result[1]["content"] == "Hello world!"


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


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"},
    reason="Qwen3.8 MLX dependencies are Apple Silicon-only",
)
def test_qwen3_8_declared_minimum_matches_installed_transformers():
    from importlib import metadata

    from packaging.version import Version
    from transformers import AutoConfig

    from agents.model_catalog import get_model_spec

    required = get_model_spec("Qwen/Qwen3.8-27B").min_transformers_version

    assert required is not None
    assert Version(metadata.version("transformers")) >= Version(required)
    assert AutoConfig.for_model("qwen3_5").model_type == "qwen3_5"


class TestFactoryGenericAutoDetection:
    def test_infer_runner_type_qwen_models(self):
        with patch("agents.factory.sys.platform", "linux"):
            assert AgentFactory._infer_runner_type("Qwen/Qwen3.8-27B") == "llama_server"
            assert AgentFactory._infer_runner_type("Qwen/Qwen3-8B") == "llama_server"
            assert AgentFactory._infer_runner_type("Qwen/Qwen3-4B") == "llama_server"
            assert AgentFactory._infer_runner_type("Qwen/Qwen2.5-3B-Instruct") == "llama_server"
            assert AgentFactory._infer_runner_type("qwen3-custom") == "llama_server"

    def test_infer_runner_type_non_qwen(self):
        native = sys.platform in {"win32", "linux"}
        assert AgentFactory._infer_runner_type("meta-llama/Meta-Llama-3.1-8B-Instruct") == (
            "llama_server" if native else "mlx_llama"
        )
        assert AgentFactory._infer_runner_type("future-org/future-model") == (
            "llama_server" if native else "transformers"
        )
        with pytest.raises(ValueError, match="server-backed"):
            AgentFactory._infer_runner_type("Qwen/Qwen2.5-72B-Instruct")

    def test_infer_runner_type_case_insensitive(self):
        with patch("agents.factory.sys.platform", "linux"):
            assert AgentFactory._infer_runner_type("QWEN/QWEN3-8B") == "llama_server"
            assert AgentFactory._infer_runner_type("qwen/qwen3-8b") == "llama_server"

    def test_factory_auto_detects_qwen3(self):
        """Qwen uses the same generic runner as other standard causal LMs."""
        context = Mock()
        context.settings = Mock()

        with (
            patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": ""}),
            patch("agents.factory.sys.platform", "linux"),
            patch("agents.local_agent.LocalAgent") as MockLocalAgent,
        ):
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="Qwen/Qwen3-8B",
            )

            MockLocalAgent.assert_called_once()
            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "llama_server"
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

        with (
            patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": ""}),
            patch("agents.factory.sys.platform", "linux"),
            patch("agents.local_agent.LocalAgent") as MockLocalAgent,
        ):
            AgentFactory.create_from_config(config, context)

            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "llama_server"
            assert kwargs["device_config"]["weights_dir"] == "/data/qwen3"


class TestSettingsDrivenQwen3Creation:
    @patch("app.services.user_settings_service.AgentFactory.create_agent")
    @patch("app.services.user_settings_service.AgentFactoryConfig.from_user_settings")
    @patch(
        "app.services.user_settings_service.UserSettingsService."
        "get_or_create_user_settings_by_id"
    )
    def test_local_settings_do_not_forward_online_backup_providers(
        self, mock_get_settings, mock_from_settings, mock_create_agent
    ):
        from app.services.user_settings_service import UserSettingsService

        factory_config = Mock(
            agent_type="local",
            model="Qwen/Qwen3.8-27B",
            endpoint=None,
            api_key=None,
            runner_type=None,
            device_config={},
            generation_config={"max_tokens": 8},
            backup_providers=[],
        )
        mock_from_settings.return_value = factory_config

        UserSettingsService.create_agent_from_user_settings(1, Mock())

        assert "backup_providers" not in mock_create_agent.call_args.kwargs


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
