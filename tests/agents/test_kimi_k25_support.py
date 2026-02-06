"""
Tests for Kimi K2.5 support in the local agent architecture.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.architectures.registry import (
    register_runner,
    get_runner,
    clear_registry,
    ensure_runners_registered,
    OnlineModelNames,
    OnlineModelProviders,
    STATIC_MODELS,
    get_model_by_id,
    get_models_for_provider,
)
from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.architectures.huggingface_runner import HuggingFaceRunner
from agents.factory import AgentFactory
from app.models.user_settings import AgentFactoryConfig, _infer_runner_type


class TestKimiK25ModelRegistry:
    """Test that Kimi K2.5 is properly registered in the model registry."""

    def test_kimi_k25_in_online_model_names(self):
        """Kimi K2.5 should be in the OnlineModelNames enum."""
        assert OnlineModelNames.KIMI_K25.value == "moonshotai/Kimi-K2.5"
        assert OnlineModelNames.KIMI_K25_LOCAL.value == "Kimi-K2.5"

    def test_kimi_k25_in_static_huggingface_models(self):
        """Kimi K2.5 should appear in the STATIC_MODELS for HuggingFace."""
        hf_models = STATIC_MODELS[OnlineModelProviders.HUGGINGFACE]
        kimi_models = [m for m in hf_models if m.id == "moonshotai/Kimi-K2.5"]
        assert len(kimi_models) == 1

        kimi = kimi_models[0]
        assert kimi.name == "Kimi K2.5"
        assert kimi.provider == OnlineModelProviders.HUGGINGFACE
        assert kimi.supports_vision is True
        assert kimi.supports_function_calling is True
        assert kimi.family == "kimi"
        assert kimi.context_window == 131072

    def test_kimi_k25_in_static_offline_models(self):
        """Kimi K2.5 should appear in the STATIC_MODELS for OFFLINE."""
        offline_models = STATIC_MODELS[OnlineModelProviders.OFFLINE]
        kimi_models = [m for m in offline_models if m.id == "Kimi-K2.5"]
        assert len(kimi_models) == 1

        kimi = kimi_models[0]
        assert kimi.name == "Kimi K2.5 (Local)"
        assert kimi.provider == OnlineModelProviders.OFFLINE
        assert kimi.supports_vision is True
        assert kimi.supports_function_calling is True
        assert kimi.family == "kimi"

    def test_kimi_k25_findable_in_static_models(self):
        """Kimi K2.5 should be findable by scanning STATIC_MODELS directly."""
        # get_model_by_id prefers DISCOVERED_MODELS, so we verify the static
        # registry entries directly to ensure they're present.
        found_hf = False
        found_offline = False
        for model in STATIC_MODELS.get(OnlineModelProviders.HUGGINGFACE, []):
            if model.id == "moonshotai/Kimi-K2.5":
                found_hf = True
        for model in STATIC_MODELS.get(OnlineModelProviders.OFFLINE, []):
            if model.id == "Kimi-K2.5":
                found_offline = True
        assert found_hf, "Kimi K2.5 not found in STATIC_MODELS[HUGGINGFACE]"
        assert found_offline, "Kimi-K2.5 not found in STATIC_MODELS[OFFLINE]"


class TestHuggingFaceRunnerRegistration:
    """Test that the HuggingFace runner is registered."""

    def setup_method(self):
        clear_registry()

    def test_huggingface_runner_registered(self):
        """The HuggingFace runner should be registered after ensure_runners_registered."""
        ensure_runners_registered()
        runner_class = get_runner("huggingface")
        assert runner_class is HuggingFaceRunner

    def test_mlx_llama_runner_registered_if_available(self):
        """mlx_llama runner should be registered if mlx is available on this platform."""
        ensure_runners_registered()
        runner_class = get_runner("mlx_llama")
        # mlx is only available on Apple Silicon; on other platforms this is None
        try:
            import mlx  # noqa: F401
            assert runner_class is not None
        except ImportError:
            assert runner_class is None


class TestHuggingFaceRunner:
    """Test the HuggingFaceRunner implementation."""

    def test_runner_is_base_runner(self):
        """HuggingFaceRunner should be a BaseRunner subclass."""
        assert issubclass(HuggingFaceRunner, BaseRunner)

    def test_runner_init(self):
        """Runner should initialize with None state."""
        runner = HuggingFaceRunner()
        assert runner.model is None
        assert runner.tokenizer is None
        assert runner.model_id is None

    def test_generate_raises_without_load(self):
        """generate should raise RuntimeError if model not loaded."""
        runner = HuggingFaceRunner()
        config = GenerationConfig(max_tokens=50)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.generate("hello", config)

    def test_complete_raises_without_load(self):
        """complete should raise RuntimeError if model not loaded."""
        runner = HuggingFaceRunner()
        config = GenerationConfig(max_tokens=50)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.complete("system", "user", config)

    def test_cleanup(self):
        """cleanup should clear model references."""
        runner = HuggingFaceRunner()
        runner.model = Mock()
        runner.tokenizer = Mock()
        runner.pipeline = Mock()

        runner.cleanup()

        assert runner.model is None
        assert runner.tokenizer is None
        assert runner.pipeline is None

    def test_load_from_hub(self):
        """load should attempt to load from HuggingFace Hub when local weights not found."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runner = HuggingFaceRunner()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "agents.architectures.huggingface_runner.os.path.exists",
            return_value=False,
        ), patch.object(
            AutoModelForCausalLM,
            "from_pretrained",
            return_value=mock_model,
        ) as mock_model_load, patch.object(
            AutoTokenizer,
            "from_pretrained",
            return_value=mock_tokenizer,
        ) as mock_tok_load:
            runner.load("moonshotai/Kimi-K2.5")

            # Should load from the model ID (hub) since local weights don't exist
            mock_tok_load.assert_called_once_with(
                "moonshotai/Kimi-K2.5", trust_remote_code=True
            )
            mock_model_load.assert_called_once()
            assert runner.model_id == "moonshotai/Kimi-K2.5"


class TestRunnerTypeInference:
    """Test the runner type inference logic."""

    def test_llama_models_use_mlx(self):
        """Llama-family models should infer mlx_llama runner."""
        assert _infer_runner_type("meta-llama/Meta-Llama-3.1-8B-Instruct") == "mlx_llama"
        assert _infer_runner_type("Meta-Llama-3.1-8B-Instruct") == "mlx_llama"
        assert _infer_runner_type("Llama-3.3-70B-Instruct") == "mlx_llama"

    def test_kimi_models_use_huggingface(self):
        """Kimi models should infer huggingface runner."""
        assert _infer_runner_type("moonshotai/Kimi-K2.5") == "huggingface"
        assert _infer_runner_type("Kimi-K2.5") == "huggingface"

    def test_other_models_use_huggingface(self):
        """Non-Llama models should infer huggingface runner."""
        assert _infer_runner_type("Qwen/Qwen2.5-72B-Instruct") == "huggingface"
        assert _infer_runner_type("mistralai/Mixtral-8x7B-Instruct-v0.1") == "huggingface"
        assert _infer_runner_type("deepseek-ai/DeepSeek-V2.5") == "huggingface"


class TestAgentFactoryWithKimi:
    """Test agent factory with Kimi K2.5 models."""

    def setup_method(self):
        clear_registry()
        register_runner("huggingface", HuggingFaceRunner)

    def test_create_local_agent_with_kimi(self):
        """Factory should create a local agent with kimi model and huggingface runner."""
        context = Mock()
        context.settings = Mock()
        context.settings.max_tokens = 16
        context.settings.temperature = 1.0
        context.settings.top_p = 1.0
        context.settings.frequency_penalty = 0.0
        context.settings.presence_penalty = 0.0
        context.settings.include_world_processing = False

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            agent = AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="moonshotai/Kimi-K2.5",
                runner_type="huggingface",
            )

            MockLocalAgent.assert_called_once()
            args, kwargs = MockLocalAgent.call_args
            assert kwargs["model_id"] == "moonshotai/Kimi-K2.5"
            assert kwargs["runner_type"] == "huggingface"


class TestAgentFactoryConfigWithKimi:
    """Test AgentFactoryConfig runner inference with Kimi models."""

    def test_from_user_settings_kimi_model(self):
        """AgentFactoryConfig should infer huggingface runner for Kimi models."""
        from app.models.user_settings import UserSettingsResponse
        from datetime import datetime

        settings = UserSettingsResponse(
            user_settings_id=1,
            user_id=1,
            default_agent_type="local",
            default_local_model="moonshotai/Kimi-K2.5",
            default_online_model="gpt-4",
            default_online_provider="openai",
            default_file_archives=[],
            enable_rag_by_default=True,
            default_max_tokens=4096,
            default_temperature=1.0,
            default_top_p=1.0,
            default_frequency_penalty=0.0,
            default_presence_penalty=0.0,
            backup_providers=[],
            ui_preferences={},
            create_date=datetime.now(),
            update_date=datetime.now(),
        )

        config = AgentFactoryConfig.from_user_settings(settings)
        assert config.agent_type == "local"
        assert config.model == "moonshotai/Kimi-K2.5"
        assert config.runner_type == "huggingface"

    def test_from_user_settings_llama_model_unchanged(self):
        """AgentFactoryConfig should still infer mlx_llama for Llama models."""
        from app.models.user_settings import UserSettingsResponse
        from datetime import datetime

        settings = UserSettingsResponse(
            user_settings_id=1,
            user_id=1,
            default_agent_type="local",
            default_local_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            default_online_model="gpt-4",
            default_online_provider="openai",
            default_file_archives=[],
            enable_rag_by_default=True,
            default_max_tokens=4096,
            default_temperature=1.0,
            default_top_p=1.0,
            default_frequency_penalty=0.0,
            default_presence_penalty=0.0,
            backup_providers=[],
            ui_preferences={},
            create_date=datetime.now(),
            update_date=datetime.now(),
        )

        config = AgentFactoryConfig.from_user_settings(settings)
        assert config.agent_type == "local"
        assert config.model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert config.runner_type == "mlx_llama"


if __name__ == "__main__":
    pytest.main([__file__])
