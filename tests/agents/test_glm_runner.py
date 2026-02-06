"""
Tests for the GLM runner and GLM offline agent support.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.architectures.registry import (
    register_runner,
    get_runner,
    clear_registry,
    ensure_runners_registered,
    get_model_by_id,
    get_models_for_provider,
    OnlineModelProviders,
    OnlineModelNames,
)
from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.architectures.glm_runner import GLMRunner
from agents.factory import AgentFactory


class TestGLMRunnerRegistry:
    """Test that the GLM runner is properly registered."""

    def setup_method(self):
        clear_registry()

    def test_glm_runner_registered(self):
        """GLM runner should be registered after ensure_runners_registered."""
        ensure_runners_registered()
        runner_class = get_runner("glm")
        assert runner_class is GLMRunner

    def test_glm_runner_is_base_runner(self):
        """GLMRunner should be a subclass of BaseRunner."""
        assert issubclass(GLMRunner, BaseRunner)


class TestGLMModelRegistry:
    """Test that GLM models appear in the model registry."""

    def test_glm_enum_values_exist(self):
        """GLM enum values should be defined."""
        assert OnlineModelNames.GLM_4_9B_CHAT.value == "THUDM/glm-4-9b-chat"
        assert OnlineModelNames.GLM_4_9B_CHAT_LOCAL.value == "glm-4-9b-chat"

    def test_glm_offline_model_exists(self):
        """GLM 4 9B Chat should appear in OFFLINE static models."""
        models = get_models_for_provider(OnlineModelProviders.OFFLINE)
        glm_models = [m for m in models if "glm" in m.id.lower()]
        assert len(glm_models) >= 1
        glm = glm_models[0]
        assert glm.id == "THUDM/glm-4-9b-chat"
        assert glm.family == "glm-4"
        assert glm.recommended is True

    def test_glm_model_lookup_by_id(self):
        """get_model_by_id should find GLM offline model."""
        model = get_model_by_id("THUDM/glm-4-9b-chat")
        assert model is not None
        assert model.family in ("glm-4", "THUDM")

    def test_glm_huggingface_model_exists(self):
        """GLM should exist in the HUGGINGFACE discovered models."""
        models = get_models_for_provider(OnlineModelProviders.HUGGINGFACE)
        glm_models = [m for m in models if "glm" in m.id.lower()]
        assert len(glm_models) >= 1


class TestGLMRunnerUnit:
    """Unit tests for GLMRunner with mocked GLMTransformer."""

    @patch("agents.architectures.glm_runner.GLMTransformer")
    def test_load(self, MockGLMTransformer):
        """GLMRunner.load should instantiate GLMTransformer with the model id."""
        runner = GLMRunner()
        runner.load("THUDM/glm-4-9b-chat")

        MockGLMTransformer.assert_called_once_with(
            model_id="THUDM/glm-4-9b-chat",
            max_new_tokens=16,
        )
        assert runner.model_id == "THUDM/glm-4-9b-chat"
        assert runner.glm is not None

    @patch("agents.architectures.glm_runner.GLMTransformer")
    def test_complete(self, MockGLMTransformer):
        """GLMRunner.complete should delegate to GLMTransformer.complete."""
        mock_instance = MockGLMTransformer.return_value
        mock_instance.complete.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        runner = GLMRunner()
        runner.load("THUDM/glm-4-9b-chat")

        config = GenerationConfig(max_tokens=128, temperature=0.7, top_p=0.9)
        result = runner.complete("You are helpful.", "Hello", config)

        mock_instance.complete.assert_called_once_with(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        )
        assert len(result) == 2
        assert result[1]["role"] == "assistant"

    @patch("agents.architectures.glm_runner.GLMTransformer")
    def test_generate_delegates_to_complete(self, MockGLMTransformer):
        """GLMRunner.generate should delegate to complete with empty system prompt."""
        mock_instance = MockGLMTransformer.return_value
        mock_instance.complete.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        runner = GLMRunner()
        runner.load("THUDM/glm-4-9b-chat")

        config = GenerationConfig(max_tokens=64)
        runner.generate("Hello", config)

        mock_instance.complete.assert_called_once()
        call_args = mock_instance.complete.call_args
        # system_prompt should default to "You are a helpful assistant."
        assert call_args.kwargs["system_prompt"] == "You are a helpful assistant."

    @patch("agents.architectures.glm_runner.GLMTransformer")
    def test_cleanup(self, MockGLMTransformer):
        """GLMRunner.cleanup should release the model."""
        runner = GLMRunner()
        runner.load("THUDM/glm-4-9b-chat")
        assert runner.glm is not None

        runner.cleanup()
        assert runner.glm is None

    def test_complete_without_load_raises(self):
        """Calling complete before load should raise RuntimeError."""
        runner = GLMRunner()
        config = GenerationConfig()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.complete("sys", "user", config)

    def test_generate_without_load_raises(self):
        """Calling generate before load should raise RuntimeError."""
        runner = GLMRunner()
        config = GenerationConfig()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.generate("prompt", config)


class TestAgentFactoryGLM:
    """Test that AgentFactory selects GLM runner for GLM models."""

    def setup_method(self):
        clear_registry()
        register_runner("glm", GLMRunner)
        register_runner("mlx_llama", Mock)

    def test_auto_select_glm_runner(self):
        """Factory should pick 'glm' runner when model contains 'glm'."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="THUDM/glm-4-9b-chat",
            )

            MockLocalAgent.assert_called_once()
            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "glm"
            assert kwargs["model_id"] == "THUDM/glm-4-9b-chat"

    def test_auto_select_glm_runner_thudm(self):
        """Factory should pick 'glm' runner when model contains 'thudm'."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="THUDM/some-future-model",
            )

            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "glm"

    def test_explicit_runner_type_overrides_auto(self):
        """Explicit runner_type should take precedence over auto-detection."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="THUDM/glm-4-9b-chat",
                runner_type="mlx_llama",
            )

            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "mlx_llama"

    def test_non_glm_model_uses_mlx_llama(self):
        """Non-GLM models should default to mlx_llama runner."""
        context = Mock()
        context.settings = Mock()

        with patch("agents.local_agent.LocalAgent") as MockLocalAgent:
            AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            )

            _, kwargs = MockLocalAgent.call_args
            assert kwargs["runner_type"] == "mlx_llama"


class TestAgentFactoryConfigGLM:
    """Test AgentFactoryConfig runner selection for GLM models."""

    def test_glm_runner_from_user_settings(self):
        """AgentFactoryConfig should auto-select GLM runner for GLM local model."""
        from app.models.user_settings import UserSettingsResponse, AgentFactoryConfig
        from datetime import datetime

        settings = UserSettingsResponse(
            user_settings_id=1,
            user_id=1,
            default_agent_type="local",
            default_local_model="THUDM/glm-4-9b-chat",
            default_online_model="gpt-4",
            default_online_provider="openai",
            default_file_archives=[],
            enable_rag_by_default=True,
            default_max_tokens=4096,
            default_temperature=0.7,
            default_top_p=0.9,
            default_frequency_penalty=0.0,
            default_presence_penalty=0.0,
            backup_providers=[],
            ui_preferences={},
            create_date=datetime.now(),
            update_date=datetime.now(),
        )

        config = AgentFactoryConfig.from_user_settings(settings)
        assert config.agent_type == "local"
        assert config.model == "THUDM/glm-4-9b-chat"
        assert config.runner_type == "glm"

    def test_llama_runner_from_user_settings(self):
        """Non-GLM local models should still use mlx_llama."""
        from app.models.user_settings import UserSettingsResponse, AgentFactoryConfig
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
            default_temperature=0.7,
            default_top_p=0.9,
            default_frequency_penalty=0.0,
            default_presence_penalty=0.0,
            backup_providers=[],
            ui_preferences={},
            create_date=datetime.now(),
            update_date=datetime.now(),
        )

        config = AgentFactoryConfig.from_user_settings(settings)
        assert config.runner_type == "mlx_llama"


if __name__ == "__main__":
    pytest.main([__file__])
