"""
Tests for the new agent architecture (LocalAgent and OnlineAgent).
"""
import pytest
from unittest.mock import Mock, patch
from agents.architectures import register_runner, get_runner, clear_registry
from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.factory import AgentFactory
from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings

class MockRunner(BaseRunner):
    """Mock runner for testing."""
    
    def __init__(self):
        self.loaded = False
        self.model_id = None
    
    def load(self, model_id: str, device_config=None):
        self.loaded = True
        self.model_id = model_id
    
    def generate(self, prompt: str, generation_config: GenerationConfig):
        return {
            "generated_text": f"Generated: {prompt}",
            "model": self.model_id
        }
    
    def complete(self, system_prompt: str, user_prompt: str, generation_config: GenerationConfig):
        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": f"Response to: {user_prompt}"}
            ]
        }

class TestRunnerRegistry:
    """Test the runner registry functionality."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()
    
    def test_register_and_get_runner(self):
        """Test registering and retrieving runners."""
        register_runner("test_runner", MockRunner)
        
        runner_class = get_runner("test_runner")
        assert runner_class == MockRunner
        
        # Test non-existent runner
        assert get_runner("non_existent") is None
    
    def test_runner_instantiation(self):
        """Test that runners can be instantiated and used."""
        register_runner("mock", MockRunner)
        
        runner_class = get_runner("mock")
        runner = runner_class()
        
        # Test loading
        runner.load("test-model")
        assert runner.loaded
        assert runner.model_id == "test-model"
        
        # Test generation
        config = GenerationConfig(max_tokens=50, temperature=0.7)
        result = runner.generate("Hello", config)
        assert "Generated: Hello" in result["generated_text"]

class TestAgentFactory:
    """Test the agent factory functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        clear_registry()
        register_runner("mock", MockRunner)
    
    def test_create_local_agent(self):
        """Test creating a local agent."""
        # Mock agent context
        settings = Mock()
        settings.max_tokens = 16
        settings.temperature = 1.0
        settings.top_p = 1.0
        settings.frequency_penalty = 0.0
        settings.presence_penalty = 0.0
        settings.include_world_processing = False
        
        context = Mock()
        context.settings = settings
        context.world_context = []
        context.task_context = []
        context.execution_context = []
        
        with patch('agents.local_agent.LocalAgent') as MockLocalAgent:
            agent = AgentFactory.create_agent(
                agent_type="local",
                agent_context=context,
                model="test-model",
                runner_type="mock"
            )
            
            MockLocalAgent.assert_called_once()
            args, kwargs = MockLocalAgent.call_args
            assert kwargs['model_id'] == "test-model"
            assert kwargs['runner_type'] == "mock"
    
    def test_create_online_agent(self):
        """Test creating an online agent."""
        settings = Mock()
        context = Mock()
        context.settings = settings
        
        with patch('agents.online_agent.OnlineAgent') as MockOnlineAgent:
            agent = AgentFactory.create_agent(
                agent_type="online",
                agent_context=context,
                model="gpt-4",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key="test-key"
            )
            
            MockOnlineAgent.assert_called_once()
            args, kwargs = MockOnlineAgent.call_args
            assert kwargs['model'] == "gpt-4"
            assert kwargs['base_url'] == "https://api.openai.com/v1/chat/completions"
            assert kwargs['api_key'] == "test-key"
    
    def test_invalid_agent_type(self):
        """Test that invalid agent types raise an error."""
        context = Mock()
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create_agent(
                agent_type="invalid",
                agent_context=context
            )

class TestUserSettingsIntegration:
    """Test integration with user settings."""
    
    def test_agent_factory_config_creation(self):
        """Test creating agent factory config from user settings."""
        from app.models.user_settings import UserSettingsResponse, AgentFactoryConfig
        from datetime import datetime
        
        # Mock user settings
        settings = UserSettingsResponse(
            user_settings_id=1,
            user_id=1,
            default_agent_type="local",
            default_local_model="test-model",
            default_online_model="gpt-4",
            default_online_provider="openai",
            default_file_archives=[],
            enable_rag_by_default=True,
            default_max_tokens=16,
            default_temperature=0.7,
            default_top_p=0.9,
            default_frequency_penalty=0.1,
            default_presence_penalty=0.1,
            backup_providers=[],
            ui_preferences={},
            create_date=datetime.now(),
            update_date=datetime.now()
        )
        
        # Test local agent config
        config = AgentFactoryConfig.from_user_settings(settings)
        assert config.agent_type == "local"
        assert config.model == "test-model"
        assert config.runner_type == "mlx_llama"
        assert config.endpoint is None
        
        # Test online agent config
        settings.default_agent_type = "online"
        config = AgentFactoryConfig.from_user_settings(settings)
        assert config.agent_type == "online"
        assert config.model == "gpt-4"
        assert config.endpoint == "https://api.openai.com/v1/chat/completions"
        assert config.runner_type is None

if __name__ == "__main__":
    pytest.main([__file__])
