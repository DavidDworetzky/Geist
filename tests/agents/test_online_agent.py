"""
Unit tests for OnlineAgent with multiple API providers.

Tests cover:
- OpenAI API
- Anthropic (Claude/Sonnet) API
- Groq API
- Grok (X.AI) API
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx
from agents.online_agent import OnlineAgent
from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from app.environment import LoadEnvironmentDictionary


# Sample API responses for different providers
OPENAI_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1728753308,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from OpenAI.",
                "refusal": None
            },
            "logprobs": None,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 8,
        "total_tokens": 22,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "system_fingerprint": None
}

ANTHROPIC_RESPONSE = {
    "id": "msg_test456",
    "object": "chat.completion",
    "created": 1728753308,
    "model": "claude-3-5-sonnet-20241022",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from Anthropic Claude.",
                "refusal": None
            },
            "logprobs": None,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 9,
        "total_tokens": 23,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    }
}

GROQ_RESPONSE = {
    "id": "chatcmpl-groq789",
    "object": "chat.completion",
    "created": 1728753308,
    "model": "llama-3.1-70b-versatile",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from Groq.",
                "refusal": None
            },
            "logprobs": None,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 7,
        "total_tokens": 21,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    }
}

GROK_RESPONSE = {
    "id": "chatcmpl-grok101",
    "object": "chat.completion",
    "created": 1728753308,
    "model": "grok-2",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from Grok.",
                "refusal": None
            },
            "logprobs": None,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 7,
        "total_tokens": 21,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    }
}


def create_mock_agent_context(include_world_processing: bool = False) -> AgentContext:
    """Create a mock agent context for testing."""
    settings = AgentSettings(
        name="test_agent",
        version="1.0",
        description="Test agent",
        max_tokens=100,
        n=1,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        include_world_processing=include_world_processing
    )
    env = LoadEnvironmentDictionary()
    context = AgentContext(settings=settings, envs=env)
    return context


class TestOnlineAgentInitialization:
    """Test OnlineAgent initialization and configuration."""

    def test_openai_initialization(self):
        """Test initialization with OpenAI configuration."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-openai-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            
            assert agent.base_url == "https://api.openai.com/v1"
            assert agent.model == "gpt-4o"
            assert agent.api_key == "test-openai-key"
            assert "Authorization" in agent.headers
            assert agent.headers["Authorization"] == "Bearer test-openai-key"

    def test_anthropic_initialization(self):
        """Test initialization with Anthropic configuration."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-anthropic-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.anthropic.com/v1",
                model="claude-3-5-sonnet-20241022"
            )
            
            assert agent.base_url == "https://api.anthropic.com/v1"
            assert agent.model == "claude-3-5-sonnet-20241022"
            assert agent.api_key == "test-anthropic-key"

    def test_groq_initialization(self):
        """Test initialization with Groq configuration."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-groq-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.1-70b-versatile"
            )
            
            assert agent.base_url == "https://api.groq.com/openai/v1"
            assert agent.model == "llama-3.1-70b-versatile"
            assert agent.api_key == "test-groq-key"

    def test_grok_initialization(self):
        """Test initialization with Grok (X.AI) configuration."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'GROK_API_KEY': 'test-grok-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.x.ai/v1",
                model="grok-2"
            )
            
            assert agent.base_url == "https://api.x.ai/v1"
            assert agent.model == "grok-2"
            assert agent.api_key == "test-grok-key"

    def test_explicit_api_key(self):
        """Test initialization with explicitly provided API key."""
        context = create_mock_agent_context()
        
        agent = OnlineAgent(
            agent_context=context,
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="explicit-api-key"
        )
        
        assert agent.api_key == "explicit-api-key"

    def test_backup_providers_configuration(self):
        """Test initialization with backup providers."""
        context = create_mock_agent_context()
        
        backup_providers = [
            {
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.1-70b-versatile",
                "api_key": "backup-groq-key"
            }
        ]
        
        agent = OnlineAgent(
            agent_context=context,
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="primary-key",
            backup_providers=backup_providers
        )
        
        assert len(agent.backup_providers) == 1
        assert agent.backup_providers[0]["model"] == "llama-3.1-70b-versatile"


class TestOnlineAgentAPIRequests:
    """Test API request handling for different providers."""

    def test_openai_complete_text(self):
        """Test complete_text with OpenAI API."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-openai-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            
            # Mock the HTTP client
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = OPENAI_RESPONSE
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt="Test prompt",
                    max_tokens=100,
                    temperature=0.7
                )
                
                # Verify the request was made
                assert mock_post.called
                call_args = mock_post.call_args
                assert "https://api.openai.com/v1/chat/completions" in call_args[0][0]
                
                # Verify the payload
                payload = call_args[1]['json']
                assert payload['model'] == "gpt-4o"
                assert payload['messages'][-1]['content'] == "Test prompt"
                assert payload['max_tokens'] == 100
                assert payload['temperature'] == 0.7
                
                # Verify the response
                assert result.choices[0].message.content == "This is a test response from OpenAI."

    def test_anthropic_complete_text(self):
        """Test complete_text with Anthropic API."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-anthropic-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.anthropic.com/v1",
                model="claude-3-5-sonnet-20241022"
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = ANTHROPIC_RESPONSE
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt="Test prompt for Claude",
                    max_tokens=150,
                    temperature=0.5
                )
                
                # Verify the request
                assert mock_post.called
                call_args = mock_post.call_args
                
                # Verify the payload
                payload = call_args[1]['json']
                assert payload['model'] == "claude-3-5-sonnet-20241022"
                assert payload['messages'][-1]['content'] == "Test prompt for Claude"
                assert payload['max_tokens'] == 150
                assert payload['temperature'] == 0.5
                
                # Verify the response
                assert result.choices[0].message.content == "This is a test response from Anthropic Claude."

    def test_groq_complete_text(self):
        """Test complete_text with Groq API."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-groq-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.1-70b-versatile"
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = GROQ_RESPONSE
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt="Test prompt for Groq",
                    max_tokens=200,
                    temperature=0.8
                )
                
                # Verify the request
                assert mock_post.called
                call_args = mock_post.call_args
                assert "https://api.groq.com/openai/v1/chat/completions" in call_args[0][0]
                
                # Verify the payload
                payload = call_args[1]['json']
                assert payload['model'] == "llama-3.1-70b-versatile"
                assert payload['messages'][-1]['content'] == "Test prompt for Groq"
                assert payload['max_tokens'] == 200
                
                # Verify the response
                assert result.choices[0].message.content == "This is a test response from Groq."

    def test_grok_complete_text(self):
        """Test complete_text with Grok (X.AI) API."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'GROK_API_KEY': 'test-grok-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.x.ai/v1",
                model="grok-2"
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = GROK_RESPONSE
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt="Test prompt for Grok",
                    max_tokens=120,
                    temperature=0.9
                )
                
                # Verify the request
                assert mock_post.called
                call_args = mock_post.call_args
                assert "https://api.x.ai/v1/chat/completions" in call_args[0][0]
                
                # Verify the payload
                payload = call_args[1]['json']
                assert payload['model'] == "grok-2"
                assert payload['messages'][-1]['content'] == "Test prompt for Grok"
                assert payload['max_tokens'] == 120
                
                # Verify the response
                assert result.choices[0].message.content == "This is a test response from Grok."

    def test_system_prompt_included(self):
        """Test that system prompt is correctly included in the request."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = OPENAI_RESPONSE
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt="User message",
                    system_prompt="You are a helpful assistant.",
                    max_tokens=50
                )
                
                # Verify system prompt is in the payload
                payload = mock_post.call_args[1]['json']
                assert len(payload['messages']) == 2
                assert payload['messages'][0]['role'] == "system"
                assert payload['messages'][0]['content'] == "You are a helpful assistant."
                assert payload['messages'][1]['role'] == "user"
                assert payload['messages'][1]['content'] == "User message"


class TestOnlineAgentRetryLogic:
    """Test retry and error handling logic."""

    def test_retry_on_failure(self):
        """Test that the agent retries on request failures."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                max_retries=3
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                # First two attempts fail, third succeeds
                mock_response_fail = Mock()
                mock_response_fail.status_code = 500
                mock_response_fail.text = "Internal Server Error"
                
                mock_response_success = Mock()
                mock_response_success.status_code = 200
                mock_response_success.json.return_value = OPENAI_RESPONSE
                
                mock_post.side_effect = [
                    mock_response_fail,
                    mock_response_fail,
                    mock_response_success
                ]
                
                result = agent.complete_text(prompt="Test prompt", max_tokens=50)
                
                # Verify it was called 3 times
                assert mock_post.call_count == 3
                
                # Verify we got a successful result
                assert result.choices[0].message.content == "This is a test response from OpenAI."

    def test_max_retries_exceeded(self):
        """Test that an exception is raised when max retries is exceeded."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                max_retries=2
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 500
                mock_response.text = "Internal Server Error"
                mock_post.return_value = mock_response
                
                with pytest.raises(Exception, match="API request failed after"):
                    agent.complete_text(prompt="Test prompt", max_tokens=50)
                
                # Verify it was called max_retries times
                assert mock_post.call_count == 2

    def test_backup_provider_on_server_error(self):
        """Test that backup provider is used on server error."""
        context = create_mock_agent_context()
        
        backup_providers = [
            {
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.1-70b-versatile",
                "api_key": "backup-key"
            }
        ]
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                backup_providers=backup_providers,
                max_retries=1
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                # Primary provider fails with 500
                mock_response_fail = Mock()
                mock_response_fail.status_code = 500
                mock_response_fail.text = "Internal Server Error"
                
                # Backup provider succeeds
                mock_response_success = Mock()
                mock_response_success.status_code = 200
                mock_response_success.json.return_value = GROQ_RESPONSE
                
                mock_post.side_effect = [
                    mock_response_fail,
                    mock_response_success
                ]
                
                result = agent.complete_text(prompt="Test prompt", max_tokens=50)
                
                # Verify backup was called
                assert mock_post.call_count == 2
                
                # Verify the backup provider was used
                second_call_args = mock_post.call_args_list[1]
                assert "groq.com" in second_call_args[0][0]

    def test_request_error_with_backup(self):
        """Test backup provider is used on request errors."""
        context = create_mock_agent_context()
        
        backup_providers = [
            {
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.1-70b-versatile",
                "api_key": "backup-key"
            }
        ]
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                backup_providers=backup_providers,
                max_retries=2
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                # Primary provider has network error
                mock_response_success = Mock()
                mock_response_success.status_code = 200
                mock_response_success.json.return_value = GROQ_RESPONSE

                mock_post.side_effect = [
                    httpx.RequestError("Connection failed"),
                    httpx.RequestError("Connection failed"),
                    # Backup provider succeeds
                    mock_response_success
                ]

                result = agent.complete_text(prompt="Test prompt", max_tokens=50)
                
                # Verify backup was eventually used
                assert mock_post.call_count == 3


class TestOnlineAgentEdgeCases:
    """Test edge cases and special scenarios."""

    def test_url_normalization(self):
        """Test that URLs are normalized correctly."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            # URL without trailing slash
            agent1 = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            assert agent1.base_url == "https://api.openai.com/v1"
            
            # URL with trailing slash
            agent2 = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1/",
                model="gpt-4o"
            )
            assert agent2.base_url == "https://api.openai.com/v1"

    def test_stop_sequence_parameter(self):
        """Test that stop sequences are included in the request."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = OPENAI_RESPONSE
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt="Test prompt",
                    max_tokens=50,
                    stop="END"
                )
                
                # Verify stop sequence is in payload
                payload = mock_post.call_args[1]['json']
                assert 'stop' in payload
                assert payload['stop'] == "END"

    def test_all_generation_parameters(self):
        """Test that all generation parameters are correctly passed."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = OPENAI_RESPONSE
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt="Test prompt",
                    max_tokens=256,
                    n=2,
                    temperature=0.75,
                    top_p=0.95,
                    frequency_penalty=0.5,
                    presence_penalty=0.3,
                    stop="STOP"
                )
                
                # Verify all parameters are in payload
                payload = mock_post.call_args[1]['json']
                assert payload['max_tokens'] == 256
                assert payload['n'] == 2
                assert payload['temperature'] == 0.75
                assert payload['top_p'] == 0.95
                assert payload['frequency_penalty'] == 0.5
                assert payload['presence_penalty'] == 0.3
                assert payload['stop'] == "STOP"

    def test_phase_out_cleanup(self):
        """Test that phase_out properly cleans up resources."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            
            with patch.object(agent.client, 'close') as mock_close:
                agent.phase_out()
                
                # Verify client was closed
                assert mock_close.called


class TestOnlineAgentMultiProvider:
    """Test scenarios involving multiple providers."""

    @pytest.mark.parametrize("provider,base_url,model,expected_content", [
        ("openai", "https://api.openai.com/v1", "gpt-4o", "This is a test response from OpenAI."),
        ("anthropic", "https://api.anthropic.com/v1", "claude-3-5-sonnet-20241022", "This is a test response from Anthropic Claude."),
        ("groq", "https://api.groq.com/openai/v1", "llama-3.1-70b-versatile", "This is a test response from Groq."),
        ("grok", "https://api.x.ai/v1", "grok-2", "This is a test response from Grok."),
    ])
    def test_multiple_providers(self, provider, base_url, model, expected_content):
        """Test that all providers work correctly with parameterized tests."""
        context = create_mock_agent_context()
        
        # Map provider to response and env var
        response_map = {
            "openai": (OPENAI_RESPONSE, "OPENAI_API_KEY"),
            "anthropic": (ANTHROPIC_RESPONSE, "ANTHROPIC_API_KEY"),
            "groq": (GROQ_RESPONSE, "GROQ_API_KEY"),
            "grok": (GROK_RESPONSE, "GROK_API_KEY")
        }
        
        response_data, env_var = response_map[provider]
        
        with patch.dict('os.environ', {env_var: f'test-{provider}-key'}):
            agent = OnlineAgent(
                agent_context=context,
                base_url=base_url,
                model=model
            )
            
            with patch.object(agent.client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = response_data
                mock_post.return_value = mock_response
                
                result = agent.complete_text(
                    prompt=f"Test prompt for {provider}",
                    max_tokens=100
                )
                
                # Verify the request was made to the correct endpoint
                assert mock_post.called
                call_url = mock_post.call_args[0][0]
                assert base_url in call_url
                
                # Verify the response content
                assert result.choices[0].message.content == expected_content


class TestOnlineAgentAPIKeyRetrieval:
    """Test API key retrieval from environment."""

    def test_openai_api_key_from_env(self):
        """Test OpenAI API key is retrieved from environment."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-openai-key'}, clear=True):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.openai.com/v1",
                model="gpt-4o"
            )
            
            assert agent.api_key == "env-openai-key"

    def test_anthropic_api_key_from_env(self):
        """Test Anthropic API key is retrieved from environment."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-anthropic-key'}, clear=True):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.anthropic.com/v1",
                model="claude-3-5-sonnet-20241022"
            )
            
            assert agent.api_key == "env-anthropic-key"

    def test_groq_api_key_from_env(self):
        """Test Groq API key is retrieved from environment."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'GROQ_API_KEY': 'env-groq-key'}, clear=True):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.1-70b-versatile"
            )
            
            assert agent.api_key == "env-groq-key"

    def test_grok_api_key_from_env(self):
        """Test Grok API key is retrieved from environment."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'GROK_API_KEY': 'env-grok-key'}, clear=True):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.x.ai/v1",
                model="grok-2"
            )
            
            assert agent.api_key == "env-grok-key"

    def test_fallback_to_generic_api_key(self):
        """Test fallback to generic API_KEY environment variable."""
        context = create_mock_agent_context()
        
        with patch.dict('os.environ', {'API_KEY': 'generic-api-key'}, clear=True):
            agent = OnlineAgent(
                agent_context=context,
                base_url="https://api.custom-provider.com/v1",
                model="custom-model"
            )
            
            assert agent.api_key == "generic-api-key"

