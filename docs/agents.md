# Agent Architecture Documentation

## Overview

The Geist agent architecture provides a flexible and extensible system for running both local and online language models. The system supports multiple inference backends and providers with automatic fallback capabilities.

## Architecture Components

### Base Classes

#### BaseAgent
The abstract base class that all agents inherit from. Defines the core interface for text completion, audio processing, and agent lifecycle management.

#### BaseRunner
Abstract base class for inference runners that handle the actual model execution.

### Agent Types

#### LocalAgent
Executes local models using pluggable runners. Supports:
- MLX (Apple Silicon optimization)
- vLLM (placeholder for future implementation)
- Custom runners via registry

#### OnlineAgent
Routes requests to OpenAI-compatible HTTP endpoints. Supports:
- OpenAI
- Anthropic
- Groq
- Grok (X.AI)
- Custom endpoints
- Backup provider fallback

### Runner Registry

The runner registry allows dynamic registration and discovery of inference backends:

```python
from agents.architectures import register_runner, get_runner

# Register a custom runner
register_runner("my_runner", MyRunnerClass)

# Get a registered runner
runner_class = get_runner("mlx_llama")
```

### Factory Pattern

The `AgentFactory` provides a unified interface for creating agent instances:

```python
from agents.factory import AgentFactory

# Create a local agent
agent = AgentFactory.create_agent(
    agent_type="local",
    agent_context=context,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    runner_type="mlx_llama"
)

# Create an online agent
agent = AgentFactory.create_agent(
    agent_type="online",
    agent_context=context,
    model="gpt-4",
    endpoint="https://api.openai.com/v1/chat/completions",
    api_key="your-api-key"
)
```

## User Settings

### Configuration

User settings control default agent behavior and can be configured via API:

```json
{
  "default_agent_type": "local",
  "default_local_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "default_online_model": "gpt-4",
  "default_online_provider": "openai",
  "backup_providers": [
    {
      "name": "groq",
      "base_url": "https://api.groq.com/openai/v1/chat/completions",
      "model": "llama-3.1-8b-instant",
      "priority": 1
    }
  ],
  "default_max_tokens": 16,
  "default_temperature": 1.0
}
```

### API Endpoints

- `GET /api/v1/user-settings/` - Get current user settings
- `PUT /api/v1/user-settings/` - Update user settings
- `POST /api/v1/user-settings/reset` - Reset to defaults
- `GET /api/v1/user-settings/agent-config/preview` - Preview agent configuration

## Environment Variables

### Local Models
- `HUGGING_FACE_HUB_TOKEN` - Required for downloading Llama models

### Online Providers
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GROQ_API_KEY` - Groq API key
- `GROK_API_KEY` - Grok (X.AI) API key
- `API_KEY` - Generic fallback API key

## Usage Examples

### Basic Text Completion

```python
from app.services.user_settings_service import UserSettingsService
from agents.agent_context import AgentContext

# Get agent context
context = get_default_agent_context()

# Create agent from user settings
agent = UserSettingsService.create_agent_from_default_user(context)

# Complete text
completion = agent.complete_text(
    prompt="Hello, how are you?",
    max_tokens=50,
    temperature=0.7
)
```

### With Configuration Overrides

```python
from app.models.user_settings import AgentConfigRequest

# Override to use online agent
overrides = AgentConfigRequest(
    agent_type="online",
    model="gpt-4",
    temperature=0.5
)

agent = UserSettingsService.create_agent_from_default_user(
    context, 
    overrides=overrides
)
```

### Custom Runner Implementation

```python
from agents.architectures.base_runner import BaseRunner, GenerationConfig

class MyCustomRunner(BaseRunner):
    def load(self, model_id: str, device_config=None):
        # Load your model
        pass
    
    def generate(self, prompt: str, generation_config: GenerationConfig):
        # Generate text
        pass
    
    def complete(self, system_prompt: str, user_prompt: str, generation_config: GenerationConfig):
        # Complete conversation
        pass

# Register the runner
from agents.architectures import register_runner
register_runner("my_custom_runner", MyCustomRunner)
```

## Migration Guide

### From Legacy Agents

The legacy `LlamaAgent` and `GPT4Agent` classes are deprecated. To migrate:

1. **Replace direct instantiation:**
   ```python
   # Old
   agent = LlamaAgent(agent_context, ckpt_dir=None)
   
   # New
   agent = AgentFactory.create_agent(
       agent_type="local",
       agent_context=agent_context,
       model="meta-llama/Meta-Llama-3.1-8B-Instruct"
   )
   ```

2. **Update completion calls:**
   ```python
   # Old
   result = agent.complete_text(prompt="Hello")
   
   # New (same interface)
   result = agent.complete_text(prompt="Hello")
   ```

3. **Use new endpoints:**
   ```python
   # Use /agent/complete_text_new instead of /agent/complete_text
   ```

### Configuration Migration

Update your configuration files to use the new user settings format:

```python
# Old agent instantiation
agent = LlamaAgent(agent_context, ckpt_dir=None)

# New user settings based approach
settings = UserSettingsService.get_or_create_user_settings_by_id(user_id)
agent = UserSettingsService.create_agent_from_user_settings(
    user_id, 
    agent_context
)
```

## Monitoring and Debugging

### Logging

Enable enhanced logging to monitor agent performance:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agents")
```

### Health Checks

Monitor agent health via the factory:

```python
from agents.architectures import list_runners

# Check available runners
runners = list_runners()
print(f"Available runners: {list(runners.keys())}")
```

### Performance Metrics

The system automatically tracks:
- Request latency
- Token usage
- Provider success/failure rates
- Fallback trigger counts

## Best Practices

1. **Use the Factory**: Always create agents via `AgentFactory` for consistency
2. **Configure User Settings**: Set appropriate defaults for your use case
3. **Handle Fallbacks**: Configure backup providers for production deployments
4. **Monitor Performance**: Track token usage and latency metrics
5. **Resource Management**: Call `phase_out()` to clean up agent resources
6. **Error Handling**: Implement proper error handling for network failures

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure `HUGGING_FACE_HUB_TOKEN` is set
   - Check disk space for model storage
   - Verify model ID is correct

2. **API Authentication Failures**
   - Verify API keys are correctly set
   - Check API key permissions
   - Ensure endpoint URLs are correct

3. **Runner Not Found**
   - Verify runner is registered: `register_runner("name", Class)`
   - Check import order
   - Ensure runner class implements `BaseRunner`

4. **Performance Issues**
   - Use appropriate device configs for local models
   - Configure timeouts for online providers
   - Monitor token usage and limits

### Debug Mode

Enable debug mode for detailed logging:

```python
import os
os.environ["ENHANCED_LOGGING"] = "true"
```

This will provide detailed information about:
- Agent creation and configuration
- Request/response payloads
- Runner loading and execution
- Error stack traces
