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

## State Snapshots

Agents persist append-only snapshots of their execution state (world, task,
and execution contexts plus the function log) to the `agent_snapshot` table:

- After every `tick()` (reason `tick`)
- On `phase_out()` (reason `phase_out`), and `phase_in()` restores the latest snapshot
- On demand via `agent.save_state_snapshot(reason="manual")`

Each snapshot gets a monotonic `step` number per agent, enabling crash
recovery, tick-by-tick inspection, and rollback to an earlier step:

```python
# Roll back to a specific step
from app.models.database.agent_snapshot import get_snapshots

history = get_snapshots(agent_identifier=context.agent_id)
agent.restore_state_snapshot(snapshot_id=history[2].snapshot_id)

# Resume the latest state after a restart
agent.restore_state_snapshot()
```

Snapshot writes never raise into the agent loop — failures are logged and the
tick continues.

### Retention

Snapshot history is bounded by time-based retention applied on every write:
snapshots older than `AgentSettings.snapshot_retention_days` (default 7) are
deleted after each new snapshot is persisted. The most recent snapshot is
always kept regardless of age — it is the agent's resume point for
`phase_in()`. Set `snapshot_retention_days = 0` to disable pruning (e.g. for
debugging). Count-based capping is also available via
`prune_snapshots(agent_identifier, keep_last=100)`. Compaction of snapshot
contents (e.g. summarizing old function logs) is future work.

## Structured Tool Calling

Adapter actions are reflected into JSON schemas (`adapters/tool_schema.py`)
from method signatures, type hints, and docstrings, giving models full
function visibility: adapter/action names, typed parameters, and required
arguments. The shared pipeline (`agents/tool_calling.py`) then parses,
validates, and dispatches model-emitted tool calls.

### OnlineAgent (native function calling)

`OnlineAgent.complete_with_tools()` exposes the schemas through the
OpenAI-compatible `tools` parameter, dispatches returned `tool_calls`, and
feeds results back as `tool` messages until the model produces a final
answer. If the endpoint rejects the `tools` payload, it automatically falls
back to the prompt-based path.

```python
completion = agent.complete_with_tools(
    "Write today's notes to notes.md",
    max_tool_iterations=4,
)
print(completion.content)        # final model answer
print(completion.tool_results)   # every dispatched ToolResult
```

### LocalAgent (schema-grounded prompting)

Local models keep reflection-based function visibility: schemas are rendered
into the prompt, and robustness comes from tolerant JSON extraction (markdown
fences, JSON embedded in prose, alias keys), schema validation with type
coercion (`"7"` → `7`), and validation-error feedback on retries:

```python
completion = local_agent.complete_with_tools("What is 7 + 8?")
```

### Dispatch guarantees

- Unknown tools, missing required arguments, and uncoercible types are
  rejected with actionable error messages (fed back to the model on retry)
- Hallucinated argument names are dropped with a warning
- Adapter exceptions are captured as unsuccessful `ToolResult`s instead of
  crashing the agent loop
- Every dispatch is journaled to the agent context's `function_log`, which is
  captured by state snapshots

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
- `MOONSHOT_API_KEY` - Moonshot API key for Kimi models
- `ZAI_API_KEY` - Z.AI API key for hosted GLM models
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `OPENAI_COMPATIBLE_BASE_URL` - Base `/v1` URL for a self-hosted inference server
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

The legacy `LlamaAgent` and `GPT4Agent` classes have been removed; `LocalAgent`
and `OnlineAgent` are the only agent implementations. The legacy `LLAMA` and
`GPT4AGENT` API agent types remain accepted and are served by `LocalAgent` and
`OnlineAgent` respectively. To migrate code that used the old classes:

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

## Script Parameter Conventions

When working with scripts in the `scripts/` directory (model management, database setup, utilities), follow these command-line parameter conventions:

### Required Conventions

- **Use `--` prefix** (double dash) for all named parameters
- **Use `argparse`** for all argument parsing
- **Provide `--help`** documentation with clear descriptions
- **Include type hints** and default values where appropriate
- **Avoid positional arguments** unless absolutely necessary

### Acceptable Patterns

1. **Named Arguments**
   ```python
   parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="HuggingFace model ID to download")
   ```

2. **Short Aliases**
   ```python
   parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
   parser.add_argument("-c", "--config", type=Path, default="containers.json",
                       help="Configuration file path")
   ```

3. **Boolean Flags**
   ```python
   parser.add_argument("--commit", action="store_true",
                       help="Commit changes to database (default: dry run)")
   parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without writing")
   ```

4. **Choice Arguments**
   ```python
   parser.add_argument("--provider",
                       choices=["openai", "anthropic", "huggingface", "all"],
                       default="all",
                       help="Specific provider to sync")
   ```

5. **Subcommand Architecture**
   ```python
   subparsers = parser.add_subparsers(dest="command", help="Command to execute")
   delete_parser = subparsers.add_parser("delete", help="Delete all weights")
   copy_parser = subparsers.add_parser("copy", help="Copy weights from source")
   copy_parser.add_argument("--source", type=str, help="Source directory")
   ```

### Example Usage

**Good Examples:**
```bash
# Model downloads with named parameters
python scripts/download_models.py --model_id meta-llama/Meta-Llama-3.1-70B-Instruct --use_cli

# Model syncing with provider selection
python scripts/sync_models.py --provider openai --verbose --dry-run

# Database operations with boolean flags
python scripts/insert_presets.py --commit --overwrite

# Subcommand with parameters
python scripts/copy_weights.py copy --source /path/to/weights

# Short aliases
python scripts/generate_compose.py -c custom.json -o docker-compose.prod.yml
```

**Avoid:**
```bash
# Positional arguments without names
python scripts/download_models.py meta-llama/Meta-Llama-3.1-8B-Instruct  # Bad

# Single dash for long names
python scripts/sync_models.py -provider openai  # Bad (should be --provider)

# Unclear abbreviations
python scripts/sync_models.py -p openai  # Unclear (what is -p?)
```

### Script Categories

The `scripts/` directory contains:

- **Model Management**: `download_models.py`, `copy_weights.py`, `sync_models.py`, `list_hf_open_models.py`
- **Environment Setup**: `generate_compose.py`
- **Database Setup**: `insert_presets.py`, `insert_default_user.py`
- **Utilities**: `test_arch_availability.py`, `model_filter_config.py`

All scripts should maintain consistent parameter naming and conventions across categories.

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
   - Pass an explicit API key for custom endpoints

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
