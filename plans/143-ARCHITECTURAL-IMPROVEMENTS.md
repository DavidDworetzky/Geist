# Architectural Improvement Plan: General Purpose Agents

## Overview
Introduce two extensible agent implementations that inherit from `BaseAgent`:
- `OnlineAgent`: routes inference requests to any OpenAI-compatible HTTP endpoint.
- `LocalAgent`: executes local MLX-backed models using the same interface as existing `LlamaAgent` implementations, while supporting swappable local inference backends (e.g., MLX, lightweight vLLM wrappers).

This plan describes architectural updates, supporting abstractions, integration touchpoints, and migration steps needed to make agent selection configurable at instantiation time across the platform.

## Goals & Non-Goals
- **Goals**
  - Establish reusable agent base classes that encapsulate transport-specific concerns (HTTP vs. local execution).
  - Support runtime configuration of model endpoints and credentials for `OnlineAgent`.
  - Provide MLX-first local inference with pluggable runners for `LocalAgent` and enable future vLLM drop-in usage.
  - Ensure existing `BaseAgent` contracts remain satisfied so downstream consumers (schedulers, pipelines, chat UI) require minimal changes.
  - Update developer tooling, documentation, and testing to reflect new agent pathways.
- **Non-Goals**
  - Implement new UI for agent selection (out of scope; ensure API hooks support future UI work).
  - Ship production-ready vLLM integration (provide placeholder interface and scaffolding only).
  - Modify existing agent business logic beyond the necessary refactors to share common utilities.

## Current Architecture Assessment
- `BaseAgent` defines common async execution methods and message handling.
- `LlamaAgent` resides under `agents/` and targets MLX Llama 3 models with direct inference logic in-module.
- Remote API-based agents currently duplicate HTTP handling or hardcode endpoints, limiting reuse.
- No unified configuration layer for selecting local vs. remote inference per workspace or request.

## Proposed Architecture
### 1. Agent Class Hierarchy
- Create `agents/base_agent.py` (existing) extension points:
  - Ensure shared utilities (prompt templating, conversation state) are factored into mixins/helpers.
- Add `agents/online_agent.py` implementing OpenAI-compatible transport:
  - Accept `base_url`, `model`, and optional `api_key`/headers at init.
  - Provide request builder that aligns with `OpenAI` schema (chat/completions) and handles streaming.
  - Support retries, timeout configuration, and JSON schema validation.
  - Supports inference providers and backup inference providers. A backup inference provider is a model provider to try if a primary provider fails.
  - Supports Open AI, Anthropic, Grok and models available on Groq (open source inference)
- Add `agents/local_agent.py` orchestrating local runtimes:
  - Accept `model_id`, `runner_cls`, and runner-specific kwargs.
  - Default runner uses MLX Llama 3 (reuse existing llama code extracted into `architectures/mlx_llama.py`).
  - Expose factory method to register additional runners (e.g., `architectures/vllm_runner.py`).
- Add `agents/database/models/user_settings.py` file to model user settings abstraction that represents default choices around models. 
  - User Settings contains settings for default models, a list of file archives to search for RAG by default, and preference for local or online models. 

### 2. Architecture Runner Abstractions
- Create `architectures/__init__.py` with registry helpers (`register_runner`, `get_runner`).
- Extract current MLX inference logic from `LlamaAgent` into new `architectures/mlx_llama.py` runner class implementing:
  - `load(model_id, device_config)`
  - `generate(prompt, generation_config)`
  - `cleanup()` if needed.
- Stub `architectures/vllm_light.py` outlining expected API for future vLLM integration.

### 3. Configuration & Dependency Injection
- Extend agent factory (if exists) or create `agents/factory.py`:
  - Work with existing completions contract in current endpoint config to specify agent_type and model.
  - Instantiate `OnlineAgent` or `LocalAgent` accordingly.
- Add settings entries (e.g., `AGENT_DEFAULT_TRANSPORT`, `AGENT_ONLINE_ENDPOINTS`, `LOCAL_MODEL_PATHS`).
- Ensure environment variable loading for API keys happens centrally (e.g., via `app/core/config.py`).

### 4. Request/Response Normalization
- Define shared dataclasses or Pydantic models for inputs/outputs consumed by agents (if not existing, otherwise use existing. Identify the existing ones that you're going to use.).
- Guarantee `OnlineAgent` and `LocalAgent` implement identical return structure.
- Update any orchestration components (queues, sessions) to read from normalized metadata (backend, model_id, latency).

### 5. Backward Compatibility & Migration
- Deprecate `LlamaAgent` and `gpt4_agent` direct usage by:
  - Mark deprecated in docstrings and eventually remove once callers migrate.
  - Audit codebase for direct `LlamaAgent` instantiations; route them through factory.
  - Provide migration checklist for configuration files (e.g., `config/agents.yml`).

## Implementation Plan
### Phase 1: Refactor Existing MLX Logic
1. Extract MLX inference code from `agents/llama_agent.py` into `architectures/mlx_llama.py` with runner class.
2. Have `LocalAgent` use the new runner abstraction, `LlamaAgent` can keep the old. 

### Phase 2: Establish Agent Factory and Runner Registry
1. Implement architecture registry utilities under `architectures/__init__.py`.
2. Create `agents/factory.py` returning configured agent instances.
3. Update existing agent instantiation sites to use factory (API endpoints, background jobs, tests).

### Phase 3: Implement `LocalAgent`
1. Create `agents/local_agent.py` inheriting from `BaseAgent`.
2. Hook into runner registry, default to MLX runner when `model_id` matches known MLX models.
3. Add configuration support for selecting runner via config/env.
4. Write unit tests covering prompt execution, error handling, and runner selection.

### Phase 4: Implement `OnlineAgent`
1. Create `agents/online_agent.py` with OpenAI-compatible HTTP client (use `httpx` if available).
2. Support streaming and non-streaming completions; include retry/backoff strategy.
3. Integrate API key retrieval and header injection via dotenv config. 
4. Add integration tests using mocked HTTP server and contract tests with recorded responses.

### Phase 5: Implement `UserSetings`
1. Create User Settings database model. 
2. Create User Settings database migration file with alembic. 
3. Create DTO models for user settings in app >> models. 
4. Create service and CRUD endpoints to allow updating of user settings tied to the default user. 

### Phase 6: Configuration & Documentation Updates
1. Extend settings modules with new agent configuration schemas.
2. Document environment variables and usage in `docs/agents.md` (create if absent).
3. Update developer onboarding docs to explain choosing between local and online agents.

### Phase 7: Quality Assurance
- Unit tests for both agents and runner registry.
- Integration tests simulating chat flows selecting each agent type.


