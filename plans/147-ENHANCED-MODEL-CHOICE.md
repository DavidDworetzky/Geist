# Enhanced Online Model Detection Plan

## Overview
Implement a dynamic model discovery system that augments the statically defined models in our model registry with additional models available from Anthropic and OpenAI APIs. This includes a script for fetching and updating available models, backend endpoints for serving model lists, and frontend synchronization through a new hook.

## Problem Statement

### Current State
- Models are statically defined in `client/geist/src/Components/AgentConfigSection.tsx`
- When providers release new models, manual code updates are required
- No validation that configured models are actually available
- No visibility into which models the user's API keys can access

### Goals
1. Automatically discover available models from Anthropic and OpenAI APIs
2. Provide a script to update model registry with newly available models
3. Synchronize frontend model options with backend-defined models
4. Reduce maintenance burden when new models are released
5. Support fallback to static models when API discovery fails
6. Set families of models that are supported (such as kimi k2, glm, qwen, grok, claude/anthropic, openai) by the script, which we search for on huggingface to implement into the registry.

## Architecture

### High-Level Flow
1. Open AI API, models as discovery source
2. Anthropic API, models as discovery source
3. Huggingface models source for offline models as discovery source. 
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Model Discovery Flow                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐  │
│  │ OpenAI API  │     │Anthropic API│     │ Static Registry     │  │
│  │ /v1/models  │     │ /v1/models  │     │ (fallback)          │  │
│  └──────┬──────┘     └──────┬──────┘     └──────────┬──────────┘  │
│         │                   │                       │              │
│         ▼                   ▼                       │              │
│  ┌─────────────────────────────────────────────────▼─────────────┐│
│  │              scripts/sync_models.py                            ││
│  │  - Fetches models from provider APIs                          ││
│  │  - Filters for supported/recommended models                   ││
│  │  - Updates model registry files                               ││
│  └───────────────────────────────────────────────────────────────┘│
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │           agents/model_registry.py (New)                     │  │
│  │  - Defines ModelInfo dataclass                               │  │
│  │  - Static + dynamic model definitions                        │  │
│  │  - Provider-to-models mapping                                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │           GET /api/v1/models/                                │  │
│  │  - Returns available models per provider                     │  │
│  │  - Optional: validates against user's API keys               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │           useAvailableModels() Hook                          │  │
│  │  - Fetches models from backend                               │  │
│  │  - Provides typed model options per provider                 │  │
│  │  - Handles loading/error states                              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │           AgentConfigSection.tsx                             │  │
│  │  - Consumes useAvailableModels()                             │  │
│  │  - Renders dynamic model dropdowns                           │  │
│  │  - Filters based on selected provider                        │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Backend Model Registry

#### 1.1 Create Model Registry Module
**File**: `agents/model_registry.py`

**Purpose**: Central source of truth for available models

**Structure**:
```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OFFLINE = "offline"

@dataclass
class ModelInfo:
    id: str                          # Model identifier (e.g., "gpt-4")
    name: str                        # Display name (e.g., "GPT-4")
    provider: Provider               # Provider enum
    context_window: Optional[int]    # Max context tokens
    max_output_tokens: Optional[int] # Max output tokens
    supports_vision: bool = False    # Multimodal support
    supports_function_calling: bool = False
    deprecated: bool = False         # Mark deprecated models
    recommended: bool = False        # Highlight recommended models

# Static registry - always available as fallback
STATIC_MODELS: dict[Provider, list[ModelInfo]] = {
    Provider.OPENAI: [
        ModelInfo(id="gpt-4", name="GPT-4", provider=Provider.OPENAI, ...),
        ModelInfo(id="gpt-4-turbo", name="GPT-4 Turbo", provider=Provider.OPENAI, ...),
        ModelInfo(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", provider=Provider.OPENAI, ...),
    ],
    Provider.ANTHROPIC: [
        ModelInfo(id="claude-3-opus-20240229", name="Claude 3 Opus", provider=Provider.ANTHROPIC, ...),
        ModelInfo(id="claude-3-sonnet-20240229", name="Claude 3 Sonnet", provider=Provider.ANTHROPIC, ...),
    ],
}

# Dynamic registry - populated by sync script
# Auto-generated - do not edit manually
DISCOVERED_MODELS: dict[Provider, list[ModelInfo]] = {}

def get_models_for_provider(provider: Provider) -> list[ModelInfo]:
    """Get all available models for a provider, preferring discovered models."""
    ...

def get_all_models() -> dict[Provider, list[ModelInfo]]:
    """Get all available models grouped by provider."""
    ...
```

#### 1.2 Create Models API Endpoint
**File**: `app/api/v1/endpoints/models.py`

**Endpoints**:
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/models/` | Get all available models |
| GET | `/api/v1/models/{provider}` | Get models for a specific provider |

**Response Schema**:
```python
class ModelResponse(BaseModel):
    id: str
    name: str
    provider: str
    context_window: Optional[int]
    max_output_tokens: Optional[int]
    supports_vision: bool
    supports_function_calling: bool
    deprecated: bool
    recommended: bool

class ModelsListResponse(BaseModel):
    providers: dict[str, list[ModelResponse]]
    last_updated: Optional[datetime]
```

### Phase 2: Model Sync Script

#### 2.1 Create Model Synchronization Script
**File**: `scripts/sync_models.py`

**Purpose**: Fetch available models from provider APIs and update registry

**Features**:
- Reads API keys from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`)
- Queries provider model listing endpoints
- Filters models based on configurable criteria (e.g., exclude deprecated, fine-tuned)
- Generates/updates the `DISCOVERED_MODELS` section in `model_registry.py`
- Supports dry-run mode for preview
- Provides diff output showing added/removed models
- Optionally updates frontend static definitions

**Usage**:
```bash
# List available models (dry run)
python scripts/sync_models.py --dry-run

# Update backend registry only
python scripts/sync_models.py --update-backend

# Update both backend and frontend
python scripts/sync_models.py --update-all

# Filter by provider
python scripts/sync_models.py --provider openai --update-backend

# Show detailed model information
python scripts/sync_models.py --verbose
```

**Implementation Details**:
```python
"""
Model Synchronization Script

Fetches available models from Anthropic and OpenAI APIs and updates
the model registry with newly discovered models.

Environment Variables:
    ANTHROPIC_API_KEY: API key for Anthropic
    OPENAI_API_KEY: API key for OpenAI

Usage:
    python scripts/sync_models.py --dry-run
    python scripts/sync_models.py --update-all
"""

import os
import argparse
import requests
from typing import Optional

def fetch_openai_models(api_key: str) -> list[dict]:
    """Fetch available models from OpenAI API."""
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    response.raise_for_status()
    models = response.json()["data"]
    
    # Filter for chat models only
    chat_models = [m for m in models if is_chat_model(m["id"])]
    return chat_models

def fetch_anthropic_models(api_key: str) -> list[dict]:
    """Fetch available models from Anthropic API."""
    response = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
    )
    response.raise_for_status()
    return response.json()["data"]

def is_chat_model(model_id: str) -> bool:
    """Filter for chat-capable models."""
    # Include GPT models, exclude embeddings, whisper, etc.
    chat_prefixes = ["gpt-4", "gpt-3.5", "o1", "o3"]
    exclude_patterns = ["instruct", "embedding", "whisper", "tts", "dall-e"]
    
    if any(pattern in model_id.lower() for pattern in exclude_patterns):
        return False
    return any(model_id.startswith(prefix) for prefix in chat_prefixes)

def generate_model_registry_code(models: dict) -> str:
    """Generate Python code for DISCOVERED_MODELS."""
    ...

def update_backend_registry(models: dict, dry_run: bool = False) -> None:
    """Update agents/model_registry.py with discovered models."""
    ...

def update_frontend_models(models: dict, dry_run: bool = False) -> None:
    """Optionally update frontend static definitions."""
    # This is optional - frontend can also fetch from API
    ...

def main():
    parser = argparse.ArgumentParser(description="Sync available models from providers")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--update-backend", action="store_true", help="Update backend registry")
    parser.add_argument("--update-frontend", action="store_true", help="Update frontend definitions")
    parser.add_argument("--update-all", action="store_true", help="Update both backend and frontend")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="Sync specific provider")
    parser.add_argument("--verbose", action="store_true", help="Show detailed model info")
    
    args = parser.parse_args()
    ...
```

#### 2.2 Model Filtering Configuration
**File**: `scripts/model_filter_config.py`

**Purpose**: Configure which models to include/exclude

```python
# Models to always include regardless of API response
ALWAYS_INCLUDE = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

# Patterns to exclude (regex)
EXCLUDE_PATTERNS = [
    r".*-instruct$",       # Exclude instruct variants (handled differently)
    r".*-vision-preview$", # Exclude preview versions
    r"ft:.*",              # Exclude fine-tuned models
]

# Model metadata overrides
MODEL_METADATA = {
    "gpt-4": {
        "name": "GPT-4",
        "recommended": True,
        "context_window": 8192,
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "recommended": True,
        "context_window": 128000,
    },
    # ... more overrides
}
```

### Phase 3: Frontend Integration

#### 3.1 Create Available Models Hook
**File**: `client/geist/src/Hooks/useAvailableModels.tsx`

**Purpose**: Fetch and manage available models from backend

```typescript
import { useState, useEffect, useCallback } from 'react';

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  context_window: number | null;
  max_output_tokens: number | null;
  supports_vision: boolean;
  supports_function_calling: boolean;
  deprecated: boolean;
  recommended: boolean;
}

export interface AvailableModels {
  providers: Record<string, ModelInfo[]>;
  lastUpdated: string | null;
}

interface UseAvailableModelsResult {
  models: AvailableModels | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  getModelsForProvider: (provider: string) => ModelInfo[];
}

export function useAvailableModels(): UseAvailableModelsResult {
  const [models, setModels] = useState<AvailableModels | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('/api/v1/models/');
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setModels(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      // Fall back to static models defined in component
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const getModelsForProvider = useCallback((provider: string): ModelInfo[] => {
    if (!models?.providers) return [];
    return models.providers[provider] || [];
  }, [models]);

  return { models, loading, error, refetch: fetchModels, getModelsForProvider };
}
```

#### 3.2 Update AgentConfigSection Component
**File**: `client/geist/src/Components/AgentConfigSection.tsx`

**Changes**:
- Import and use `useAvailableModels` hook
- Replace static `modelsByProvider` with dynamic data
- Add loading state for model dropdown
- Provide fallback to static models if API fails

```typescript
import { useAvailableModels } from '../Hooks/useAvailableModels';

const AgentConfigSection: React.FC<AgentConfigSectionProps> = ({...}) => {
  const { models, loading: modelsLoading, getModelsForProvider } = useAvailableModels();

  // Static fallback models (used if API fails)
  const staticModelsByProvider = {
    openai: [...],
    anthropic: [...],
    custom: [...]
  };

  // Get models for current provider (dynamic or static fallback)
  const getModelOptions = (provider: string) => {
    const dynamicModels = getModelsForProvider(provider);
    if (dynamicModels.length > 0) {
      return dynamicModels.map(m => ({ value: m.id, label: m.name }));
    }
    return staticModelsByProvider[provider] || [];
  };

  const onlineModelOptions = getModelOptions(onlineProvider);
  
  // ... rest of component
};
```

### Phase 4: Runtime Model Validation (Optional)

#### 4.1 Model Availability Check
**Purpose**: Validate that configured models are accessible with user's API keys

**Endpoint**: `POST /api/v1/models/validate`

**Request**:
```json
{
  "provider": "openai",
  "model_id": "gpt-4"
}
```

**Response**:
```json
{
  "valid": true,
  "model_id": "gpt-4",
  "provider": "openai",
  "error": null
}
```

**Use Cases**:
- Validate model selection before saving settings
- Check if user has access to premium models (e.g., GPT-4)
- Graceful fallback suggestions if model is unavailable

## Implementation Checklist

### Phase 1: Backend Model Registry
- [ ] Create `agents/model_registry.py` with ModelInfo dataclass
- [ ] Define static model registry with current models
- [ ] Add provider enum and helper functions
- [ ] Create `app/api/v1/endpoints/models.py`
- [ ] Add Pydantic models for API responses
- [ ] Register router in `app/main.py`
- [ ] Write unit tests for model registry

### Phase 2: Model Sync Script
- [ ] Create `scripts/sync_models.py`
- [ ] Implement OpenAI model fetching
- [ ] Implement Anthropic model fetching
- [ ] Add model filtering logic
- [ ] Implement code generation for registry update
- [ ] Add dry-run and verbose modes
- [ ] Create `scripts/model_filter_config.py`
- [ ] Add documentation for script usage
- [ ] Test with real API keys

### Phase 3: Frontend Integration
- [ ] Create `useAvailableModels.tsx` hook
- [ ] Define TypeScript interfaces for models
- [ ] Update `AgentConfigSection.tsx` to use hook
- [ ] Add loading states for model dropdown
- [ ] Implement static model fallback
- [ ] Update tests for dynamic model loading
- [ ] Test provider switching with dynamic models

### Phase 4: Runtime Validation (Optional)
- [ ] Add model validation endpoint
- [ ] Implement API key validation logic
- [ ] Add frontend validation before save
- [ ] Create fallback suggestion system

## File Structure Summary

```
├── agents/
│   └── model_registry.py              # Central model registry
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── models.py          # Models API endpoint
│   └── models/
│       └── model.py                   # Pydantic models for API
├── scripts/
│   ├── sync_models.py                 # Model sync script
│   └── model_filter_config.py         # Filter configuration
├── client/geist/src/
│   ├── Hooks/
│   │   └── useAvailableModels.tsx     # Models hook
│   └── Components/
│       └── AgentConfigSection.tsx     # Updated component
└── tests/
    ├── test_model_registry.py         # Registry tests
    └── test_models_api.py             # API tests
```

## API Reference

### Provider Model Listing APIs

#### OpenAI
```
GET https://api.openai.com/v1/models
Headers:
  Authorization: Bearer {OPENAI_API_KEY}
```

#### Anthropic
```
GET https://api.anthropic.com/v1/models
Headers:
  x-api-key: {ANTHROPIC_API_KEY}
  anthropic-version: 2023-06-01
```

### Geist API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/models/` | Get all available models |
| GET | `/api/v1/models/{provider}` | Get models for specific provider |
| POST | `/api/v1/models/refresh` | Refresh model cache from APIs |
| POST | `/api/v1/models/validate` | Validate model availability |

## Configuration

### Environment Variables
```bash
# Required for sync script
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Enable runtime model discovery
ENABLE_DYNAMIC_MODEL_DISCOVERY=true
MODEL_CACHE_TTL_SECONDS=3600
```

## Error Handling

### Sync Script Failures
- If API is unreachable: Log warning, continue with other providers
- If API key is invalid: Log error, skip provider
- If no models returned: Preserve existing static models

### Runtime Failures
- If models endpoint fails: Frontend falls back to static models
- If validation fails: Show warning but allow selection
- If model becomes unavailable: Suggest alternative from same provider

## Testing Strategy

### Unit Tests
- Model registry helper functions
- Model filtering logic
- API response parsing

### Integration Tests
- Models API endpoint responses
- Model validation endpoint
- Frontend hook with mocked API

### Manual Testing
- Run sync script with real API keys
- Verify models appear in frontend dropdown
- Test provider switching behavior
- Verify fallback to static models

## Migration Path

1. **Phase 1**: Deploy backend model registry and API (backward compatible)
2. **Phase 2**: Deploy sync script, run initial sync
3. **Phase 3**: Update frontend to use dynamic models with static fallback
4. **Phase 4**: (Optional) Enable runtime validation

## Success Criteria

- [ ] Sync script successfully fetches models from OpenAI and Anthropic
- [ ] New models appear in registry after running sync script
- [ ] Frontend displays dynamically fetched models
- [ ] Fallback to static models works when API unavailable
- [ ] Model filtering correctly shows only provider-relevant models
- [ ] No regression in existing model selection functionality
- [ ] Documentation updated with sync script usage

## Future Enhancements

1. **Automatic Sync**: Scheduled job to periodically sync models
2. **Model Capabilities**: Enhanced filtering by capabilities (vision, function calling)
3. **Cost Information**: Display pricing information per model
4. **Usage Recommendations**: Suggest models based on use case
5. **Additional Providers**: Extend to support Google (Gemini), Mistral, etc.
6. **Model Comparison**: UI for comparing model capabilities side-by-side

