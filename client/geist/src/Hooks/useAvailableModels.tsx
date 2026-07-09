import { useState, useEffect, useCallback } from 'react';

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  context_window: number | null;
  max_output_tokens: number | null;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supports_streaming: boolean;
  recommended: boolean;
  family: string | null;
}

export interface AvailableModels {
  providers: Record<string, ModelInfo[]>;
  last_updated: string | null;
}

interface UseAvailableModelsReturn {
  models: AvailableModels | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  getModelsForProvider: (provider: string) => ModelInfo[];
  getModelById: (modelId: string) => ModelInfo | undefined;
  getRecommendedModels: (provider?: string) => ModelInfo[];
  providers: string[];
}

// Static fallback models in case the API fails
const STATIC_MODELS: AvailableModels = {
  providers: {
    openai: [
      {
        id: 'gpt-4',
        name: 'GPT-4',
        provider: 'openai',
        context_window: 8192,
        max_output_tokens: 4096,
        supports_vision: false,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: false,
        family: 'gpt-4',
      },
      {
        id: 'gpt-4-turbo',
        name: 'GPT-4 Turbo',
        provider: 'openai',
        context_window: 128000,
        max_output_tokens: 4096,
        supports_vision: true,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'gpt-4',
      },
      {
        id: 'gpt-4o',
        name: 'GPT-4o',
        provider: 'openai',
        context_window: 128000,
        max_output_tokens: 16384,
        supports_vision: true,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'gpt-4o',
      },
      {
        id: 'gpt-4o-mini',
        name: 'GPT-4o Mini',
        provider: 'openai',
        context_window: 128000,
        max_output_tokens: 16384,
        supports_vision: true,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'gpt-4o',
      },
      {
        id: 'gpt-3.5-turbo',
        name: 'GPT-3.5 Turbo',
        provider: 'openai',
        context_window: 16385,
        max_output_tokens: 4096,
        supports_vision: false,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: false,
        family: 'gpt-3.5',
      },
    ],
    anthropic: [
      {
        id: 'claude-3-opus-20240229',
        name: 'Claude 3 Opus',
        provider: 'anthropic',
        context_window: 200000,
        max_output_tokens: 4096,
        supports_vision: true,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'claude-3',
      },
      {
        id: 'claude-3-sonnet-20240229',
        name: 'Claude 3 Sonnet',
        provider: 'anthropic',
        context_window: 200000,
        max_output_tokens: 4096,
        supports_vision: true,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'claude-3',
      },
      {
        id: 'claude-3-5-sonnet-20241022',
        name: 'Claude 3.5 Sonnet',
        provider: 'anthropic',
        context_window: 200000,
        max_output_tokens: 8192,
        supports_vision: true,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'claude-3.5',
      },
    ],
    xai: [
      {
        id: 'grok-2',
        name: 'Grok 2',
        provider: 'xai',
        context_window: 131072,
        max_output_tokens: 32768,
        supports_vision: false,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'grok',
      },
      {
        id: 'grok-3',
        name: 'Grok 3',
        provider: 'xai',
        context_window: 131072,
        max_output_tokens: 32768,
        supports_vision: false,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'grok',
      },
    ],
    groq: [
      {
        id: 'llama-3.3-70b-versatile',
        name: 'Llama 3.3 70B Versatile',
        provider: 'groq',
        context_window: 128000,
        max_output_tokens: 32768,
        supports_vision: false,
        supports_function_calling: true,
        supports_streaming: true,
        recommended: true,
        family: 'llama-3',
      },
    ],
    offline: [
      {
        id: 'Meta-Llama-3.1-8B-Instruct',
        name: 'Meta Llama 3.1 8B Instruct (Local)',
        provider: 'offline',
        context_window: 131072,
        max_output_tokens: 8192,
        supports_vision: false,
        supports_function_calling: false,
        supports_streaming: true,
        recommended: true,
        family: 'llama-3',
      },
    ],
  },
  last_updated: null,
};

export const useAvailableModels = (): UseAvailableModelsReturn => {
  const [models, setModels] = useState<AvailableModels | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/v1/models/', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }

      const data = await response.json();
      setModels(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch models';
      setError(errorMessage);
      console.error('Error fetching models:', err);
      // Use static fallback on error
      setModels(STATIC_MODELS);
    } finally {
      setLoading(false);
    }
  }, []);

  const getModelsForProvider = useCallback((provider: string): ModelInfo[] => {
    const source = models || STATIC_MODELS;
    return source.providers[provider] || [];
  }, [models]);

  const getModelById = useCallback((modelId: string): ModelInfo | undefined => {
    const source = models || STATIC_MODELS;
    for (const providerModels of Object.values(source.providers)) {
      const found = providerModels.find(m => m.id === modelId);
      if (found) return found;
    }
    return undefined;
  }, [models]);

  const getRecommendedModels = useCallback((provider?: string): ModelInfo[] => {
    const source = models || STATIC_MODELS;
    let allModels: ModelInfo[] = [];

    if (provider) {
      allModels = source.providers[provider] || [];
    } else {
      for (const providerModels of Object.values(source.providers)) {
        allModels = allModels.concat(providerModels);
      }
    }

    return allModels.filter(m => m.recommended);
  }, [models]);

  const providers = models
    ? Object.keys(models.providers)
    : Object.keys(STATIC_MODELS.providers);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return {
    models,
    loading,
    error,
    refetch: fetchModels,
    getModelsForProvider,
    getModelById,
    getRecommendedModels,
    providers,
  };
};

export default useAvailableModels;
