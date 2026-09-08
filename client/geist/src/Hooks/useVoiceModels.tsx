import { useEffect, useState } from 'react';

export interface VoiceOption {
  id: string;
  display_name: string;
}

export interface LanguageOption {
  code: string;
  display_name: string;
}

export interface TTSModelInfo {
  id: string;
  display_name: string;
  sample_rate: number;
  supports_streaming: boolean;
  streaming_mode: string;
  supports_instruction_control: boolean;
  supports_voice_cloning: boolean;
  voices: VoiceOption[];
  languages: LanguageOption[];
  artifact?: {
    id: string;
    status: string;
    supported: boolean;
    runtime_ready: boolean;
    runtime_detail?: string | null;
    license?: string | null;
    license_url?: string | null;
  };
}

export interface TTSProviderInfo {
  provider: string;
  display_name: string;
  type: string;
  default_model: string;
  models: TTSModelInfo[];
}

export interface VoiceModelsResponse {
  default_provider: string;
  providers: TTSProviderInfo[];
}

export interface UseVoiceModelsReturn {
  data: VoiceModelsResponse | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetches the supported voice/TTS provider catalog from the backend.
 * The fetch is deferred until `enabled` is true so the chat input does not
 * hit the API unless the user actually opens the voice settings panel.
 */
const useVoiceModels = (enabled: boolean): UseVoiceModelsReturn => {
  const [data, setData] = useState<VoiceModelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let cancelled = false;
    setLoading(true);

    fetch('/api/v1/voice/models')
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load voice models: ${response.statusText}`);
        }
        return response.json();
      })
      .then((json: VoiceModelsResponse) => {
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load voice models');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [enabled]);

  return { data, loading, error };
};

export default useVoiceModels;
