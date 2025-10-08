import { useState, useEffect, useCallback } from 'react';

export interface BackupProvider {
  name: string;
  base_url: string;
  model: string;
  api_key?: string;
  priority: number;
}

export interface UserSettings {
  user_settings_id: number;
  user_id: number;
  default_agent_type: string;
  default_local_model: string;
  default_online_model: string;
  default_online_provider: string;
  default_file_archives: number[];
  enable_rag_by_default: boolean;
  default_max_tokens: number;
  default_temperature: number;
  default_top_p: number;
  default_frequency_penalty: number;
  default_presence_penalty: number;
  backup_providers: BackupProvider[];
  ui_preferences: Record<string, any>;
  create_date: string;
  update_date: string;
}

export interface UserSettingsUpdate {
  default_agent_type?: string;
  default_local_model?: string;
  default_online_model?: string;
  default_online_provider?: string;
  default_file_archives?: number[];
  enable_rag_by_default?: boolean;
  default_max_tokens?: number;
  default_temperature?: number;
  default_top_p?: number;
  default_frequency_penalty?: number;
  default_presence_penalty?: number;
  backup_providers?: BackupProvider[];
  ui_preferences?: Record<string, any>;
}

interface UseUserSettingsReturn {
  settings: UserSettings | null;
  loading: boolean;
  error: string | null;
  updateSettings: (updates: UserSettingsUpdate) => Promise<void>;
  resetSettings: () => Promise<void>;
  refetch: () => Promise<void>;
}

export const useUserSettings = (): UseUserSettingsReturn => {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSettings = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/v1/user-settings/', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch settings: ${response.statusText}`);
      }

      const data = await response.json();
      setSettings(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch settings';
      setError(errorMessage);
      console.error('Error fetching settings:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const updateSettings = useCallback(async (updates: UserSettingsUpdate) => {
    try {
      setError(null);
      
      const response = await fetch('/api/v1/user-settings/', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error(`Failed to update settings: ${response.statusText}`);
      }

      const data = await response.json();
      setSettings(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update settings';
      setError(errorMessage);
      console.error('Error updating settings:', err);
      throw err; // Re-throw so the caller can handle it
    }
  }, []);

  const resetSettings = useCallback(async () => {
    try {
      setError(null);
      
      const response = await fetch('/api/v1/user-settings/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to reset settings: ${response.statusText}`);
      }

      const data = await response.json();
      setSettings(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to reset settings';
      setError(errorMessage);
      console.error('Error resetting settings:', err);
      throw err;
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  return {
    settings,
    loading,
    error,
    updateSettings,
    resetSettings,
    refetch: fetchSettings,
  };
};

export default useUserSettings;

