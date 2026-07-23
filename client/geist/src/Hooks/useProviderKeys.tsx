import { useState, useEffect, useCallback } from 'react';

export interface ProviderKeyStatus {
  id: string;
  name: string;
  description: string;
  api_key_env: string;
  env_configured: boolean;
  has_stored_key: boolean;
  key_hint: string | null;
  supports_base_url: boolean;
  base_url: string | null;
  updated_at: string | null;
}

interface UseProviderKeysReturn {
  providers: ProviderKeyStatus[];
  loading: boolean;
  error: string | null;
  saveKey: (providerId: string, apiKey: string, baseUrl?: string) => Promise<void>;
  removeKey: (providerId: string) => Promise<void>;
  refetch: () => Promise<void>;
}

export const useProviderKeys = (): UseProviderKeysReturn => {
  const [providers, setProviders] = useState<ProviderKeyStatus[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProviders = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/v1/providers/', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch providers: ${response.statusText}`);
      }
      setProviders(await response.json());
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch providers';
      setError(errorMessage);
      console.error('Error fetching providers:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const applyUpdatedStatus = (updated: ProviderKeyStatus) => {
    setProviders((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
  };

  const saveKey = useCallback(async (providerId: string, apiKey: string, baseUrl?: string) => {
    const response = await fetch(`/api/v1/providers/${encodeURIComponent(providerId)}/key`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey, base_url: baseUrl || null }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || `Failed to save key: ${response.statusText}`);
    }
    applyUpdatedStatus(await response.json());
  }, []);

  const removeKey = useCallback(async (providerId: string) => {
    const response = await fetch(`/api/v1/providers/${encodeURIComponent(providerId)}/key`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || `Failed to remove key: ${response.statusText}`);
    }
    applyUpdatedStatus(await response.json());
  }, []);

  useEffect(() => {
    fetchProviders();
  }, [fetchProviders]);

  return { providers, loading, error, saveKey, removeKey, refetch: fetchProviders };
};

export default useProviderKeys;
