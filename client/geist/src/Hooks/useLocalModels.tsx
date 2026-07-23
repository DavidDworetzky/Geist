import { useState, useEffect, useCallback, useRef } from 'react';

export interface LocalModelStatus {
  id: string;
  name: string;
  family: string | null;
  backend: string | null;
  gated: boolean;
  parameter_count: string | null;
  downloaded: boolean;
  weights_path: string;
  size_bytes: number;
  download_status: string | null;
  download_job_id: number | null;
  download_error: string | null;
}

export interface DetectedWeightsDirectory {
  directory: string;
  weights_path: string;
  size_bytes: number;
}

export interface LocalModelsResponse {
  models: LocalModelStatus[];
  detected_directories: DetectedWeightsDirectory[];
  weights_root: string;
}

interface UseLocalModelsReturn {
  localModels: LocalModelStatus[];
  detectedDirectories: DetectedWeightsDirectory[];
  weightsRoot: string | null;
  loading: boolean;
  error: string | null;
  startDownload: (modelId: string) => Promise<void>;
  refetch: () => Promise<void>;
}

const ACTIVE_STATUSES = ['queued', 'running'];
const POLL_INTERVAL_MS = 4000;

export const useLocalModels = (): UseLocalModelsReturn => {
  const [localModels, setLocalModels] = useState<LocalModelStatus[]>([]);
  const [detectedDirectories, setDetectedDirectories] = useState<DetectedWeightsDirectory[]>([]);
  const [weightsRoot, setWeightsRoot] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchLocalModels = useCallback(async () => {
    try {
      setError(null);

      const response = await fetch('/api/v1/models/local', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch local models: ${response.statusText}`);
      }
      const data: LocalModelsResponse = await response.json();
      setLocalModels(data.models);
      setDetectedDirectories(data.detected_directories);
      setWeightsRoot(data.weights_root);

      const hasActiveDownload = data.models.some(
        (model) => model.download_status && ACTIVE_STATUSES.includes(model.download_status)
      );
      if (pollTimer.current) {
        clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
      if (hasActiveDownload) {
        pollTimer.current = setTimeout(() => void fetchLocalModels(), POLL_INTERVAL_MS);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch local models';
      setError(errorMessage);
      console.error('Error fetching local models:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const startDownload = useCallback(async (modelId: string) => {
    const response = await fetch('/api/v1/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || `Failed to start download: ${response.statusText}`);
    }
    await fetchLocalModels();
  }, [fetchLocalModels]);

  useEffect(() => {
    fetchLocalModels();
    return () => {
      if (pollTimer.current) {
        clearTimeout(pollTimer.current);
      }
    };
  }, [fetchLocalModels]);

  return {
    localModels,
    detectedDirectories,
    weightsRoot,
    loading,
    error,
    startDownload,
    refetch: fetchLocalModels,
  };
};

export default useLocalModels;
