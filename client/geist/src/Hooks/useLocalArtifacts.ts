import { useCallback, useEffect, useRef, useState } from 'react';
import useUserSettings from './useUserSettings';

export interface LocalArtifact {
  id: string;
  model_id: string;
  display_name: string;
  format: string;
  backend: string;
  quantization?: string | null;
  status: string;
  bytes_downloaded: number;
  total_bytes?: number | null;
  source: string;
  error?: string | null;
  supported?: boolean;
  requires_auth?: boolean;
  progress_unit?: 'bytes' | 'files';
  progress_completed?: number | null;
  progress_total?: number | null;
}

interface UseLocalArtifactsReturn {
  artifacts: LocalArtifact[];
  loading: boolean;
  error: string | null;
  refreshLocalArtifacts: () => Promise<void>;
  activateArtifact: (artifact: LocalArtifact) => Promise<void>;
}

interface UseLocalArtifactsOptions {
  enabled?: boolean;
  pollWhileBusy?: boolean;
}

type ArtifactListener = (artifacts: LocalArtifact[]) => void;

const artifactListeners = new Set<ArtifactListener>();
let pendingArtifactRequest: Promise<LocalArtifact[]> | null = null;

function publishArtifacts(artifacts: LocalArtifact[]): void {
  artifactListeners.forEach(listener => listener(artifacts));
}

async function requestLocalArtifacts(): Promise<LocalArtifact[]> {
  if (!pendingArtifactRequest) {
    pendingArtifactRequest = (async () => {
      const response = await fetch('/api/v1/models/local/artifacts');
      if (!response.ok) {
        throw new Error(`Local model status failed: ${response.statusText}`);
      }
      const payload = await response.json();
      const artifacts = payload.artifacts ?? [];
      publishArtifacts(artifacts);
      return artifacts;
    })().finally(() => {
      pendingArtifactRequest = null;
    });
  }
  return pendingArtifactRequest;
}

export default function useLocalArtifacts({
  enabled = true,
  pollWhileBusy = false,
}: UseLocalArtifactsOptions = {}): UseLocalArtifactsReturn {
  const { updateSettings } = useUserSettings();
  const [artifacts, setArtifacts] = useState<LocalArtifact[]>([]);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refreshLocalArtifacts = useCallback(async () => {
    if (mounted.current) {
      setLoading(true);
      setError(null);
    }
    try {
      await requestLocalArtifacts();
    } catch (requestError) {
      if (mounted.current) {
        setError(requestError instanceof Error ? requestError.message : 'Local model status failed');
      }
    } finally {
      if (mounted.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const handleArtifacts = (nextArtifacts: LocalArtifact[]) => {
      setArtifacts(nextArtifacts);
      setError(null);
      setLoading(false);
    };
    artifactListeners.add(handleArtifacts);

    if (enabled) {
      void refreshLocalArtifacts();
    } else {
      setLoading(false);
    }

    const handleFocus = () => {
      if (enabled) void refreshLocalArtifacts();
    };
    window.addEventListener('focus', handleFocus);

    return () => {
      artifactListeners.delete(handleArtifacts);
      window.removeEventListener('focus', handleFocus);
    };
  }, [enabled, refreshLocalArtifacts]);

  useEffect(() => {
    if (!enabled || !pollWhileBusy || !artifacts.some(artifact => (
      ['queued', 'downloading', 'cancelling'].includes(artifact.status)
    ))) {
      return undefined;
    }
    const interval = window.setInterval(() => void refreshLocalArtifacts(), 1000);
    return () => window.clearInterval(interval);
  }, [artifacts, enabled, pollWhileBusy, refreshLocalArtifacts]);

  const activateArtifact = useCallback(async (artifact: LocalArtifact) => {
    await updateSettings({
      default_agent_type: 'local',
      default_local_model: artifact.model_id,
      default_local_artifact_id: artifact.id,
    });
  }, [updateSettings]);

  return {
    artifacts,
    loading,
    error,
    refreshLocalArtifacts,
    activateArtifact,
  };
}
