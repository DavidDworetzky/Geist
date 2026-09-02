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
  loaded: boolean;
  error: string | null;
  refreshLocalArtifacts: () => Promise<void>;
  downloadArtifact: (artifact: LocalArtifact) => Promise<void>;
  activateArtifact: (artifact: LocalArtifact) => Promise<void>;
}

interface UseLocalArtifactsOptions {
  enabled?: boolean;
  pollWhileBusy?: boolean;
  pollModelId?: string | null;
}

type ArtifactListener = (artifacts: LocalArtifact[]) => void;

const artifactListeners = new Set<ArtifactListener>();
let pendingArtifactRequest: Promise<LocalArtifact[]> | null = null;
const pendingArtifactDownloads = new Map<string, Promise<void>>();
let artifactCache: LocalArtifact[] = [];
let artifactCatalogLoaded = false;

function publishArtifacts(artifacts: LocalArtifact[]): void {
  artifactCache = artifacts;
  artifactCatalogLoaded = true;
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

async function requestArtifactDownload(artifactId: string): Promise<void> {
  const existingRequest = pendingArtifactDownloads.get(artifactId);
  if (existingRequest) {
    return existingRequest;
  }

  const request = (async () => {
    const response = await fetch(`/api/v1/models/local/artifacts/${artifactId}/download`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error(`Model download failed: ${response.statusText}`);
    }

    const updatedArtifact = await response.json();
    publishArtifacts(artifactCache.map(artifact => (
      artifact.id === artifactId ? { ...artifact, ...updatedArtifact } : artifact
    )));
  })().finally(() => {
    pendingArtifactDownloads.delete(artifactId);
  });

  pendingArtifactDownloads.set(artifactId, request);
  return request;
}

export default function useLocalArtifacts({
  enabled = true,
  pollWhileBusy = false,
  pollModelId = null,
}: UseLocalArtifactsOptions = {}): UseLocalArtifactsReturn {
  const { updateSettings } = useUserSettings();
  const [artifacts, setArtifacts] = useState<LocalArtifact[]>(artifactCache);
  const [loading, setLoading] = useState(enabled && !artifactCatalogLoaded);
  const [loaded, setLoaded] = useState(!enabled || artifactCatalogLoaded);
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
      setLoaded(true);
    };
    artifactListeners.add(handleArtifacts);

    if (enabled) {
      setLoaded(artifactCatalogLoaded);
      void refreshLocalArtifacts();
    } else {
      setLoading(false);
      setLoaded(true);
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
      && (!pollModelId || artifact.model_id === pollModelId)
    ))) {
      return undefined;
    }
    const interval = window.setInterval(() => void refreshLocalArtifacts(), 1000);
    return () => window.clearInterval(interval);
  }, [artifacts, enabled, pollModelId, pollWhileBusy, refreshLocalArtifacts]);

  const downloadArtifact = useCallback(async (artifact: LocalArtifact) => {
    await requestArtifactDownload(artifact.id);
  }, []);

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
    loaded,
    error,
    refreshLocalArtifacts,
    downloadArtifact,
    activateArtifact,
  };
}
