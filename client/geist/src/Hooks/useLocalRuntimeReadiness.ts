import { useCallback, useEffect, useState } from 'react';
import { ModelLoadStatus } from '../chatTypes';
import { UserSettings } from './useUserSettings';

interface LocalRuntimeReadiness {
  status: ModelLoadStatus | null;
  retry: () => void;
}

const failedStatus = (modelId: string, detail: string): ModelLoadStatus => ({
  model_id: modelId,
  state: 'failed',
  detail,
  started_at: null,
  updated_at: new Date().toISOString(),
});

export default function useLocalRuntimeReadiness(
  settings: UserSettings | null,
  enabled = true,
): LocalRuntimeReadiness {
  const [status, setStatus] = useState<ModelLoadStatus | null>(null);
  const [retrySequence, setRetrySequence] = useState(0);
  const retry = useCallback(() => setRetrySequence(current => current + 1), []);

  useEffect(() => {
    if (!enabled || settings?.default_agent_type !== 'local' || !settings.default_local_model) {
      setStatus(null);
      return undefined;
    }

    const modelId = settings.default_local_model;
    let stopped = false;
    let pollTimer: number | undefined;

    const updateStatus = (nextStatus: ModelLoadStatus) => {
      if (!stopped) setStatus(nextStatus);
    };

    const poll = async () => {
      try {
        const response = await fetch(
          `/api/v1/models/status/${encodeURIComponent(modelId)}`,
          { headers: { 'Content-Type': 'application/json' } },
        );
        if (!response.ok) throw new Error(`status ${response.status}`);
        const nextStatus = await response.json() as ModelLoadStatus;
        updateStatus(nextStatus);
        if (!stopped && nextStatus.state === 'loading') {
          pollTimer = window.setTimeout(poll, 750);
        }
      } catch (error) {
        updateStatus(failedStatus(
          modelId,
          `Could not check local model readiness: ${error instanceof Error ? error.message : error}`,
        ));
      }
    };

    const start = async () => {
      updateStatus({
        model_id: modelId,
        state: 'loading',
        detail: 'Checking and starting the configured local model.',
        started_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
      try {
        const response = await fetch('/api/v1/models/local/runtime/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
        if (!response.ok) throw new Error(`start failed with status ${response.status}`);
        const nextStatus = await response.json() as ModelLoadStatus;
        updateStatus(nextStatus);
        if (!stopped && nextStatus.state === 'loading') {
          pollTimer = window.setTimeout(poll, 250);
        }
      } catch (error) {
        updateStatus(failedStatus(
          modelId,
          `Could not start the local model: ${error instanceof Error ? error.message : error}`,
        ));
      }
    };

    void start();
    return () => {
      stopped = true;
      if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    };
  }, [
    enabled,
    retrySequence,
    settings?.default_agent_type,
    settings?.default_local_artifact_id,
    settings?.default_local_model,
  ]);

  return { status, retry };
}
