import React, { ChangeEvent, useCallback, useEffect, useState } from 'react';
import useAvailableModels, { ModelInfo } from './Hooks/useAvailableModels';
import useUserSettings from './Hooks/useUserSettings';

function formatNumber(value: number | null): string {
  if (!value) {
    return 'Unknown';
  }
  return new Intl.NumberFormat().format(value);
}

function capabilityLabels(model: ModelInfo): string[] {
  const labels = [];
  if (model.supports_streaming) labels.push('Streaming');
  if (model.supports_function_calling) labels.push('Tools');
  if (model.supports_vision) labels.push('Vision');
  if (model.recommended) labels.push('Recommended');
  return labels;
}

interface LocalArtifact {
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

function formatBytes(value?: number | null): string {
  if (!value) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const unit = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / (1024 ** unit)).toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export default function Models(): JSX.Element {
  const { models, loading, error, refetch, providers } = useAvailableModels();
  const { settings, updateSettings } = useUserSettings();
  const [localArtifacts, setLocalArtifacts] = useState<LocalArtifact[]>([]);
  const [localError, setLocalError] = useState<string | null>(null);
  const [localAction, setLocalAction] = useState<string | null>(null);

  const refreshLocalArtifacts = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/models/local/artifacts');
      if (!response.ok) throw new Error(`Local model status failed: ${response.statusText}`);
      const payload = await response.json();
      setLocalArtifacts(payload.artifacts ?? []);
      setLocalError(null);
    } catch (requestError) {
      setLocalError(requestError instanceof Error ? requestError.message : 'Local model status failed');
    }
  }, []);

  useEffect(() => {
    void refreshLocalArtifacts();
  }, [refreshLocalArtifacts]);

  useEffect(() => {
    if (!localArtifacts.some(artifact => ['queued', 'downloading', 'cancelling'].includes(artifact.status))) {
      return undefined;
    }
    const interval = window.setInterval(() => void refreshLocalArtifacts(), 1000);
    return () => window.clearInterval(interval);
  }, [localArtifacts, refreshLocalArtifacts]);

  const runArtifactAction = async (
    artifactId: string,
    action: 'download' | 'cancel' | 'remove',
  ) => {
    setLocalAction(artifactId);
    try {
      const endpoint = action === 'remove'
        ? `/api/v1/models/local/artifacts/${artifactId}`
        : `/api/v1/models/local/artifacts/${artifactId}/${action}`;
      const response = await fetch(endpoint, {
        method: action === 'remove' ? 'DELETE' : 'POST',
      });
      if (!response.ok) throw new Error(`Model ${action} failed: ${response.statusText}`);
      await refreshLocalArtifacts();
    } catch (requestError) {
      setLocalError(requestError instanceof Error ? requestError.message : `Model ${action} failed`);
    } finally {
      setLocalAction(null);
    }
  };

  const importGguf = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setLocalAction('import');
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/v1/models/local/import', { method: 'POST', body });
      if (!response.ok) throw new Error(`GGUF import failed: ${response.statusText}`);
      await refreshLocalArtifacts();
    } catch (requestError) {
      setLocalError(requestError instanceof Error ? requestError.message : 'GGUF import failed');
    } finally {
      event.target.value = '';
      setLocalAction(null);
    }
  };

  const activateArtifact = async (artifact: LocalArtifact) => {
    setLocalAction(artifact.id);
    try {
      await updateSettings({
        default_agent_type: 'local',
        default_local_model: artifact.model_id,
        default_local_artifact_id: artifact.id,
      });
    } catch (requestError) {
      setLocalError(requestError instanceof Error ? requestError.message : 'Model selection failed');
    } finally {
      setLocalAction(null);
    }
  };

  const activeModel = settings?.default_agent_type === 'online'
    ? settings.default_online_model
    : settings?.default_local_model;
  const activeProvider = settings?.default_agent_type === 'online'
    ? settings.default_online_provider
    : 'offline';
  const visibleLocalArtifacts = localArtifacts.filter(artifact => artifact.supported !== false);
  const ggufSupported = visibleLocalArtifacts.some(artifact => artifact.format === 'gguf');
  const mlxSupported = visibleLocalArtifacts.some(artifact => artifact.backend === 'mlx_llama');

  if (loading && !models) {
    return (
      <section className="page-surface page-surface-centered">
        <h2>Loading models</h2>
        <p>Gathering local and online provider options.</p>
      </section>
    );
  }

  return (
    <section className="models-page">
      <div className="page-header">
        <div>
          <p className="section-eyebrow">Providers</p>
          <h2>Model inventory</h2>
          <p>
            Review available models and the current default runtime without leaving the workbench.
          </p>
        </div>
        <button className="button button-secondary" onClick={() => void refetch()}>
          Refresh
        </button>
      </div>

      {error && (
        <div className="notice notice-warning">
          Live model discovery failed, so Geist is showing fallback model data. {error}
        </div>
      )}

      <section className="provider-panel" aria-labelledby="local-model-files-heading">
        <div className="provider-panel-header">
          <div>
            <h3 id="local-model-files-heading">Local model files</h3>
            <p>
              {mlxSupported
                ? 'Download and select the managed MLX snapshot used on Apple silicon.'
                : 'Download a curated GGUF or import one already on this computer.'}
            </p>
          </div>
          {ggufSupported && (
            <label className="button button-secondary">
              {localAction === 'import' ? 'Importing…' : 'Import GGUF'}
              <input
                aria-label="Import GGUF model"
                type="file"
                accept=".gguf"
                disabled={localAction === 'import'}
                onChange={importGguf}
                hidden
              />
            </label>
          )}
        </div>
        {localError && <div className="notice notice-error">{localError}</div>}
        <div className="model-table">
          <div className="model-table-row model-table-heading">
            <span>Artifact</span>
            <span>Format</span>
            <span>Status</span>
            <span>Actions</span>
          </div>
          {visibleLocalArtifacts.map(artifact => {
            const active = settings?.default_local_artifact_id === artifact.id;
            const busy = ['queued', 'downloading', 'cancelling'].includes(artifact.status);
            const supported = artifact.supported !== false;
            const total = artifact.total_bytes ?? 0;
            return (
              <div className="model-table-row" key={artifact.id}>
                <span>
                  <strong>{artifact.display_name}</strong>
                  <small>{artifact.model_id}</small>
                </span>
                <span>{artifact.quantization || artifact.format.toUpperCase()}</span>
                <span>
                  {supported ? artifact.status : 'unavailable on this platform'}
                  {busy && (
                    <small>
                      {artifact.progress_unit === 'files'
                        ? `${artifact.progress_completed ?? 0} / ${artifact.progress_total ?? '?'} files`
                        : `${formatBytes(artifact.bytes_downloaded)} / ${formatBytes(total)}`}
                    </small>
                  )}
                  {artifact.requires_auth && artifact.status !== 'installed' && (
                    <small>Requires accepted Hugging Face access and an HF token.</small>
                  )}
                  {artifact.error && <small>{artifact.error}</small>}
                </span>
                <span className="settings-inline-actions">
                  {artifact.status === 'installed' ? (
                    <>
                      <button
                        className="button button-secondary button-small"
                        disabled={!supported || active || localAction === artifact.id}
                        onClick={() => void activateArtifact(artifact)}
                      >
                        {active ? 'Active' : 'Use'}
                      </button>
                      <button
                        className="button button-secondary button-small"
                        disabled={active || localAction === artifact.id}
                        title={active ? 'Select another model before removing this one.' : undefined}
                        onClick={() => void runArtifactAction(artifact.id, 'remove')}
                      >
                        Remove
                      </button>
                    </>
                  ) : busy ? (
                    <button
                      className="button button-secondary button-small"
                      disabled={localAction === artifact.id}
                      onClick={() => void runArtifactAction(artifact.id, 'cancel')}
                    >
                      Cancel
                    </button>
                  ) : (
                    <>
                      <button
                        className="button button-secondary button-small"
                        disabled={!supported || localAction === artifact.id || artifact.source === 'imported'}
                        onClick={() => void runArtifactAction(artifact.id, 'download')}
                      >
                        Download
                      </button>
                      {artifact.bytes_downloaded > 0 && (
                        <button
                          className="button button-secondary button-small"
                          disabled={localAction === artifact.id}
                          onClick={() => void runArtifactAction(artifact.id, 'remove')}
                        >
                          Clear
                        </button>
                      )}
                    </>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      </section>

      <div className="model-summary-grid">
        <article className="metric-card">
          <span className="metric-label">Active provider</span>
          <strong>{activeProvider || 'Not selected'}</strong>
        </article>
        <article className="metric-card">
          <span className="metric-label">Active model</span>
          <strong>{activeModel || 'Not selected'}</strong>
        </article>
        <article className="metric-card">
          <span className="metric-label">Providers</span>
          <strong>{providers.length}</strong>
        </article>
      </div>

      <div className="model-inventory-scroll" role="region" aria-label="Model inventory">
        <div className="provider-stack">
          {providers.map((provider) => {
            const providerModels = models?.providers[provider] ?? [];
            return (
              <section className="provider-panel" key={provider}>
                <div className="provider-panel-header">
                  <div>
                    <h3>{provider}</h3>
                    <p>{providerModels.length} models</p>
                  </div>
                </div>

                <div className="model-table">
                  <div className="model-table-row model-table-heading">
                    <span>Model</span>
                    <span>Context</span>
                    <span>Output</span>
                    <span>Capabilities</span>
                  </div>
                  {providerModels.map((model) => (
                    <div className="model-table-row" key={`${provider}-${model.id}`}>
                      <span>
                        <strong>{model.name}</strong>
                        <small>{model.id}</small>
                      </span>
                      <span>{formatNumber(model.context_window)}</span>
                      <span>{formatNumber(model.max_output_tokens)}</span>
                      <span className="capability-list">
                        {capabilityLabels(model).map((label) => (
                          <span className="capability-pill" key={label}>{label}</span>
                        ))}
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </section>
  );
}
