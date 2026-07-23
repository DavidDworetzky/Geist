import React, { useState } from 'react';
import useAvailableModels, { ModelInfo } from './Hooks/useAvailableModels';
import useLocalModels, { LocalModelStatus } from './Hooks/useLocalModels';
import useUserSettings from './Hooks/useUserSettings';

function formatNumber(value: number | null): string {
  if (!value) {
    return 'Unknown';
  }
  return new Intl.NumberFormat().format(value);
}

function formatSize(bytes: number): string {
  if (!bytes) {
    return '—';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function downloadStateLabel(model: LocalModelStatus): string {
  if (model.downloaded) {
    return `Downloaded (${formatSize(model.size_bytes)})`;
  }
  if (model.download_status === 'queued') {
    return 'Queued...';
  }
  if (model.download_status === 'running') {
    return 'Downloading...';
  }
  if (model.download_status === 'failed') {
    return 'Download failed';
  }
  return 'Not downloaded';
}

function capabilityLabels(model: ModelInfo): string[] {
  const labels = [];
  if (model.supports_streaming) labels.push('Streaming');
  if (model.supports_function_calling) labels.push('Tools');
  if (model.supports_vision) labels.push('Vision');
  if (model.recommended) labels.push('Recommended');
  return labels;
}

interface LocalWeightsPanelProps {
  localModels: LocalModelStatus[];
  detectedDirectories: { directory: string; weights_path: string; size_bytes: number }[];
  weightsRoot: string | null;
  localError: string | null;
  onDownload: (modelId: string) => Promise<void>;
}

function LocalWeightsPanel({
  localModels,
  detectedDirectories,
  weightsRoot,
  localError,
  onDownload,
}: LocalWeightsPanelProps): JSX.Element {
  const [actionError, setActionError] = useState<string | null>(null);
  const [pendingModel, setPendingModel] = useState<string | null>(null);

  const handleDownload = async (modelId: string) => {
    try {
      setPendingModel(modelId);
      setActionError(null);
      await onDownload(modelId);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to start download');
    } finally {
      setPendingModel(null);
    }
  };

  return (
    <section className="provider-panel local-weights-panel">
      <div className="provider-panel-header">
        <div>
          <h3>Local weights</h3>
          <p>
            {weightsRoot
              ? `Download models from Hugging Face into ${weightsRoot}.`
              : 'Download models from Hugging Face into the packed weights folder.'}
          </p>
        </div>
      </div>

      {localError && (
        <div className="notice notice-warning">Local weight scan failed. {localError}</div>
      )}
      {actionError && <div className="notice notice-error">{actionError}</div>}

      <div className="model-table">
        <div className="model-table-row model-table-heading">
          <span>Model</span>
          <span>Params</span>
          <span>Status</span>
          <span>Action</span>
        </div>
        {localModels.map((model) => {
          const active =
            model.download_status === 'queued' || model.download_status === 'running';
          return (
            <div className="model-table-row" key={`local-${model.id}`}>
              <span>
                <strong>{model.name}</strong>
                <small>{model.id}</small>
              </span>
              <span>{model.parameter_count ?? 'Unknown'}</span>
              <span className="capability-list">
                <span
                  className={`capability-pill ${model.downloaded ? 'downloaded-pill' : ''}`}
                  data-testid={`local-status-${model.id}`}
                >
                  {downloadStateLabel(model)}
                </span>
                {model.gated && <span className="capability-pill">Gated</span>}
              </span>
              <span>
                {!model.downloaded && (
                  <button
                    className="button button-secondary button-small"
                    type="button"
                    disabled={active || pendingModel === model.id}
                    onClick={() => void handleDownload(model.id)}
                  >
                    {active ? 'In progress' : model.download_status === 'failed' ? 'Retry' : 'Download'}
                  </button>
                )}
              </span>
            </div>
          );
        })}
      </div>

      {detectedDirectories.length > 0 && (
        <div className="detected-weights">
          <h4>Also detected in the weights folder</h4>
          {detectedDirectories.map((entry) => (
            <div className="model-table-row" key={`detected-${entry.directory}`}>
              <span>
                <strong>{entry.directory}</strong>
                <small>{entry.weights_path}</small>
              </span>
              <span>{formatSize(entry.size_bytes)}</span>
              <span className="capability-list">
                <span className="capability-pill">Unmanaged</span>
              </span>
              <span />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default function Models(): JSX.Element {
  const { models, loading, error, refetch, providers } = useAvailableModels();
  const {
    localModels,
    detectedDirectories,
    weightsRoot,
    error: localError,
    startDownload,
    refetch: refetchLocal,
  } = useLocalModels();
  const { settings } = useUserSettings();

  const activeModel = settings?.default_agent_type === 'online'
    ? settings.default_online_model
    : settings?.default_local_model;
  const activeProvider = settings?.default_agent_type === 'online'
    ? settings.default_online_provider
    : 'offline';

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
        <button
          className="button button-secondary"
          onClick={() => {
            void refetch();
            void refetchLocal();
          }}
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="notice notice-warning">
          Live model discovery failed, so Geist is showing fallback model data. {error}
        </div>
      )}

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

      <LocalWeightsPanel
        localModels={localModels}
        detectedDirectories={detectedDirectories}
        weightsRoot={weightsRoot}
        localError={localError}
        onDownload={startDownload}
      />

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
