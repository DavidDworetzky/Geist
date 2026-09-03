import React, { ChangeEvent, useEffect, useState } from 'react';
import useAvailableModels, { ModelInfo } from './Hooks/useAvailableModels';
import useLocalArtifacts, { LocalArtifact } from './Hooks/useLocalArtifacts';
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

type ModelsTab = 'local' | 'online';

const NON_ONLINE_PROVIDER_IDS = new Set(['offline', 'huggingface', 'self-hosted']);

function formatBytes(value?: number | null): string {
  if (!value) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const unit = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / (1024 ** unit)).toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function artifactStatusLabel(status: string, active: boolean): string {
  if (status === 'installed') return active ? 'Downloaded · Active' : 'Downloaded';
  if (status === 'not_installed') return 'Not downloaded';
  if (status === 'queued') return 'Download queued';
  if (status === 'downloading') return 'Downloading';
  if (status === 'cancelling') return 'Cancelling download';
  if (status === 'cancelled') return 'Download cancelled';
  if (status === 'failed') return 'Download failed';
  return status;
}

export default function Models(): JSX.Element {
  const { models, loading, error, refetch, providers } = useAvailableModels();
  const {
    settings,
    loading: settingsLoading,
    updateSettings,
  } = useUserSettings();
  const {
    artifacts: localArtifacts,
    error: localArtifactsError,
    refreshLocalArtifacts,
    activateArtifact,
  } = useLocalArtifacts({ pollWhileBusy: true });
  const [localActionError, setLocalActionError] = useState<string | null>(null);
  const [localAction, setLocalAction] = useState<string | null>(null);
  const [providerActionError, setProviderActionError] = useState<string | null>(null);
  const [providerAction, setProviderAction] = useState<string | null>(null);
  const [expandedProviders, setExpandedProviders] = useState<Set<string> | null>(null);
  const [activeTab, setActiveTab] = useState<ModelsTab>('local');

  const onlineProviders = providers.filter(provider => !NON_ONLINE_PROVIDER_IDS.has(provider));
  const compatibleLocalArtifacts = localArtifacts.filter(artifact => artifact.supported !== false);

  useEffect(() => {
    if (onlineProviders.length === 0 || settingsLoading) return;
    setExpandedProviders(current => {
      if (current !== null) return current;
      const initialProvider = onlineProviders.includes(settings?.default_online_provider ?? '')
        ? settings?.default_online_provider
        : onlineProviders[0];
      return new Set(initialProvider ? [initialProvider] : []);
    });
  }, [onlineProviders, settings?.default_online_provider, settingsLoading]);

  const runArtifactAction = async (
    artifactId: string,
    action: 'download' | 'cancel' | 'remove',
  ) => {
    setLocalAction(artifactId);
    setLocalActionError(null);
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
      setLocalActionError(requestError instanceof Error ? requestError.message : `Model ${action} failed`);
    } finally {
      setLocalAction(null);
    }
  };

  const importGguf = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setLocalAction('import');
    setLocalActionError(null);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/v1/models/local/import', { method: 'POST', body });
      if (!response.ok) throw new Error(`GGUF import failed: ${response.statusText}`);
      await refreshLocalArtifacts();
    } catch (requestError) {
      setLocalActionError(requestError instanceof Error ? requestError.message : 'GGUF import failed');
    } finally {
      event.target.value = '';
      setLocalAction(null);
    }
  };

  const selectArtifact = async (artifact: LocalArtifact) => {
    setLocalAction(artifact.id);
    setLocalActionError(null);
    try {
      await activateArtifact(artifact);
    } catch (requestError) {
      setLocalActionError(requestError instanceof Error ? requestError.message : 'Model selection failed');
    } finally {
      setLocalAction(null);
    }
  };

  const selectProviderModel = async (provider: string, model: ModelInfo) => {
    const actionId = `${provider}-${model.id}`;
    setProviderAction(actionId);
    setProviderActionError(null);
    try {
      await updateSettings({
        default_agent_type: 'online',
        default_online_provider: provider,
        default_online_model: model.id,
      });
    } catch (requestError) {
      setProviderActionError(
        requestError instanceof Error ? requestError.message : 'Provider model selection failed',
      );
    } finally {
      setProviderAction(null);
    }
  };

  const toggleProvider = (provider: string) => {
    setExpandedProviders(current => {
      const next = new Set(current ?? []);
      if (next.has(provider)) {
        next.delete(provider);
      } else {
        next.add(provider);
      }
      return next;
    });
  };

  const inferenceMode = settings?.default_agent_type === 'online' ? 'Online' : 'Local';
  const onlineDefault = settings
    ? `${settings.default_online_provider} · ${settings.default_online_model}`
    : 'Not selected';
  const ggufSupported = localArtifacts.some(
    artifact => artifact.format === 'gguf' && artifact.supported !== false,
  );
  const mlxSupported = localArtifacts.some(
    artifact => artifact.backend === 'mlx_llama' && artifact.supported !== false,
  );

  return (
    <section className="models-page">
      <div className="page-header">
        <div>
          <p className="section-eyebrow">Inference</p>
          <h2>Models</h2>
          <p>
            Download a compatible model to run locally, or choose one from an online provider.
          </p>
        </div>
        <button
          className="button button-secondary"
          onClick={() => {
            void refreshLocalArtifacts();
            void refetch();
          }}
        >
          Refresh
        </button>
      </div>

      <div className="settings-tabs models-tabs" role="tablist" aria-label="Model sections">
        <button
          id="local-models-tab"
          type="button"
          className={`settings-tab ${activeTab === 'local' ? 'active' : ''}`}
          role="tab"
          aria-selected={activeTab === 'local'}
          aria-controls="local-models-panel"
          onClick={() => setActiveTab('local')}
        >
          Local
        </button>
        <button
          id="online-models-tab"
          type="button"
          className={`settings-tab ${activeTab === 'online' ? 'active' : ''}`}
          role="tab"
          aria-selected={activeTab === 'online'}
          aria-controls="online-models-panel"
          onClick={() => setActiveTab('online')}
        >
          Online
        </button>
      </div>

      {activeTab === 'local' && (
        <div
          id="local-models-panel"
          className="models-tab-panel model-inventory-scroll"
          role="tabpanel"
          aria-labelledby="local-models-tab"
        >
          <section
            className="provider-panel local-model-panel"
            aria-labelledby="local-model-files-heading"
          >
        <div className="provider-panel-header">
          <div>
            <h3 id="local-model-files-heading">Local model files</h3>
            <p>
              {mlxSupported
                ? 'Download and select the managed MLX snapshot used on Apple silicon.'
                : 'Download a curated GGUF or import one already on this computer.'}
              {' '}Only models compatible with this computer are shown. Source labels identify
              where weights come from; inference still runs locally.
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
        {(localArtifactsError || localActionError) && (
          <div className="notice notice-error">{localActionError ?? localArtifactsError}</div>
        )}
        <div className="model-table">
          <div className="model-table-row model-table-heading">
            <span>Artifact</span>
            <span>Format</span>
            <span>Status</span>
            <span>Actions</span>
          </div>
          {compatibleLocalArtifacts.map(artifact => {
            const active = settings?.default_agent_type === 'local' && (
              settings.default_local_artifact_id === artifact.id
              || (!settings.default_local_artifact_id
                && settings.default_local_model === artifact.model_id)
            );
            const busy = ['queued', 'downloading', 'cancelling'].includes(artifact.status);
            const total = artifact.total_bytes ?? 0;
            return (
              <div className="model-table-row" key={artifact.id}>
                <span>
                  <strong>{artifact.display_name}</strong>
                  <small>{artifact.model_id}</small>
                  <small>Source · {artifact.repo_id || 'Imported file'}</small>
                </span>
                <span>{artifact.quantization || artifact.format.toUpperCase()}</span>
                <span>
                  {artifactStatusLabel(artifact.status, active)}
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
                        disabled={active || localAction === artifact.id}
                        onClick={() => void selectArtifact(artifact)}
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
                        disabled={localAction === artifact.id || artifact.source === 'imported'}
                        onClick={() => void runArtifactAction(artifact.id, 'download')}
                      >
                        Download to use
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
          {compatibleLocalArtifacts.length === 0 && (
            <div className="model-table-empty">No compatible local models are available.</div>
          )}
        </div>
          </section>
        </div>
      )}

      {activeTab === 'online' && (
        <div
          id="online-models-panel"
          className="models-tab-panel"
          role="tabpanel"
          aria-labelledby="online-models-tab"
        >
          {error && (
            <div className="notice notice-warning">
              Live model discovery failed, so Geist is showing fallback model data. {error}
            </div>
          )}

          {providerActionError && (
            <div className="notice notice-error">{providerActionError}</div>
          )}

          {loading && !models ? (
            <section className="page-surface page-surface-centered models-providers-loading">
              <h3>Loading model providers</h3>
              <p>Gathering online provider options.</p>
            </section>
          ) : (
            <>
              <div className="model-summary-grid">
                <article className="metric-card">
                  <span className="metric-label">Inference mode</span>
                  <strong>{inferenceMode}</strong>
                </article>
                <article className="metric-card">
                  <span className="metric-label">Online default</span>
                  <strong>{onlineDefault}</strong>
                </article>
                <article className="metric-card">
                  <span className="metric-label">Available providers</span>
                  <strong>{onlineProviders.length}</strong>
                </article>
              </div>

              <div className="model-inventory-scroll" role="region" aria-label="Online models">
                <div className="provider-stack">
                  {onlineProviders.map((provider) => {
                    const providerModels = models?.providers[provider] ?? [];
                    const expanded = expandedProviders?.has(provider) ?? false;
                    const panelId = `provider-models-${provider.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
                    const toggleId = `${panelId}-toggle`;
                    return (
                      <section
                        className="provider-panel"
                        key={provider}
                        aria-labelledby={toggleId}
                      >
                        <div className="provider-panel-header provider-panel-disclosure-header">
                          <h3 className="provider-panel-disclosure-title">
                            <button
                              id={toggleId}
                              type="button"
                              className="provider-panel-toggle"
                              aria-expanded={expanded}
                              aria-controls={panelId}
                              onClick={() => toggleProvider(provider)}
                            >
                              <span>
                                <span>{provider}</span>
                                <small>{providerModels.length} models</small>
                              </span>
                              <span className="provider-panel-chevron" aria-hidden="true">
                                {expanded ? '−' : '+'}
                              </span>
                            </button>
                          </h3>
                        </div>

                        {expanded && <div id={panelId} className="model-table provider-model-table">
                          <div className="model-table-row model-table-heading">
                            <span>Model</span>
                            <span>Context</span>
                            <span>Output</span>
                            <span>Capabilities</span>
                            <span>Actions</span>
                          </div>
                          {providerModels.map((model) => {
                            const actionId = `${provider}-${model.id}`;
                            const active = settings?.default_agent_type === 'online'
                              && settings.default_online_provider === provider
                              && settings.default_online_model === model.id;
                            return (
                              <div
                                className={`model-table-row ${active ? 'model-table-row-active' : ''}`}
                                key={actionId}
                              >
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
                                <span className="settings-inline-actions">
                                  <button
                                    type="button"
                                    className="button button-secondary button-small"
                                    disabled={active || providerAction !== null}
                                    onClick={() => void selectProviderModel(provider, model)}
                                  >
                                    {active
                                      ? 'Active'
                                      : providerAction === actionId ? 'Selecting…' : 'Use'}
                                  </button>
                                </span>
                              </div>
                            );
                          })}
                        </div>}
                      </section>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
