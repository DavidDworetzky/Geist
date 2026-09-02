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

type ModelsTab = 'local' | 'providers';

function formatBytes(value?: number | null): string {
  if (!value) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const unit = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / (1024 ** unit)).toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
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

  const onlineProviders = providers.filter(provider => provider !== 'offline');

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

  const activeModel = settings?.default_agent_type === 'online'
    ? settings.default_online_model
    : settings?.default_local_model;
  const activeProvider = settings?.default_agent_type === 'online'
    ? settings.default_online_provider
    : 'offline';
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
          <p className="section-eyebrow">Providers</p>
          <h2>Models</h2>
          <p>
            Manage models installed on this computer or browse available providers.
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
          Local Models
        </button>
        <button
          id="model-providers-tab"
          type="button"
          className={`settings-tab ${activeTab === 'providers' ? 'active' : ''}`}
          role="tab"
          aria-selected={activeTab === 'providers'}
          aria-controls="model-providers-panel"
          onClick={() => setActiveTab('providers')}
        >
          Model Providers
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
              {' '}Downloads are limited to pinned, verified artifacts; unsupported artifacts remain
              visible for reference.
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
          {localArtifacts.map(artifact => {
            const active = settings?.default_local_artifact_id === artifact.id;
            const busy = ['queued', 'downloading', 'cancelling'].includes(artifact.status);
            const supported = artifact.supported !== false;
            const total = artifact.total_bytes ?? 0;
            return (
              <div
                className={`model-table-row ${supported ? '' : 'model-table-row-unavailable'}`}
                key={artifact.id}
              >
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
                        title={!supported ? 'This model cannot run on the current platform.' : undefined}
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
                        disabled={!supported || localAction === artifact.id || artifact.source === 'imported'}
                        title={!supported ? 'This artifact requires a backend unavailable on this platform.' : undefined}
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
        </div>
      )}

      {activeTab === 'providers' && (
        <div
          id="model-providers-panel"
          className="models-tab-panel"
          role="tabpanel"
          aria-labelledby="model-providers-tab"
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
              <p>Gathering local and online provider options.</p>
            </section>
          ) : (
            <>
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
                  <strong>{onlineProviders.length}</strong>
                </article>
              </div>

              <div className="model-inventory-scroll" role="region" aria-label="Model providers">
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
