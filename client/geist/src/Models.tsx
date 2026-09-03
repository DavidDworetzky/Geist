import React, { ChangeEvent, useEffect, useState } from 'react';
import useAvailableModels, { ModelInfo } from './Hooks/useAvailableModels';
import useLocalArtifacts, {
  installProgress,
  isArtifactInstalling,
  LocalArtifact,
  responseError,
} from './Hooks/useLocalArtifacts';
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

function artifactStatusLabel(artifact: LocalArtifact, active: boolean): string {
  if (artifact.status === 'installed') return active ? 'Installed · Active' : 'Installed';
  if (isArtifactInstalling(artifact)) return installProgress(artifact).label;
  if (artifact.status === 'failed') return 'Install failed';
  return 'Not installed';
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
    downloadArtifact,
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
    action: 'cancel' | 'remove',
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
      if (!response.ok) throw await responseError(response, `Could not ${action} model.`);
      await refreshLocalArtifacts();
    } catch (requestError) {
      setLocalActionError(requestError instanceof Error ? requestError.message : `Model ${action} failed`);
    } finally {
      setLocalAction(null);
    }
  };

  const installArtifact = async (artifact: LocalArtifact) => {
    setLocalAction(artifact.id);
    setLocalActionError(null);
    try {
      await downloadArtifact(artifact);
    } catch (requestError) {
      setLocalActionError(
        requestError instanceof Error ? requestError.message : 'Could not install model.',
      );
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
      if (!response.ok) throw await responseError(response, 'Could not import model.');
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

  return (
    <section className="models-page">
      <div className="page-header">
        <div>
          <p className="section-eyebrow">Inference</p>
          <h2>Models</h2>
          <p>Run models locally or use an online provider.</p>
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
            <h3 id="local-model-files-heading">Local models</h3>
            <p>Only models compatible with this computer are shown.</p>
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
            <span>Model</span>
            <span>Variant</span>
            <span>Status</span>
            <span>Actions</span>
          </div>
          {compatibleLocalArtifacts.map(artifact => {
            const active = settings?.default_agent_type === 'local' && (
              settings.default_local_artifact_id === artifact.id
              || (!settings.default_local_artifact_id
                && settings.default_local_model === artifact.model_id)
            );
            const busy = isArtifactInstalling(artifact);
            const total = artifact.total_bytes ?? 0;
            return (
              <div className="model-table-row" key={artifact.id}>
                <span>
                  <strong>{artifact.display_name}</strong>
                  <small>{artifact.model_id}</small>
                </span>
                <span>
                  {artifact.quantization
                    || (artifact.backend === 'mlx_llama' ? 'MLX' : artifact.format.toUpperCase())}
                </span>
                <span>
                  {artifactStatusLabel(artifact, active)}
                  {artifact.status === 'downloading' && (
                    <small>
                      {artifact.progress_unit === 'files'
                        ? `${artifact.progress_completed ?? 0} / ${artifact.progress_total ?? '?'} files`
                        : `${formatBytes(artifact.bytes_downloaded)} / ${formatBytes(total)}`}
                    </small>
                  )}
                  {artifact.requires_auth && artifact.status !== 'installed' && (
                    <small>Requires Hugging Face access and a token.</small>
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
                  ) : artifact.status === 'cancelling' ? null : busy ? (
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
                        onClick={() => void installArtifact(artifact)}
                      >
                        {artifact.status === 'failed' ? 'Retry' : 'Install'}
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
              <h3>Loading providers…</h3>
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
                                <small>
                                  {providerModels.length} {providerModels.length === 1 ? 'model' : 'models'}
                                </small>
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
