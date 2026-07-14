import React from 'react';
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

export default function Models(): JSX.Element {
  const { models, loading, error, refetch, providers } = useAvailableModels();
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
        <button className="button button-secondary" onClick={() => void refetch()}>
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
