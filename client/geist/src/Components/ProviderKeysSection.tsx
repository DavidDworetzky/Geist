import React, { useState } from 'react';
import useProviderKeys, { ProviderKeyStatus } from '../Hooks/useProviderKeys';

interface ProviderKeyRowProps {
  provider: ProviderKeyStatus;
  onSave: (providerId: string, apiKey: string, baseUrl?: string) => Promise<void>;
  onRemove: (providerId: string) => Promise<void>;
}

function statusLabel(provider: ProviderKeyStatus): string {
  if (provider.has_stored_key) {
    return `Stored key ${provider.key_hint ?? ''}`.trim();
  }
  if (provider.env_configured) {
    return `From environment (${provider.api_key_env})`;
  }
  return 'Not configured';
}

const ProviderKeyRow: React.FC<ProviderKeyRowProps> = ({ provider, onSave, onRemove }) => {
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState(provider.base_url ?? '');
  const [busy, setBusy] = useState(false);
  const [rowError, setRowError] = useState<string | null>(null);
  const [rowSuccess, setRowSuccess] = useState<string | null>(null);

  const handleSave = async () => {
    if (!apiKey.trim()) {
      setRowError('Enter an API key before saving.');
      return;
    }
    try {
      setBusy(true);
      setRowError(null);
      setRowSuccess(null);
      await onSave(provider.id, apiKey.trim(), provider.supports_base_url ? baseUrl.trim() : undefined);
      setApiKey('');
      setRowSuccess('Key saved.');
    } catch (err) {
      setRowError(err instanceof Error ? err.message : 'Failed to save key');
    } finally {
      setBusy(false);
    }
  };

  const handleRemove = async () => {
    try {
      setBusy(true);
      setRowError(null);
      setRowSuccess(null);
      await onRemove(provider.id);
      setRowSuccess('Stored key removed.');
    } catch (err) {
      setRowError(err instanceof Error ? err.message : 'Failed to remove key');
    } finally {
      setBusy(false);
    }
  };

  const configured = provider.has_stored_key || provider.env_configured;

  return (
    <div className="provider-key-row" data-testid={`provider-key-row-${provider.id}`}>
      <div className="provider-key-heading">
        <div>
          <span className="settings-label">{provider.name}</span>
          <p className="settings-description">{provider.description}</p>
        </div>
        <span className={`provider-key-status ${configured ? 'configured' : ''}`}>
          {statusLabel(provider)}
        </span>
      </div>

      <div className="provider-key-controls">
        <input
          type="password"
          className="form-control provider-key-input"
          placeholder={provider.has_stored_key ? 'Replace stored key' : 'Enter API key'}
          value={apiKey}
          autoComplete="off"
          aria-label={`${provider.name} API key`}
          onChange={(e) => {
            setApiKey(e.target.value);
            setRowError(null);
            setRowSuccess(null);
          }}
        />
        {provider.supports_base_url && (
          <input
            type="text"
            className="form-control provider-key-input"
            placeholder="Base URL (e.g. http://localhost:8000/v1)"
            value={baseUrl}
            aria-label={`${provider.name} base URL`}
            onChange={(e) => setBaseUrl(e.target.value)}
          />
        )}
        <div className="settings-inline-actions">
          <button className="button button-small" type="button" onClick={handleSave} disabled={busy}>
            {busy ? 'Saving...' : 'Save Key'}
          </button>
          {provider.has_stored_key && (
            <button
              className="button button-secondary button-small"
              type="button"
              onClick={handleRemove}
              disabled={busy}
            >
              Remove
            </button>
          )}
        </div>
      </div>

      {rowError && <p className="provider-key-message error">{rowError}</p>}
      {rowSuccess && <p className="provider-key-message success">{rowSuccess}</p>}
    </div>
  );
};

const ProviderKeysSection: React.FC = () => {
  const { providers, loading, error, saveKey, removeKey, refetch } = useProviderKeys();

  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>Provider API Keys</h3>
        <p>
          Store keys for online providers and Hugging Face downloads. Saved keys take
          precedence over environment variables and are never shown again after saving.
        </p>
      </header>

      {loading && <div className="empty-state compact">Loading providers...</div>}

      {error && !loading && (
        <div className="notice notice-error">
          <span>Could not load providers: {error}</span>
          <button className="button button-small" type="button" onClick={() => void refetch()}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && providers.map((provider) => (
        <ProviderKeyRow
          key={provider.id}
          provider={provider}
          onSave={saveKey}
          onRemove={removeKey}
        />
      ))}
    </section>
  );
};

export default ProviderKeysSection;
