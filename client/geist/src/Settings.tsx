import React, { useCallback, useState, useEffect, useRef } from 'react';
import './Settings.css';
import { useUserSettings, UserSettingsUpdate } from './Hooks/useUserSettings';
import AgentConfigSection from './Components/AgentConfigSection';
import { LLAMA_COMPUTE_VALIDATION_MESSAGE_ID } from './Components/LlamaComputeSection';
import GenerationParamsSection from './Components/GenerationParamsSection';
import RAGSettingsSection from './Components/RAGSettingsSection';
import UIPreferencesSection from './Components/UIPreferencesSection';
import SettingsSelect from './Components/SettingsSelect';
import AboutSection from './Components/AboutSection';
import useOverflowObserver from './Hooks/useOverflowObserver';
import {
  useGeistPluginDiagnostics,
  useHostDevelopmentEnabled,
} from './plugins/runtime';

type Tab = 'general' | 'models' | 'generation' | 'rag' | 'ui' | 'developer' | 'about';

const agentTypeOptions = [
  { value: 'local', label: 'Local Model' },
  { value: 'online', label: 'Online Model' }
];

const SETTINGS_LLAMA_COMPUTE_VALIDATION_MESSAGE_ID = 'settings-llama-compute-validation';

const llamaComputeSelectionSignature = (value: any): string => JSON.stringify([
  value?.llama_backend ?? null,
  [...(value?.llama_gpu_device_ids ?? [])].sort(),
]);

const fallbackLlamaComputeValidity = (value: any): boolean => {
  const deviceIds = value?.llama_gpu_device_ids ?? [];
  return value?.llama_backend !== 'gpu'
    || (deviceIds.length > 0 && new Set(deviceIds).size === deviceIds.length);
};

const Settings: React.FC = () => {
  const { settings, loading, error, updateSettings, resetSettings, refetch } = useUserSettings();
  const [activeTab, setActiveTab] = useState<Tab>('general');
  const [dirtyKeys, setDirtyKeys] = useState<Set<string>>(() => new Set());
  const [localSettings, setLocalSettings] = useState<any>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [llamaComputeValidityBySignature, setLlamaComputeValidityBySignature] = useState(
    () => new Map<string, boolean>(),
  );
  const [pendingLlamaComputeValidity, setPendingLlamaComputeValidity] = useState<{
    signature: string;
    valid: boolean;
  } | null>(null);
  const refreshedOnMount = useRef(false);
  const mounted = useRef(true);
  const [refreshingOnMount, setRefreshingOnMount] = useState(true);
  const [mountRefreshSettled, setMountRefreshSettled] = useState(false);
  const lastMergedSettings = useRef<any>(null);
  const hasUnsavedChanges = dirtyKeys.size > 0;
  const hostDevelopmentEnabled = useHostDevelopmentEnabled();
  const pluginDiagnostics = useGeistPluginDiagnostics();
  const llamaComputeSignature = llamaComputeSelectionSignature(localSettings);
  const cachedLlamaComputeValidity = llamaComputeValidityBySignature.get(llamaComputeSignature);
  const pendingCurrentLlamaComputeValidity = (
    pendingLlamaComputeValidity?.signature === llamaComputeSignature
      ? pendingLlamaComputeValidity.valid
      : undefined
  );
  const llamaComputeValid = pendingCurrentLlamaComputeValidity
    ?? cachedLlamaComputeValidity
    ?? fallbackLlamaComputeValidity(localSettings);
  const handleLlamaComputeValidityChange = useCallback((valid: boolean, settled: boolean) => {
    if (!settled) {
      setPendingLlamaComputeValidity({ signature: llamaComputeSignature, valid });
      return;
    }
    setLlamaComputeValidityBySignature(current => {
      if (current.get(llamaComputeSignature) === valid) {
        return current;
      }
      const next = new Map(current);
      next.set(llamaComputeSignature, valid);
      return next;
    });
    setPendingLlamaComputeValidity(current => (
      current?.signature === llamaComputeSignature ? null : current
    ));
  }, [llamaComputeSignature]);
  const {
    ref: settingsScrollRef,
    hasOverflow: hasSettingsScrollbar,
  } = useOverflowObserver<HTMLDivElement>(Boolean(localSettings));

  useEffect(() => {
    if (!settings) return;
    if (lastMergedSettings.current === settings) {
      if (dirtyKeys.size === 0) {
        setLocalSettings(settings);
      }
      return;
    }
    lastMergedSettings.current = settings;
    setLocalSettings((current: any) => {
      if (!current) return settings;
      const merged = { ...current, ...settings };
      dirtyKeys.forEach(key => {
        merged[key] = current[key];
      });
      return merged;
    });
  }, [settings, dirtyKeys]);

  useEffect(() => {
    if (mountRefreshSettled) {
      setRefreshingOnMount(false);
    }
  }, [mountRefreshSettled]);

  useEffect(() => {
    if (loading || refreshedOnMount.current) {
      return;
    }
    refreshedOnMount.current = true;
    if (!settings) {
      setMountRefreshSettled(true);
      return;
    }
    void refetch().finally(() => {
      if (mounted.current) {
        setMountRefreshSettled(true);
      }
    });
  }, [loading, refetch, settings]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!hostDevelopmentEnabled && activeTab === 'developer') {
      setActiveTab('general');
    }
  }, [activeTab, hostDevelopmentEnabled]);

  useEffect(() => {
    if (activeTab !== 'models' || localSettings?.default_agent_type !== 'local') {
      setPendingLlamaComputeValidity(current => (current === null ? current : null));
    }
  }, [activeTab, localSettings?.default_agent_type]);

  const updateLocalSetting = (key: string, value: any) => {
    setLocalSettings((prev: any) => ({
      ...prev,
      [key]: value
    }));
    setDirtyKeys(previous => {
      const next = new Set(previous);
      next.add(key);
      return next;
    });
    setSaveStatus('idle');
  };

  const handleSave = async () => {
    if (refreshingOnMount || !localSettings) return;
    if (!llamaComputeValid) {
      setSaveStatus('error');
      setStatusMessage('Resolve the GPU device selection before saving.');
      return;
    }

    try {
      setSaveStatus('saving');
      setStatusMessage('');

      const updates: UserSettingsUpdate = {
        default_agent_type: localSettings.default_agent_type,
        default_local_model: localSettings.default_local_model,
        default_local_artifact_id: localSettings.default_local_artifact_id,
        ...(localSettings.llama_backend === null ? {} : {
          llama_backend: localSettings.llama_backend,
          llama_gpu_device_ids: localSettings.llama_gpu_device_ids,
        }),
        default_online_model: localSettings.default_online_model,
        default_online_provider: localSettings.default_online_provider,
        default_file_archives: localSettings.default_file_archives,
        enable_rag_by_default: localSettings.enable_rag_by_default,
        default_max_tokens: localSettings.default_max_tokens,
        default_temperature: localSettings.default_temperature,
        default_top_p: localSettings.default_top_p,
        default_frequency_penalty: localSettings.default_frequency_penalty,
        default_presence_penalty: localSettings.default_presence_penalty,
        backup_providers: localSettings.backup_providers,
        ui_preferences: localSettings.ui_preferences
      };

      await updateSettings(updates);
      setDirtyKeys(new Set());
      setSaveStatus('success');
      setStatusMessage('Settings saved successfully.');

      setTimeout(() => {
        setSaveStatus('idle');
        setStatusMessage('');
      }, 3000);
    } catch (err) {
      setSaveStatus('error');
      setStatusMessage(err instanceof Error ? err.message : 'Failed to save settings');
    }
  };

  const handleCancel = () => {
    if (settings) {
      setLocalSettings(settings);
      setDirtyKeys(new Set());
      setSaveStatus('idle');
      setStatusMessage('');
    }
  };

  const handleReset = async () => {
    if (refreshingOnMount) return;
    if (!window.confirm('Are you sure you want to reset all settings to their default values? This cannot be undone.')) {
      return;
    }

    try {
      setSaveStatus('saving');
      setStatusMessage('');
      await resetSettings();
      setDirtyKeys(new Set());
      setLlamaComputeValidityBySignature(new Map());
      setPendingLlamaComputeValidity(null);
      setSaveStatus('success');
      setStatusMessage('Settings reset to defaults successfully.');

      setTimeout(() => {
        setSaveStatus('idle');
        setStatusMessage('');
      }, 3000);
    } catch (err) {
      setSaveStatus('error');
      setStatusMessage(err instanceof Error ? err.message : 'Failed to reset settings');
    }
  };

  const tabs = [
    { id: 'general' as Tab, label: 'General' },
    { id: 'models' as Tab, label: 'Models and Providers' },
    { id: 'generation' as Tab, label: 'Generation' },
    { id: 'rag' as Tab, label: 'Files and RAG' },
    { id: 'ui' as Tab, label: 'Appearance' },
    ...(hostDevelopmentEnabled
      ? [{ id: 'developer' as Tab, label: 'Developer' }]
      : []),
    { id: 'about' as Tab, label: 'About' }
  ];

  if (loading && !localSettings) {
    return (
      <div className="settings-page page-surface page-surface-centered">
        <div className="empty-state">Loading settings...</div>
      </div>
    );
  }

  if (error && !localSettings) {
    return (
      <div className="settings-page page-surface">
        <div className="notice notice-error">Error loading settings: {error}</div>
        <button className="button" onClick={() => refetch()}>
          Retry
        </button>
      </div>
    );
  }

  if (!localSettings) {
    return null;
  }

  return (
    <div className={`settings-page page-surface settings-page-interactive${activeTab === 'about' ? ' settings-page-about' : ''}${hasSettingsScrollbar ? ' settings-scrollbar-visible' : ''}`}>
      <div className="settings-scroll-region" ref={settingsScrollRef}>
        <header className="settings-header">
          <div>
            <p className="section-eyebrow">Workspace</p>
            <h1>Settings</h1>
          </div>
          {hasUnsavedChanges && <span className="unsaved-pill">Unsaved Changes</span>}
        </header>

        {statusMessage && (
          <div className={`notice ${saveStatus === 'success' ? 'notice-success' : 'notice-error'} settings-status-message`}>
            <span>{statusMessage}</span>
            <button className="icon-action" type="button" onClick={() => setStatusMessage('')} aria-label="Dismiss status">
              X
            </button>
          </div>
        )}

        <div className="settings-tabs" role="tablist" aria-label="Settings sections">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`settings-tab ${activeTab === tab.id ? 'active' : ''}`}
              role="tab"
              aria-selected={activeTab === tab.id}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {refreshingOnMount && (
          <p className="settings-description settings-refresh-status" role="status">
            Refreshing settings… You can keep editing. Save and Reset will be available when refresh finishes.
          </p>
        )}

        <fieldset
          className="settings-tab-panel"
          aria-busy={refreshingOnMount}
          aria-label="Settings controls"
        >
          {activeTab === 'general' && (
            <section className="settings-section">
              <header className="settings-section-header">
                <h3>General</h3>
                <p>Choose the default runtime mode for new conversations.</p>
              </header>
              <SettingsSelect
                label="Default Agent Type"
                value={localSettings.default_agent_type}
                options={agentTypeOptions}
                onChange={(value) => updateLocalSetting('default_agent_type', value)}
                description="Choose whether to use a local or online language model by default."
              />
            </section>
          )}

          {activeTab === 'models' && (
            <AgentConfigSection
              agentType={localSettings.default_agent_type}
              localModel={localSettings.default_local_model}
              onlineProvider={localSettings.default_online_provider}
              onlineModel={localSettings.default_online_model}
              llamaBackend={localSettings.llama_backend}
              llamaGpuDeviceIds={localSettings.llama_gpu_device_ids}
              onLocalModelChange={(value) => {
                updateLocalSetting('default_local_model', value);
                updateLocalSetting('default_local_artifact_id', null);
                if (localSettings.default_agent_type !== 'local') {
                  updateLocalSetting('default_agent_type', 'local');
                }
              }}
              onOnlineProviderChange={(value) => {
                updateLocalSetting('default_online_provider', value);
                if (localSettings.default_agent_type !== 'online') {
                  updateLocalSetting('default_agent_type', 'online');
                }
              }}
              onOnlineModelChange={(value) => {
                updateLocalSetting('default_online_model', value);
                if (localSettings.default_agent_type !== 'online') {
                  updateLocalSetting('default_agent_type', 'online');
                }
              }}
              onLlamaBackendChange={(value) => updateLocalSetting('llama_backend', value)}
              onLlamaGpuDeviceIdsChange={(value) => updateLocalSetting('llama_gpu_device_ids', value)}
              onLlamaComputeValidityChange={handleLlamaComputeValidityChange}
            />
          )}

          {activeTab === 'generation' && (
            <GenerationParamsSection
              temperature={localSettings.default_temperature}
              maxTokens={localSettings.default_max_tokens}
              topP={localSettings.default_top_p}
              frequencyPenalty={localSettings.default_frequency_penalty}
              presencePenalty={localSettings.default_presence_penalty}
              onTemperatureChange={(value) => updateLocalSetting('default_temperature', value)}
              onMaxTokensChange={(value) => updateLocalSetting('default_max_tokens', value)}
              onTopPChange={(value) => updateLocalSetting('default_top_p', value)}
              onFrequencyPenaltyChange={(value) => updateLocalSetting('default_frequency_penalty', value)}
              onPresencePenaltyChange={(value) => updateLocalSetting('default_presence_penalty', value)}
            />
          )}

          {activeTab === 'rag' && (
            <RAGSettingsSection
              enableRagByDefault={localSettings.enable_rag_by_default}
              defaultFileArchives={localSettings.default_file_archives}
              onEnableRagChange={(value) => updateLocalSetting('enable_rag_by_default', value)}
              onFileArchivesChange={(value) => updateLocalSetting('default_file_archives', value)}
            />
          )}

          {activeTab === 'ui' && (
            <UIPreferencesSection
              uiPreferences={localSettings.ui_preferences}
              onUiPreferencesChange={(value) => updateLocalSetting('ui_preferences', value)}
            />
          )}

          {hostDevelopmentEnabled && activeTab === 'developer' && (
            <section className="settings-section">
              <header className="settings-section-header">
                <h3>Developer</h3>
                <p>Inspect host integration and plugins active in this page.</p>
              </header>
              <div className="settings-readonly-grid">
                <div className="settings-readonly-item">
                  <span className="settings-label">Branding Source</span>
                  <span className="settings-description">Host override with neutral fallback</span>
                </div>
                <div className="settings-readonly-item">
                  <span className="settings-label">Theme Contract</span>
                  <span className="settings-description">Semantic CSS variables</span>
                </div>
              </div>
              <div className="settings-subsection-header">
                <div>
                  <h4>Active Plugins</h4>
                  <p className="settings-description">
                    Full-trust host plugins registered with host API 1.
                  </p>
                </div>
                <span className="settings-value-pill">{pluginDiagnostics.length}</span>
              </div>
              {pluginDiagnostics.length === 0 ? (
                <p className="settings-description">No host plugins are active.</p>
              ) : (
                <div className="plugin-diagnostic-list">
                  {pluginDiagnostics.map((plugin) => {
                    const mountedCount = plugin.contributions.filter(
                      (contribution) => contribution.mounted
                    ).length;
                    return (
                      <article className="plugin-diagnostic-item" key={plugin.id}>
                        <div>
                          <span className="settings-label">{plugin.name}</span>
                          <span className="settings-description plugin-diagnostic-id">
                            {plugin.provider} - {plugin.id} v{plugin.version} - API {plugin.apiVersion}
                          </span>
                        </div>
                        <div className="plugin-diagnostic-summary">
                          <span className={`plugin-status plugin-status-${plugin.status}`}>
                            {plugin.status}
                          </span>
                          <span className="settings-description">
                            {mountedCount}/{plugin.contributions.length} mounted
                          </span>
                        </div>
                        <p className={`settings-description${plugin.lastError ? ' plugin-diagnostic-error' : ''}`}>
                          Last error: {plugin.lastError ?? 'None'}
                        </p>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>
          )}

          {activeTab === 'about' && <AboutSection />}
        </fieldset>
      </div>

      {activeTab !== 'about' && (
        <footer className="settings-actions" aria-label="Settings actions">
          {!llamaComputeValid && activeTab !== 'models' && (
            <span
              id={SETTINGS_LLAMA_COMPUTE_VALIDATION_MESSAGE_ID}
              className="settings-description settings-action-validation"
              role="alert"
            >
              Resolve the GPU device selection before saving.
            </span>
          )}
          <button className="button button-danger" onClick={handleReset} disabled={refreshingOnMount || saveStatus === 'saving'}>
            Reset to Defaults
          </button>
          <button className="button button-secondary" onClick={handleCancel} disabled={!hasUnsavedChanges || saveStatus === 'saving'}>
            Cancel
          </button>
          <button
            className="button"
            onClick={handleSave}
            disabled={refreshingOnMount || !hasUnsavedChanges || saveStatus === 'saving' || !llamaComputeValid}
            aria-describedby={!llamaComputeValid
              ? (activeTab === 'models'
                ? LLAMA_COMPUTE_VALIDATION_MESSAGE_ID
                : SETTINGS_LLAMA_COMPUTE_VALIDATION_MESSAGE_ID)
              : undefined}
          >
            {saveStatus === 'saving' ? 'Saving...' : 'Save Changes'}
          </button>
        </footer>
      )}
    </div>
  );
};

export default Settings;
