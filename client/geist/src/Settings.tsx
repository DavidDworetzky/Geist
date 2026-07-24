import React, { useState, useEffect } from 'react';
import './Settings.css';
import { useUserSettings, UserSettingsUpdate } from './Hooks/useUserSettings';
import AgentConfigSection from './Components/AgentConfigSection';
import GenerationParamsSection from './Components/GenerationParamsSection';
import RAGSettingsSection from './Components/RAGSettingsSection';
import UIPreferencesSection from './Components/UIPreferencesSection';
import SettingsSelect from './Components/SettingsSelect';

type Tab = 'general' | 'models' | 'generation' | 'rag' | 'ui' | 'developer';

const agentTypeOptions = [
  { value: 'local', label: 'Local Model' },
  { value: 'online', label: 'Online Model' }
];

const Settings: React.FC = () => {
  const { settings, loading, error, updateSettings, resetSettings, refetch } = useUserSettings();
  const [activeTab, setActiveTab] = useState<Tab>('general');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [localSettings, setLocalSettings] = useState<any>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState<string>('');

  useEffect(() => {
    if (settings) {
      setLocalSettings(settings);
    }
  }, [settings]);

  const updateLocalSetting = (key: string, value: any) => {
    setLocalSettings((prev: any) => ({
      ...prev,
      [key]: value
    }));
    setHasUnsavedChanges(true);
    setSaveStatus('idle');
  };

  const handleSave = async () => {
    if (!localSettings) return;

    try {
      setSaveStatus('saving');
      setStatusMessage('');

      const updates: UserSettingsUpdate = {
        default_agent_type: localSettings.default_agent_type,
        default_local_model: localSettings.default_local_model,
        default_local_artifact_id: localSettings.default_local_artifact_id,
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
      setHasUnsavedChanges(false);
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
      setHasUnsavedChanges(false);
      setSaveStatus('idle');
      setStatusMessage('');
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Are you sure you want to reset all settings to their default values? This cannot be undone.')) {
      return;
    }

    try {
      setSaveStatus('saving');
      setStatusMessage('');
      await resetSettings();
      setHasUnsavedChanges(false);
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
    { id: 'developer' as Tab, label: 'Developer' }
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
    <div className="settings-page page-surface">
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

      <div className="settings-tab-panel">
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

        {activeTab === 'developer' && (
          <section className="settings-section">
            <header className="settings-section-header">
              <h3>Developer</h3>
              <p>Inspect host branding and runtime integration defaults.</p>
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
          </section>
        )}
      </div>

      <footer className="settings-actions">
        <button className="button button-danger" onClick={handleReset} disabled={saveStatus === 'saving'}>
          Reset to Defaults
        </button>
        <button className="button button-secondary" onClick={handleCancel} disabled={!hasUnsavedChanges || saveStatus === 'saving'}>
          Cancel
        </button>
        <button className="button" onClick={handleSave} disabled={!hasUnsavedChanges || saveStatus === 'saving'}>
          {saveStatus === 'saving' ? 'Saving...' : 'Save Changes'}
        </button>
      </footer>
    </div>
  );
};

export default Settings;
