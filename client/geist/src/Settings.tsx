import React, { useState, useEffect } from 'react';
import './Settings.css';
import { useUserSettings, UserSettingsUpdate } from './Hooks/useUserSettings';
import AgentConfigSection from './Components/AgentConfigSection';
import GenerationParamsSection from './Components/GenerationParamsSection';
import RAGSettingsSection from './Components/RAGSettingsSection';
import BackupProvidersSection from './Components/BackupProvidersSection';
import UIPreferencesSection from './Components/UIPreferencesSection';

type Tab = 'agent' | 'generation' | 'rag' | 'providers' | 'ui';

const Settings: React.FC = () => {
  const { settings, loading, error, updateSettings, resetSettings, refetch } = useUserSettings();
  const [activeTab, setActiveTab] = useState<Tab>('agent');
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
      setStatusMessage('Settings saved successfully!');
      
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
      setStatusMessage('Settings reset to defaults successfully!');
      
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
    { id: 'agent' as Tab, label: 'Agent Config', icon: '🤖' },
    { id: 'generation' as Tab, label: 'Generation', icon: '⚙️' },
    { id: 'rag' as Tab, label: 'RAG & Files', icon: '📁' },
    { id: 'providers' as Tab, label: 'Backup Providers', icon: '🔄' },
    { id: 'ui' as Tab, label: 'UI Preferences', icon: '🎨' }
  ];

  if (loading && !localSettings) {
    return (
      <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto', textAlign: 'center' }}>
        <h1 style={{ marginBottom: '20px', color: '#333' }}>Settings</h1>
        <div style={{ padding: '60px', color: '#6c757d', fontSize: '18px' }}>
          Loading settings...
        </div>
      </div>
    );
  }

  if (error && !localSettings) {
    return (
      <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
        <h1 style={{ marginBottom: '20px', color: '#333' }}>Settings</h1>
        <div style={{
          padding: '20px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderRadius: '5px'
        }}>
          Error loading settings: {error}
        </div>
        <button
          onClick={() => refetch()}
          style={{
            marginTop: '15px',
            padding: '10px 20px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (!localSettings) {
    return null;
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '30px' 
      }}>
        <h1 style={{ margin: '0', color: '#333' }}>Settings</h1>
        {hasUnsavedChanges && (
          <span style={{
            padding: '6px 12px',
            backgroundColor: '#ffc107',
            color: '#000',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 'bold'
          }}>
            Unsaved Changes
          </span>
        )}
      </div>

      {statusMessage && (
        <div style={{
          padding: '12px',
          marginBottom: '20px',
          backgroundColor: saveStatus === 'success' ? '#d4edda' : '#f8d7da',
          color: saveStatus === 'success' ? '#155724' : '#721c24',
          border: `1px solid ${saveStatus === 'success' ? '#c3e6cb' : '#f5c6cb'}`,
          borderRadius: '5px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span>{statusMessage}</span>
          <button
            onClick={() => setStatusMessage('')}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '18px',
              cursor: 'pointer',
              padding: '0 5px',
              color: saveStatus === 'success' ? '#155724' : '#721c24'
            }}
          >
            ×
          </button>
        </div>
      )}

      <div className="settings-tabs" style={{
        display: 'flex',
        gap: '5px',
        marginBottom: '25px',
        borderBottom: '2px solid #ddd',
        overflowX: 'auto'
      }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`settings-tab ${activeTab === tab.id ? 'active' : ''}`}
            style={{
              padding: '12px 20px',
              backgroundColor: activeTab === tab.id ? '#007bff' : 'transparent',
              color: activeTab === tab.id ? 'white' : '#333',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #007bff' : '2px solid transparent',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: activeTab === tab.id ? 'bold' : 'normal',
              transition: 'all 0.2s',
              borderRadius: '5px 5px 0 0',
              whiteSpace: 'nowrap'
            }}
          >
            <span style={{ marginRight: '8px' }}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      <div style={{ minHeight: '400px' }}>
        {activeTab === 'agent' && (
          <AgentConfigSection
            agentType={localSettings.default_agent_type}
            localModel={localSettings.default_local_model}
            onlineProvider={localSettings.default_online_provider}
            onlineModel={localSettings.default_online_model}
            onAgentTypeChange={(value) => updateLocalSetting('default_agent_type', value)}
            onLocalModelChange={(value) => updateLocalSetting('default_local_model', value)}
            onOnlineProviderChange={(value) => updateLocalSetting('default_online_provider', value)}
            onOnlineModelChange={(value) => updateLocalSetting('default_online_model', value)}
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

        {activeTab === 'providers' && (
          <BackupProvidersSection
            providers={localSettings.backup_providers}
            onProvidersChange={(value) => updateLocalSetting('backup_providers', value)}
          />
        )}

        {activeTab === 'ui' && (
          <UIPreferencesSection
            uiPreferences={localSettings.ui_preferences}
            onUiPreferencesChange={(value) => updateLocalSetting('ui_preferences', value)}
          />
        )}
      </div>

      <div style={{
        display: 'flex',
        gap: '10px',
        justifyContent: 'flex-end',
        marginTop: '30px',
        paddingTop: '20px',
        borderTop: '2px solid #ddd'
      }}>
        <button
          onClick={handleReset}
          disabled={saveStatus === 'saving'}
          style={{
            padding: '10px 20px',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: saveStatus === 'saving' ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
            opacity: saveStatus === 'saving' ? 0.6 : 1
          }}
        >
          Reset to Defaults
        </button>
        
        <button
          onClick={handleCancel}
          disabled={!hasUnsavedChanges || saveStatus === 'saving'}
          style={{
            padding: '10px 20px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: !hasUnsavedChanges || saveStatus === 'saving' ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
            opacity: !hasUnsavedChanges || saveStatus === 'saving' ? 0.6 : 1
          }}
        >
          Cancel
        </button>
        
        <button
          onClick={handleSave}
          disabled={!hasUnsavedChanges || saveStatus === 'saving'}
          style={{
            padding: '10px 20px',
            backgroundColor: '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: !hasUnsavedChanges || saveStatus === 'saving' ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
            opacity: !hasUnsavedChanges || saveStatus === 'saving' ? 0.6 : 1
          }}
        >
          {saveStatus === 'saving' ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  );
};

export default Settings;

