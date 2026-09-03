import React, { useMemo } from 'react';
import SettingsSelect from './SettingsSelect';
import { useAvailableModels } from '../Hooks/useAvailableModels';

interface AgentConfigSectionProps {
  agentType: string;
  localModel: string;
  onlineProvider: string;
  onlineModel: string;
  onOnlineProviderChange: (value: string) => void;
  onOnlineModelChange: (value: string) => void;
}

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  xai: 'xAI (Grok)',
  groq: 'Groq',
  moonshot: 'Moonshot AI',
  zai: 'Z.AI',
  deepseek: 'DeepSeek',
  openrouter: 'OpenRouter',
  custom: 'Custom Provider'
};

const NON_ONLINE_PROVIDER_IDS = new Set(['offline', 'huggingface', 'self-hosted']);

const AgentConfigSection: React.FC<AgentConfigSectionProps> = ({
  agentType,
  localModel,
  onlineProvider,
  onlineModel,
  onOnlineProviderChange,
  onOnlineModelChange
}) => {
  const {
    getModelById,
    getModelsForProvider,
    loading: modelsLoading,
    providers,
  } = useAvailableModels();

  const onlineProviderOptions = useMemo(() => {
    const onlineProviders = providers.filter(p => !NON_ONLINE_PROVIDER_IDS.has(p));

    return onlineProviders.map(p => ({
      value: p,
      label: PROVIDER_DISPLAY_NAMES[p] || p.charAt(0).toUpperCase() + p.slice(1)
    }));
  }, [providers]);

  const onlineModelOptions = useMemo(() => {
    const providerModels = getModelsForProvider(onlineProvider);
    if (providerModels.length > 0) {
      return providerModels.map(m => ({ value: m.id, label: m.name }));
    }

    return [{ value: 'custom-model', label: 'Custom Model' }];
  }, [getModelsForProvider, onlineProvider]);

  const handleProviderChange = (newProvider: string) => {
    onOnlineProviderChange(newProvider);
    const newProviderModels = getModelsForProvider(newProvider);
    const modelOptions = newProviderModels.map(m => m.id);
    const currentModelAvailable = modelOptions.includes(onlineModel);

    if (!currentModelAvailable && newProviderModels.length > 0) {
      const recommendedModel = newProviderModels.find(m => m.recommended);
      onOnlineModelChange(recommendedModel?.id || newProviderModels[0].id);
    }
  };

  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>Models and Providers</h3>
        <p>Choose model defaults for the selected runtime mode.</p>
      </header>

      {agentType === 'local' ? (
        <div className="settings-field">
          <span className="settings-label">Local model</span>
          <p className="settings-description">
            {getModelById(localModel)?.performance_note ||
              'Download and select a compatible local model from the Models page.'}
          </p>
          <div className="settings-readonly-item">
            <span className="settings-value-pill">
              {getModelById(localModel)?.name || localModel || 'No local model selected'}
            </span>
            <a className="button button-secondary button-small" href="/models">
              Manage local models
            </a>
          </div>
        </div>
      ) : (
        <>
          <SettingsSelect
            label="Online Provider"
            value={onlineProvider}
            options={onlineProviderOptions}
            onChange={handleProviderChange}
            description="Select the online API provider Geist should use."
          />

          <SettingsSelect
            label="Online Model"
            value={onlineModel}
            options={onlineModelOptions}
            onChange={onOnlineModelChange}
            description={
              modelsLoading
                ? 'Loading models...'
                : getModelById(onlineModel)?.performance_note ||
                  'Choose which model from the provider to use'
            }
          />
        </>
      )}
    </section>
  );
};

export default AgentConfigSection;
