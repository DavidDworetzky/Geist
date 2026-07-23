import React, { useMemo } from 'react';
import SettingsSelect from './SettingsSelect';
import { useAvailableModels } from '../Hooks/useAvailableModels';
import { useLocalModels } from '../Hooks/useLocalModels';

interface AgentConfigSectionProps {
  agentType: string;
  localModel: string;
  onlineProvider: string;
  onlineModel: string;
  onLocalModelChange: (value: string) => void;
  onOnlineProviderChange: (value: string) => void;
  onOnlineModelChange: (value: string) => void;
}

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  xai: 'xAI (Grok)',
  groq: 'Groq',
  huggingface: 'Hugging Face',
  moonshot: 'Moonshot AI',
  zai: 'Z.AI',
  deepseek: 'DeepSeek',
  'self-hosted': 'Self-hosted Server',
  offline: 'Local/Offline',
  custom: 'Custom Provider'
};

const AgentConfigSection: React.FC<AgentConfigSectionProps> = ({
  agentType,
  localModel,
  onlineProvider,
  onlineModel,
  onLocalModelChange,
  onOnlineProviderChange,
  onOnlineModelChange
}) => {
  const {
    getModelById,
    getModelsForProvider,
    loading: modelsLoading,
    providers,
  } = useAvailableModels();
  const { localModels } = useLocalModels();

  const downloadedModelIds = useMemo(
    () => new Set(localModels.filter(m => m.downloaded).map(m => m.id)),
    [localModels]
  );

  const localModelOptions = useMemo(() => {
    const offlineModels = getModelsForProvider('offline');
    if (offlineModels.length > 0) {
      const options = offlineModels.map(m => ({
        value: m.id,
        label: downloadedModelIds.has(m.id) ? `${m.name} • downloaded` : m.name,
        downloaded: downloadedModelIds.has(m.id)
      }));
      // Surface models already present in the weights folder first.
      return options
        .sort((a, b) => Number(b.downloaded) - Number(a.downloaded))
        .map(({ value, label }) => ({ value, label }));
    }

    return [
      { value: 'Meta-Llama-3.1-8B-Instruct', label: 'Meta-Llama-3.1-8B-Instruct' },
      { value: 'Meta-Llama-3.1-8B', label: 'Meta-Llama-3.1-8B' },
      { value: 'Meta-Llama-3-8B-Instruct', label: 'Meta-Llama-3-8B-Instruct' }
    ];
  }, [getModelsForProvider, downloadedModelIds]);

  const onlineProviderOptions = useMemo(() => {
    const onlineProviders = providers.filter(p => p !== 'offline');

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
        <SettingsSelect
          label="Local Model"
          value={localModel}
          options={localModelOptions}
          onChange={onLocalModelChange}
          description={
            localModels.length > 0 && !downloadedModelIds.has(localModel)
              ? 'Weights not downloaded yet - download them from the Models page.'
              : getModelById(localModel)?.performance_note ||
                'Select which local model to use for generation'
          }
        />
      ) : (
        <>
          <SettingsSelect
            label="Online Provider"
            value={onlineProvider}
            options={onlineProviderOptions}
            onChange={handleProviderChange}
            description="Select your preferred online API provider."
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
