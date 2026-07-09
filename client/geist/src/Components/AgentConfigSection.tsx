import React, { useMemo } from 'react';
import SettingsSelect from './SettingsSelect';
import { useAvailableModels } from '../Hooks/useAvailableModels';

interface AgentConfigSectionProps {
  agentType: string;
  localModel: string;
  onlineProvider: string;
  onlineModel: string;
  onAgentTypeChange: (value: string) => void;
  onLocalModelChange: (value: string) => void;
  onOnlineProviderChange: (value: string) => void;
  onOnlineModelChange: (value: string) => void;
}

// Provider display names mapping
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  xai: 'xAI (Grok)',
  groq: 'Groq',
  huggingface: 'Hugging Face',
  offline: 'Local/Offline',
  custom: 'Custom Provider',
};

const AgentConfigSection: React.FC<AgentConfigSectionProps> = ({
  agentType,
  localModel,
  onlineProvider,
  onlineModel,
  onAgentTypeChange,
  onLocalModelChange,
  onOnlineProviderChange,
  onOnlineModelChange
}) => {
  const { getModelsForProvider, loading: modelsLoading, providers } = useAvailableModels();

  const agentTypeOptions = [
    { value: 'local', label: 'Local Model' },
    { value: 'online', label: 'Online Model' }
  ];

  // Get local models from the offline provider
  const localModelOptions = useMemo(() => {
    const offlineModels = getModelsForProvider('offline');
    if (offlineModels.length > 0) {
      return offlineModels.map(m => ({ value: m.id, label: m.name }));
    }
    // Fallback to static options
    return [
      { value: 'Meta-Llama-3.1-8B-Instruct', label: 'Meta-Llama-3.1-8B-Instruct' },
      { value: 'Meta-Llama-3.1-8B', label: 'Meta-Llama-3.1-8B' },
      { value: 'Meta-Llama-3-8B-Instruct', label: 'Meta-Llama-3-8B-Instruct' }
    ];
  }, [getModelsForProvider]);

  // Generate online provider options from dynamic providers
  const onlineProviderOptions = useMemo(() => {
    // Filter out 'offline' as it's for local models
    const onlineProviders = providers.filter(p => p !== 'offline');

    return onlineProviders.map(p => ({
      value: p,
      label: PROVIDER_DISPLAY_NAMES[p] || p.charAt(0).toUpperCase() + p.slice(1)
    }));
  }, [providers]);

  // Get model options for current provider
  const onlineModelOptions = useMemo(() => {
    const providerModels = getModelsForProvider(onlineProvider);
    if (providerModels.length > 0) {
      return providerModels.map(m => ({ value: m.id, label: m.name }));
    }
    // Fallback for custom or unknown providers
    return [{ value: 'custom-model', label: 'Custom Model' }];
  }, [getModelsForProvider, onlineProvider]);

  // Handle provider change - reset model if current model isn't available for new provider
  const handleProviderChange = (newProvider: string) => {
    onOnlineProviderChange(newProvider);
    const newProviderModels = getModelsForProvider(newProvider);
    const modelOptions = newProviderModels.map(m => m.id);
    const currentModelAvailable = modelOptions.includes(onlineModel);

    if (!currentModelAvailable && newProviderModels.length > 0) {
      // Prefer recommended models, otherwise use first available
      const recommendedModel = newProviderModels.find(m => m.recommended);
      onOnlineModelChange(recommendedModel?.id || newProviderModels[0].id);
    }
  };

  return (
    <div style={{
      backgroundColor: 'white',
      padding: '25px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      marginBottom: '20px'
    }}>
      <h3 style={{
        margin: '0 0 20px 0',
        color: '#333',
        fontSize: '18px',
        borderBottom: '2px solid #007bff',
        paddingBottom: '10px'
      }}>
        Agent Configuration
      </h3>

      <SettingsSelect
        label="Default Agent Type"
        value={agentType}
        options={agentTypeOptions}
        onChange={onAgentTypeChange}
        description="Choose whether to use a local or online language model by default"
      />

      {agentType === 'local' ? (
        <SettingsSelect
          label="Local Model"
          value={localModel}
          options={localModelOptions}
          onChange={onLocalModelChange}
          description="Select which local model to use for generation"
        />
      ) : (
        <>
          <SettingsSelect
            label="Online Provider"
            value={onlineProvider}
            options={onlineProviderOptions}
            onChange={handleProviderChange}
            description="Select your preferred online API provider"
          />

          <SettingsSelect
            label="Online Model"
            value={onlineModel}
            options={onlineModelOptions}
            onChange={onOnlineModelChange}
            description={modelsLoading ? "Loading models..." : "Choose which model from the provider to use"}
          />
        </>
      )}
    </div>
  );
};

export default AgentConfigSection;
