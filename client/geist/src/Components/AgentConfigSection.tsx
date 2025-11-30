import React from 'react';
import SettingsSelect from './SettingsSelect';

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
  const agentTypeOptions = [
    { value: 'local', label: 'Local Model' },
    { value: 'online', label: 'Online Model' }
  ];

  const localModelOptions = [
    { value: 'Meta-Llama-3.1-8B-Instruct', label: 'Meta-Llama-3.1-8B-Instruct' },
    { value: 'Meta-Llama-3.1-8B', label: 'Meta-Llama-3.1-8B' },
    { value: 'Meta-Llama-3-8B-Instruct', label: 'Meta-Llama-3-8B-Instruct' }
  ];

  const onlineProviderOptions = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'anthropic', label: 'Anthropic' },
    { value: 'custom', label: 'Custom Provider' }
  ];

  // Models organized by provider
  const modelsByProvider: Record<string, { value: string; label: string }[]> = {
    openai: [
      { value: 'gpt-4', label: 'GPT-4' },
      { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
      { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
    ],
    anthropic: [
      { value: 'claude-3-opus', label: 'Claude 3 Opus' },
      { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' }
    ],
    custom: [
      { value: 'custom-model', label: 'Custom Model' }
    ]
  };

  // Get filtered model options based on selected provider
  const onlineModelOptions = modelsByProvider[onlineProvider] || [];

  // Handle provider change - reset model if current model isn't available for new provider
  const handleProviderChange = (newProvider: string) => {
    onOnlineProviderChange(newProvider);
    const newProviderModels = modelsByProvider[newProvider] || [];
    const currentModelAvailable = newProviderModels.some(m => m.value === onlineModel);
    if (!currentModelAvailable && newProviderModels.length > 0) {
      onOnlineModelChange(newProviderModels[0].value);
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
            description="Choose which model from the provider to use"
          />
        </>
      )}
    </div>
  );
};

export default AgentConfigSection;

