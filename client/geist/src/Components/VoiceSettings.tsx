import React, { useState } from 'react';
import useVoiceModels, { TTSModelInfo, TTSProviderInfo, UseVoiceModelsReturn } from '../Hooks/useVoiceModels';

export interface VoiceSelection {
  sttProvider: string;
  ttsProvider: string;
  ttsModel?: string;
  ttsVoice?: string;
  ttsLanguage?: string;
}

// Fallback while the installed-model catalog is loading; recording waits for it.
export const DEFAULT_VOICE_SELECTION: VoiceSelection = {
  sttProvider: 'mms',
  ttsProvider: 'sesame'
};

const STT_PROVIDERS = [
  { id: 'mms', display_name: 'MMS (local)' },
  { id: 'whisper', display_name: 'Whisper (OpenAI API)' }
];

export const modelIsReady = (model: TTSModelInfo): boolean => (
  !model.artifact
  || (
    model.artifact.supported !== false
    && model.artifact.status === 'installed'
    && model.artifact.runtime_ready === true
  )
);

const providerIsReady = (provider: TTSProviderInfo): boolean => (
  provider.models.some(modelIsReady)
);

interface VoiceSettingsProps {
  selection: VoiceSelection;
  onChange: (selection: VoiceSelection) => void;
  disabled?: boolean;
  catalog?: UseVoiceModelsReturn;
}

const VoiceSettings: React.FC<VoiceSettingsProps> = ({
  selection,
  onChange,
  disabled = false,
  catalog
}) => {
  const [expanded, setExpanded] = useState(false);
  const localCatalog = useVoiceModels(expanded && !catalog);
  const { data, loading, error } = catalog || localCatalog;

  const providerInfo: TTSProviderInfo | undefined = data?.providers.find(
    p => p.provider === selection.ttsProvider
  );
  const activeModelId = selection.ttsModel || providerInfo?.default_model;
  const modelInfo: TTSModelInfo | undefined = providerInfo?.models.find(
    m => m.id === activeModelId
  );

  const selectionForModel = (
    provider: TTSProviderInfo,
    model: TTSModelInfo | undefined
  ): VoiceSelection => ({
    ...selection,
    ttsProvider: provider.provider,
    ttsModel: model?.id,
    ttsVoice: model?.voices[0]?.id,
    ttsLanguage: model?.languages[0]?.code
  });

  const handleProviderChange = (providerId: string) => {
    const provider = data?.providers.find(p => p.provider === providerId);
    if (!provider || !providerIsReady(provider)) {
      return;
    }
    const model = provider.models.find(
      m => m.id === provider.default_model && modelIsReady(m)
    ) || provider.models.find(modelIsReady);
    onChange(selectionForModel(provider, model));
  };

  const handleModelChange = (modelId: string) => {
    if (!providerInfo) {
      return;
    }
    const model = providerInfo.models.find(m => m.id === modelId);
    if (!model || !modelIsReady(model)) {
      return;
    }
    onChange(selectionForModel(providerInfo, model));
  };

  return (
    <div className="voice-settings">
      <button
        type="button"
        className={`voice-settings-toggle${expanded ? ' expanded' : ''}`}
        onClick={() => setExpanded(prev => !prev)}
        disabled={disabled}
        aria-expanded={expanded}
        aria-label="Voice settings"
        title={disabled ? 'Stop recording to change voice settings' : 'Voice settings'}
      >
        <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          {expanded ? (
            <path d="M10 6l6 8H4l6-8z" />
          ) : (
            <path d="M10 14l-6-8h12l-6 8z" />
          )}
        </svg>
      </button>

      {expanded && (
        <div className="voice-settings-panel">
          <div className="voice-settings-title">Voice settings</div>

          <label className="voice-settings-field">
            Speech to text
            <select
              value={selection.sttProvider}
              onChange={e => onChange({ ...selection, sttProvider: e.target.value })}
            >
              {STT_PROVIDERS.map(provider => (
                <option key={provider.id} value={provider.id}>
                  {provider.display_name}
                </option>
              ))}
            </select>
          </label>

          {loading && <div className="voice-settings-status">Loading voice models...</div>}
          {error && <div className="voice-settings-status voice-settings-error">{error}</div>}

          {data && (
            <>
              <label className="voice-settings-field">
                Voice provider
                <select
                  value={selection.ttsProvider}
                  onChange={e => handleProviderChange(e.target.value)}
                >
                  {data.providers.map(provider => {
                    const ready = providerIsReady(provider);
                    return (
                      <option key={provider.provider} value={provider.provider} disabled={!ready}>
                        {provider.display_name}{ready ? '' : ' (download or runtime required)'}
                      </option>
                    );
                  })}
                </select>
              </label>

              {providerInfo && providerInfo.models.length > 1 && (
                <label className="voice-settings-field">
                  Model
                  <select
                    value={activeModelId || ''}
                    onChange={e => handleModelChange(e.target.value)}
                  >
                    {providerInfo.models.map(model => {
                      const ready = modelIsReady(model);
                      return (
                        <option key={model.id} value={model.id} disabled={!ready}>
                          {model.display_name}{ready ? '' : ' (not ready)'}
                        </option>
                      );
                    })}
                  </select>
                </label>
              )}

              {modelInfo?.artifact && !modelIsReady(modelInfo) && (
                <div className="voice-settings-status">
                  {modelInfo.artifact.status !== 'installed'
                    ? 'Download this voice model from the Models tab first.'
                    : modelInfo.artifact.runtime_detail || 'The local voice runtime is not ready.'}
                </div>
              )}

              {modelInfo && modelInfo.voices.length > 1 && (
                <label className="voice-settings-field">
                  Voice
                  <select
                    value={selection.ttsVoice || modelInfo.voices[0].id}
                    onChange={e => onChange({ ...selection, ttsVoice: e.target.value })}
                  >
                    {modelInfo.voices.map(voice => (
                      <option key={voice.id} value={voice.id}>
                        {voice.display_name}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              {modelInfo && modelInfo.languages.length > 1 && (
                <label className="voice-settings-field">
                  Language
                  <select
                    value={selection.ttsLanguage || modelInfo.languages[0].code}
                    onChange={e => onChange({ ...selection, ttsLanguage: e.target.value })}
                  >
                    {modelInfo.languages.map(language => (
                      <option key={language.code} value={language.code}>
                        {language.display_name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default VoiceSettings;
