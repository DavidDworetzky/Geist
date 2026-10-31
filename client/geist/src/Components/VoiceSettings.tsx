import React, { useState } from 'react';
import useVoiceModels, { TTSModelInfo, TTSProviderInfo, UseVoiceModelsReturn } from '../Hooks/useVoiceModels';
import { CONVERSATION_PROVIDERS } from '../Utils/pipelineVoiceSession';
import BrandMark from './BrandMark';

export interface VoiceSelection {
  mode: 'dictation' | 'live';
  sttProvider: string;
  ttsProvider: string;
  ttsModel?: string;
  ttsVoice?: string;
  ttsLanguage?: string;
}

export const DEFAULT_VOICE_SELECTION: VoiceSelection = {
  mode: 'dictation', sttProvider: 'mms', ttsProvider: 'openai_live',
  ttsModel: 'gpt-live-1', ttsVoice: 'marin'
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

/** Live chat offers speech-to-speech providers and downloadable local voices. */
export const isLiveProvider = (provider: TTSProviderInfo): boolean => (
  provider.mode === 'conversation' || provider.models.some(model => Boolean(model.artifact))
);

export const usesVoicePipeline = (selection: VoiceSelection): boolean => (
  selection.mode === 'live' && !CONVERSATION_PROVIDERS.has(selection.ttsProvider)
);

interface VoiceSettingsProps {
  selection: VoiceSelection;
  onChange: (selection: VoiceSelection) => void;
  disabled?: boolean;
  catalog?: UseVoiceModelsReturn;
}

const VoiceSettings: React.FC<VoiceSettingsProps> = ({ selection, onChange, disabled = false, catalog }) => {
  const [expanded, setExpanded] = useState(false);
  const localCatalog = useVoiceModels(expanded && selection.mode === 'live' && !catalog);
  const { data, loading, error } = catalog || localCatalog;
  const providers = data?.providers.filter(isLiveProvider) || [];
  const provider = providers.find(item => item.provider === selection.ttsProvider);
  const model = provider?.models.find(item => item.id === (selection.ttsModel || provider.default_model));
  const sttSelect = (
    <label className="voice-settings-field">Speech to text
      <select disabled={disabled} value={selection.sttProvider}
        onChange={event => onChange({ ...selection, sttProvider: event.target.value })}>
        {STT_PROVIDERS.map(item => <option key={item.id} value={item.id}>{item.display_name}</option>)}
      </select>
    </label>
  );

  const selectModel = (value: string) => {
    const selectedProvider = providers.find(item => item.models.some(m => `${item.provider}/${m.id}` === value));
    const selectedModel = selectedProvider?.models.find(item => `${selectedProvider?.provider}/${item.id}` === value);
    if (!selectedProvider || !selectedModel || !modelIsReady(selectedModel)) return;
    onChange({ ...selection, ttsProvider: selectedProvider.provider, ttsModel: selectedModel.id,
      ttsVoice: selectedModel.voices[0]?.id, ttsLanguage: selectedModel.languages?.[0]?.code });
  };

  return (
    <div className="voice-controls">
      <div className="voice-settings">
        <button type="button" className="voice-settings-toggle" disabled={disabled}
          aria-expanded={expanded} aria-label="Voice settings" onClick={() => setExpanded(value => !value)}>⚙</button>
        {expanded && (
          <div className="voice-settings-panel">
            <div className="voice-settings-title">{selection.mode === 'live' ? 'Live chat' : 'Dictation'}</div>
            {selection.mode === 'dictation' ? (
              <>
                <div className="voice-settings-status">Transcribe into your message. Review it, then press Send.</div>
                {sttSelect}
              </>
            ) : (
              <>
                <div className="voice-settings-status">A continuous conversation with AI-generated speech.</div>
                {provider?.description && <div className="voice-settings-status">{provider.description}</div>}
                {loading && <div className="voice-settings-status">Loading voice models...</div>}
                {error && <div className="voice-settings-error">{error}</div>}
                {data && <label className="voice-settings-field">Live voice model
                  <select disabled={disabled} value={`${selection.ttsProvider}/${selection.ttsModel || provider?.default_model || ''}`}
                    onChange={event => selectModel(event.target.value)}>
                    {providers.flatMap(item => item.models.map(option => {
                      const ready = modelIsReady(option);
                      return (
                        <option key={`${item.provider}/${option.id}`} value={`${item.provider}/${option.id}`} disabled={!ready}>
                          {item.display_name} · {option.display_name}{ready ? '' : ' (download or runtime required)'}
                        </option>
                      );
                    }))}
                  </select>
                </label>}
                {model?.artifact && !modelIsReady(model) && (
                  <div className="voice-settings-status">
                    {model.artifact.status !== 'installed'
                      ? 'Download this voice model from the Models tab first.'
                      : model.artifact.runtime_detail || 'The local voice runtime is not ready.'}
                  </div>
                )}
                {usesVoicePipeline(selection) && sttSelect}
                {model && model.voices.length > 0 && <label className="voice-settings-field">Voice
                  <select disabled={disabled} value={selection.ttsVoice || model.voices[0]?.id}
                    onChange={event => onChange({ ...selection, ttsVoice: event.target.value })}>
                    {model.voices.map(voice => <option key={voice.id} value={voice.id}>{voice.display_name}</option>)}
                  </select>
                </label>}
                {model && (model.languages?.length ?? 0) > 1 && <label className="voice-settings-field">Language
                  <select disabled={disabled} value={selection.ttsLanguage || model.languages[0].code}
                    onChange={event => onChange({ ...selection, ttsLanguage: event.target.value })}>
                    {model.languages.map(language => (
                      <option key={language.code} value={language.code}>{language.display_name}</option>
                    ))}
                  </select>
                </label>}
              </>
            )}
          </div>
        )}
      </div>
      <button type="button" className="live-chat-toggle" disabled={disabled}
        aria-label="Live chat" title="Live chat" aria-pressed={selection.mode === 'live'}
        onClick={() => onChange({ ...selection, mode: selection.mode === 'live' ? 'dictation' : 'live' })}>
        <BrandMark />
      </button>
    </div>
  );
};

export default VoiceSettings;
