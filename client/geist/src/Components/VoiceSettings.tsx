import React, { useState } from 'react';
import useVoiceModels from '../Hooks/useVoiceModels';

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

interface VoiceSettingsProps {
  selection: VoiceSelection;
  onChange: (selection: VoiceSelection) => void;
  disabled?: boolean;
}

const VoiceSettings: React.FC<VoiceSettingsProps> = ({ selection, onChange, disabled = false }) => {
  const [expanded, setExpanded] = useState(false);
  const { data, loading, error } = useVoiceModels(expanded && selection.mode === 'live');
  const providers = data?.providers.filter(provider => provider.mode === 'conversation') || [];
  const provider = providers.find(item => item.provider === selection.ttsProvider);
  const model = provider?.models.find(item => item.id === (selection.ttsModel || provider.default_model));

  return (
    <div className="voice-controls">
      <div className="voice-mode-switch" role="group" aria-label="Voice mode">
        <button type="button" disabled={disabled} aria-pressed={selection.mode === 'dictation'}
          onClick={() => onChange({ ...selection, mode: 'dictation' })}>Dictation</button>
        <button type="button" disabled={disabled} aria-pressed={selection.mode === 'live'}
          onClick={() => onChange({ ...selection, mode: 'live' })}>Live chat</button>
      </div>
      <div className="voice-settings">
        <button type="button" className="voice-settings-toggle" disabled={disabled}
          aria-expanded={expanded} aria-label="Voice settings" onClick={() => setExpanded(value => !value)}>⚙</button>
        {expanded && (
          <div className="voice-settings-panel">
            <div className="voice-settings-title">{selection.mode === 'live' ? 'Live chat' : 'Dictation'}</div>
            {selection.mode === 'dictation' ? (
              <>
                <div className="voice-settings-status">Transcribe into your message. Review it, then press Send.</div>
                <label className="voice-settings-field">Speech to text
                  <select disabled={disabled} value={selection.sttProvider}
                    onChange={event => onChange({ ...selection, sttProvider: event.target.value })}>
                    <option value="mms">MMS (local)</option>
                    <option value="whisper">Whisper (OpenAI API)</option>
                  </select>
                </label>
              </>
            ) : (
              <>
                <div className="voice-settings-status">A continuous conversation with AI-generated speech.</div>
                {selection.ttsProvider === 'moshi' && <div className="voice-settings-status">Requires Apple Silicon and the local Moshi runtime. English; five-minute calls. Captions show Moshi's speech.</div>}
                {loading && <div className="voice-settings-status">Loading voice models...</div>}
                {error && <div className="voice-settings-error">{error}</div>}
                {data && <label className="voice-settings-field">Live voice model
                  <select disabled={disabled} value={selection.ttsModel || ''} onChange={event => {
                    const selectedProvider = providers.find(item => item.models.some(m => m.id === event.target.value));
                    const selectedModel = selectedProvider?.models.find(item => item.id === event.target.value);
                    if (selectedProvider && selectedModel) onChange({ ...selection, ttsProvider: selectedProvider.provider,
                      ttsModel: selectedModel.id, ttsVoice: selectedModel.voices[0]?.id });
                  }}>
                    {providers.flatMap(item => item.models.map(model => (
                      <option key={model.id} value={model.id}>{item.display_name} · {model.display_name}</option>
                    )))}
                  </select>
                </label>}
                {model && <label className="voice-settings-field">Voice
                  <select disabled={disabled} value={selection.ttsVoice || model.voices[0]?.id}
                    onChange={event => onChange({ ...selection, ttsVoice: event.target.value })}>
                    {model.voices.map(voice => <option key={voice.id} value={voice.id}>{voice.display_name}</option>)}
                  </select>
                </label>}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceSettings;
