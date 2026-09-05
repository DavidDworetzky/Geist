import { useCallback, useState } from 'react';
import { DEFAULT_VOICE_SELECTION, modelIsReady, VoiceSelection } from '../Components/VoiceSettings';
import useVoiceModels, { VoiceModelsResponse } from './useVoiceModels';

export const VOICE_SELECTION_KEY = 'geist.voice-selection.v1';

export function resolveVoiceSelection(
  catalog: VoiceModelsResponse, saved: VoiceSelection | null,
): VoiceSelection | null {
  const savedProvider = catalog.providers.find(p => p.provider === saved?.ttsProvider);
  const savedModel = savedProvider?.models.find(m => (
    m.id === (saved?.ttsModel || savedProvider.default_model) && modelIsReady(m)
  ));
  const provider = savedModel ? savedProvider : (
    catalog.providers.find(p => p.provider === catalog.default_provider && p.models.some(modelIsReady))
    || catalog.providers.find(p => p.models.some(modelIsReady))
  );
  const model = savedModel || provider?.models.find(m => m.id === provider.default_model && modelIsReady(m))
    || provider?.models.find(modelIsReady);
  if (!provider || !model) return null;
  return {
    sttProvider: saved?.sttProvider === 'whisper' ? 'whisper' : 'mms',
    ttsProvider: provider.provider,
    ttsModel: model.id,
    ttsVoice: savedModel && model.voices.some(v => v.id === saved?.ttsVoice)
      ? saved?.ttsVoice : model.voices[0]?.id,
    ttsLanguage: savedModel && model.languages.some(l => l.code === saved?.ttsLanguage)
      ? saved?.ttsLanguage : model.languages[0]?.code,
  };
}

export default function useVoiceSelection(enabled: boolean) {
  const catalog = useVoiceModels(enabled);
  const [saved, setSaved] = useState<VoiceSelection | null>(() => {
    try { return JSON.parse(localStorage.getItem(VOICE_SELECTION_KEY) || 'null'); }
    catch { return null; }
  });
  const resolved = catalog.data ? resolveVoiceSelection(catalog.data, saved) : null;
  const setSelection = useCallback((selection: VoiceSelection) => {
    setSaved(selection);
    try { localStorage.setItem(VOICE_SELECTION_KEY, JSON.stringify(selection)); }
    catch { /* Storage can be disabled in private browsers. */ }
  }, []);
  return { selection: resolved || DEFAULT_VOICE_SELECTION, setSelection, catalog,
    ready: !!resolved && !catalog.loading };
}
