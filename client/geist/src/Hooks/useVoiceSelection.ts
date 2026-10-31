import { useCallback, useState } from 'react';
import { DEFAULT_VOICE_SELECTION, isLiveProvider, modelIsReady, VoiceSelection } from '../Components/VoiceSettings';
import { CONVERSATION_PROVIDERS } from '../Utils/pipelineVoiceSession';
import useVoiceModels, { VoiceModelsResponse } from './useVoiceModels';

export const VOICE_SELECTION_KEY = 'geist.voice-selection.v1';

/** Keep a saved live voice when it is still installed and runnable; otherwise pick the best ready one. */
export function resolveVoiceSelection(
  catalog: VoiceModelsResponse, saved: VoiceSelection,
): VoiceSelection | null {
  const providers = catalog.providers.filter(isLiveProvider);
  const savedProvider = providers.find(p => p.provider === saved.ttsProvider);
  const savedModel = savedProvider?.models.find(m => (
    m.id === (saved.ttsModel || savedProvider.default_model) && modelIsReady(m)
  ));
  const provider = savedModel ? savedProvider : (
    providers.find(p => p.provider === catalog.default_provider && p.models.some(modelIsReady))
    || providers.find(p => p.models.some(modelIsReady))
  );
  const model = savedModel || provider?.models.find(m => m.id === provider.default_model && modelIsReady(m))
    || provider?.models.find(modelIsReady);
  if (!provider || !model) return null;
  return {
    mode: saved.mode,
    sttProvider: saved.sttProvider,
    ttsProvider: provider.provider,
    ttsModel: model.id,
    ttsVoice: savedModel && model.voices.some(v => v.id === saved.ttsVoice)
      ? saved.ttsVoice : model.voices[0]?.id,
    ttsLanguage: savedModel && model.languages?.some(l => l.code === saved.ttsLanguage)
      ? saved.ttsLanguage : model.languages?.[0]?.code,
  };
}

function loadSelection(): VoiceSelection {
  let saved: Partial<VoiceSelection> | null = null;
  try {
    saved = JSON.parse(localStorage.getItem(VOICE_SELECTION_KEY) || 'null');
  } catch {
    saved = null;
  }
  return {
    ...DEFAULT_VOICE_SELECTION,
    ...saved,
    mode: saved?.mode === 'live' ? 'live' : 'dictation',
    sttProvider: saved?.sttProvider === 'whisper' ? 'whisper' : 'mms',
  };
}

export default function useVoiceSelection(enabled: boolean) {
  const [saved, setSaved] = useState<VoiceSelection>(loadSelection);
  const live = saved.mode === 'live';
  const catalog = useVoiceModels(enabled && live);
  const resolved = live && catalog.data ? resolveVoiceSelection(catalog.data, saved) : null;
  const setSelection = useCallback((selection: VoiceSelection) => {
    setSaved(selection);
    try {
      localStorage.setItem(VOICE_SELECTION_KEY, JSON.stringify(selection));
    } catch {
      // Storage can be disabled in private browsers.
    }
  }, []);
  // Local voices must be confirmed installed before recording; hosted and
  // speech-to-speech providers can start while the catalog is unavailable.
  const ready = !live || (resolved
    ? !catalog.loading
    : !catalog.data && CONVERSATION_PROVIDERS.has(saved.ttsProvider));
  return { selection: resolved || saved, setSelection, catalog, ready };
}
