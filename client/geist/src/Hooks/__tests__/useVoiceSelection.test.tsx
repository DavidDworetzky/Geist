import React, { StrictMode } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import useVoiceSelection, { resolveVoiceSelection, VOICE_SELECTION_KEY } from '../useVoiceSelection';
import { VoiceModelsResponse } from '../useVoiceModels';
import { DEFAULT_VOICE_SELECTION, VoiceSelection } from '../../Components/VoiceSettings';

const catalog: VoiceModelsResponse = {
  default_provider: 'kokoro',
  providers: [
    ...['kokoro', 'qwen3'].map(provider => ({
      provider, display_name: provider, type: 'local', default_model: provider,
      models: [{
        id: provider, display_name: provider, sample_rate: 24000,
        supports_streaming: true, streaming_mode: 'sentence',
        supports_instruction_control: false, supports_voice_cloning: false,
        voices: [{ id: provider === 'kokoro' ? 'af_heart' : 'Vivian', display_name: 'Voice' }],
        languages: [{ code: 'en', display_name: 'English' }],
        artifact: { id: provider, status: 'installed', supported: true, runtime_ready: true },
      }],
    })),
    {
      provider: 'openai_live', display_name: 'GPT-Live', type: 'api', mode: 'conversation', default_model: 'gpt-live-1',
      models: [{
        id: 'gpt-live-1', display_name: 'GPT-Live 1', sample_rate: 24000, supports_streaming: true,
        streaming_mode: 'webrtc', supports_instruction_control: false, supports_voice_cloning: false,
        voices: [{ id: 'marin', display_name: 'Marin' }], languages: [],
      }],
    },
  ],
};

const liveSelection: VoiceSelection = { ...DEFAULT_VOICE_SELECTION, mode: 'live' };

beforeEach(() => {
  localStorage.clear();
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => catalog });
});

it('defaults to dictation without fetching the voice catalog', () => {
  const { result } = renderHook(() => useVoiceSelection(true));
  expect(result.current.selection.mode).toBe('dictation');
  expect(result.current.ready).toBe(true);
  expect(global.fetch).not.toHaveBeenCalled();
});

it('applies the installed local default in live mode, including in StrictMode', async () => {
  localStorage.setItem(VOICE_SELECTION_KEY, JSON.stringify({ mode: 'live', sttProvider: 'mms', ttsProvider: 'kokoro' }));
  const wrapper = ({ children }: { children: React.ReactNode }) => <StrictMode>{children}</StrictMode>;
  const { result } = renderHook(() => useVoiceSelection(true), { wrapper });
  expect(result.current.ready).toBe(false);
  await waitFor(() => expect(result.current.ready).toBe(true));
  expect(result.current.selection.ttsProvider).toBe('kokoro');
  expect(result.current.selection.ttsVoice).toBe('af_heart');
});

it('lets GPT-Live start before the catalog arrives', () => {
  global.fetch = jest.fn(() => new Promise(() => {})) as any;
  localStorage.setItem(VOICE_SELECTION_KEY, JSON.stringify(liveSelection));
  const { result } = renderHook(() => useVoiceSelection(true));
  expect(result.current.ready).toBe(true);
  expect(result.current.selection.ttsProvider).toBe('openai_live');
});

it('preserves an explicit valid selection across remounts', async () => {
  localStorage.setItem(VOICE_SELECTION_KEY, JSON.stringify(liveSelection));
  const { result, unmount } = renderHook(() => useVoiceSelection(true));
  await waitFor(() => expect(result.current.ready).toBe(true));
  act(() => result.current.setSelection({ ...liveSelection, ttsProvider: 'qwen3', ttsModel: 'qwen3' }));
  unmount();
  expect(localStorage.getItem(VOICE_SELECTION_KEY)).toContain('qwen3');
  const { result: restored } = renderHook(() => useVoiceSelection(true));
  await waitFor(() => expect(restored.current.ready).toBe(true));
  expect(restored.current.selection.ttsProvider).toBe('qwen3');
});

it('falls back when a saved model becomes unavailable on this architecture', () => {
  const unavailable = JSON.parse(JSON.stringify(catalog));
  unavailable.providers[1].models[0].artifact.runtime_ready = false;
  expect(resolveVoiceSelection(unavailable, { ...liveSelection, ttsProvider: 'qwen3', ttsModel: 'qwen3' })?.ttsProvider)
    .toBe('kokoro');
});

it('falls back to GPT-Live when no local voice is installed', () => {
  const missing = JSON.parse(JSON.stringify(catalog));
  missing.providers.slice(0, 2).forEach((provider: any) => { provider.models[0].artifact.status = 'not_installed'; });
  missing.default_provider = 'sesame';
  expect(resolveVoiceSelection(missing, { ...liveSelection, ttsProvider: 'kokoro' })?.ttsProvider)
    .toBe('openai_live');
});
