import React, { StrictMode } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import useVoiceSelection, { resolveVoiceSelection, VOICE_SELECTION_KEY } from '../useVoiceSelection';
import { VoiceModelsResponse } from '../useVoiceModels';

const catalog: VoiceModelsResponse = {
  default_provider: 'kokoro',
  providers: ['kokoro', 'qwen3'].map(provider => ({
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
};

beforeEach(() => {
  localStorage.clear();
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => catalog });
});

it('applies the installed default without opening settings, including in StrictMode', async () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => <StrictMode>{children}</StrictMode>;
  const { result } = renderHook(() => useVoiceSelection(true), { wrapper });
  expect(result.current.ready).toBe(false);
  await waitFor(() => expect(result.current.ready).toBe(true));
  expect(result.current.selection.ttsProvider).toBe('kokoro');
  expect(result.current.selection.ttsVoice).toBe('af_heart');
});

it('preserves an explicit valid selection across remounts', async () => {
  const { result, unmount } = renderHook(() => useVoiceSelection(true));
  await waitFor(() => expect(result.current.ready).toBe(true));
  act(() => result.current.setSelection({ sttProvider: 'mms', ttsProvider: 'qwen3' }));
  unmount();
  expect(localStorage.getItem(VOICE_SELECTION_KEY)).toContain('qwen3');
  const { result: restored } = renderHook(() => useVoiceSelection(true));
  await waitFor(() => expect(restored.current.ready).toBe(true));
  expect(restored.current.selection.ttsProvider).toBe('qwen3');
});

it('falls back when a saved model becomes unavailable on this architecture', () => {
  const unavailable = JSON.parse(JSON.stringify(catalog));
  unavailable.providers[1].models[0].artifact.runtime_ready = false;
  expect(resolveVoiceSelection(unavailable, { sttProvider: 'mms', ttsProvider: 'qwen3' })?.ttsProvider)
    .toBe('kokoro');
});
