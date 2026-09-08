import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import EnhancedChatInput from '../EnhancedChatInput';

let mockRecording = false;
const mockSetSelection = jest.fn();
const mockVoiceOptions = jest.fn();

jest.mock('../../Hooks/useVoiceChat', () => ({
  __esModule: true,
  default: (options: unknown) => {
    mockVoiceOptions(options);
    return {
      isRecording: mockRecording,
      isProcessing: false,
      partialTranscript: '',
      toggleRecording: jest.fn(),
    };
  },
}));

jest.mock('../../Hooks/useVoiceSelection', () => ({
  __esModule: true,
  default: () => ({
    selection: { sttProvider: 'mms', ttsProvider: 'kokoro' },
    setSelection: mockSetSelection,
    ready: true,
    catalog: {
      loading: false,
      error: null,
      data: {
        default_provider: 'kokoro',
        providers: [{
          provider: 'kokoro', display_name: 'Vera (Kokoro)', default_model: 'kokoro',
          models: [{ id: 'kokoro', voices: [{ id: 'af_heart' }], languages: [{ code: 'a' }] }],
        }],
      },
    },
  }),
}));

beforeEach(() => {
  mockRecording = false;
  mockSetSelection.mockClear();
  mockVoiceOptions.mockClear();
});

it.each(['local', 'online'])('passes the %s chat runtime to voice chat', (agentType) => {
  render(<EnhancedChatInput value="" onChange={jest.fn()} onSubmit={jest.fn()} agentType={agentType} />);
  expect(mockVoiceOptions).toHaveBeenLastCalledWith(expect.objectContaining({ agentType }));
});

it('allows voice selection while chat is disabled, but still blocks recording', () => {
  render(<EnhancedChatInput value="" onChange={jest.fn()} onSubmit={jest.fn()} disabled />);
  const settings = screen.getByRole('button', { name: 'Voice settings' });
  expect(settings).toBeEnabled();
  fireEvent.click(settings);
  const provider = screen.getByLabelText('Voice provider');
  expect(provider).toBeEnabled();
  fireEvent.change(provider, { target: { value: 'kokoro' } });
  expect(mockSetSelection).toHaveBeenCalledWith(expect.objectContaining({
    ttsProvider: 'kokoro', ttsVoice: 'af_heart',
  }));
  expect(screen.getByRole('button', { name: 'Voice chat disabled' })).toBeDisabled();
});

it('locks an already-open provider selector when recording starts', () => {
  const props = { value: '', onChange: jest.fn(), onSubmit: jest.fn() };
  const { rerender } = render(<EnhancedChatInput {...props} />);
  fireEvent.click(screen.getByRole('button', { name: 'Voice settings' }));
  mockRecording = true;
  rerender(<EnhancedChatInput {...props} />);
  expect(screen.getByRole('button', { name: 'Voice settings' })).toBeDisabled();
  expect(screen.getByLabelText('Voice provider')).toBeDisabled();
  expect(screen.getByLabelText('Speech to text')).toBeDisabled();
});
