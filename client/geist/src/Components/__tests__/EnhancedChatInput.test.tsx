import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import EnhancedChatInput from '../EnhancedChatInput';
import { VoiceSelection } from '../VoiceSettings';

let mockRecording = false;
let mockProcessing = false;
let mockSelection: VoiceSelection;
let mockReady = true;
const mockSetSelection = jest.fn();
const mockVoiceOptions = jest.fn();
const mockToggle = jest.fn();

jest.mock('../BrandMark', () => () => <span>Brand</span>);

jest.mock('../../Hooks/useVoiceChat', () => ({
  __esModule: true,
  default: (options: unknown) => {
    mockVoiceOptions(options);
    return {
      isRecording: mockRecording,
      isProcessing: mockProcessing,
      partialTranscript: '',
      assistantText: '',
      status: '',
      audioLevel: 0,
      toggleRecording: mockToggle,
    };
  },
}));

jest.mock('../../Hooks/useVoiceSelection', () => ({
  __esModule: true,
  default: () => ({
    selection: mockSelection,
    setSelection: mockSetSelection,
    ready: mockReady,
    catalog: {
      loading: false,
      error: null,
      data: {
        default_provider: 'kokoro',
        providers: [{
          provider: 'kokoro', display_name: 'Vera (Kokoro)', default_model: 'kokoro',
          models: [{
            id: 'kokoro', display_name: 'Kokoro 82M', voices: [{ id: 'af_heart', display_name: 'Vera' }],
            languages: [{ code: 'a', display_name: 'English' }],
            artifact: { id: 'kokoro-82m-bf16-mlx', status: 'installed', supported: true, runtime_ready: true },
          }],
        }],
      },
    },
  }),
}));

const vera: VoiceSelection = { mode: 'live', sttProvider: 'mms', ttsProvider: 'kokoro', ttsModel: 'kokoro', ttsVoice: 'af_heart' };
const props = { value: '', onChange: jest.fn(), onSubmit: jest.fn() };

beforeEach(() => {
  mockRecording = false;
  mockProcessing = false;
  mockReady = true;
  mockSelection = vera;
  mockSetSelection.mockClear();
  mockVoiceOptions.mockClear();
  mockToggle.mockClear();
});

it.each(['local', 'online'])('passes the %s chat runtime and selected voice to voice chat', (agentType) => {
  render(<EnhancedChatInput {...props} agentType={agentType} />);
  expect(mockVoiceOptions).toHaveBeenLastCalledWith(expect.objectContaining({
    agentType, mode: 'live', ttsProvider: 'kokoro', ttsVoice: 'af_heart',
  }));
});

it('locks an already-open voice selector when recording starts', () => {
  const { rerender } = render(<EnhancedChatInput {...props} agentType="local" />);
  fireEvent.click(screen.getByRole('button', { name: 'Voice settings' }));
  mockRecording = true;
  rerender(<EnhancedChatInput {...props} agentType="local" />);
  expect(screen.getByRole('button', { name: 'Voice settings' })).toBeDisabled();
  expect(screen.getByLabelText('Live voice model')).toBeDisabled();
  expect(screen.getByLabelText('Speech to text')).toBeDisabled();
});

it('keeps the microphone stoppable when dictation input becomes disabled', () => {
  mockSelection = { ...vera, mode: 'dictation' };
  mockRecording = true;
  render(<EnhancedChatInput {...props} disabled />);
  fireEvent.click(screen.getByRole('button', { name: 'Finish dictation' }));
  expect(mockToggle).toHaveBeenCalledTimes(1);
});

it('keeps the call stoppable when the voice runtime leaves the ready state', () => {
  mockRecording = true;
  mockReady = false;
  render(<EnhancedChatInput {...props} agentType="local" />);
  fireEvent.click(screen.getByRole('button', { name: 'End voice call' }));
  expect(mockToggle).toHaveBeenCalledTimes(1);
});

it('keeps voice controls mounted for an active session after the chat disables voice', () => {
  mockRecording = true;
  render(<EnhancedChatInput {...props} agentType="local" enableVoice={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'End voice call' }));
  expect(mockToggle).toHaveBeenCalledTimes(1);
});

it('explains why a local voice cannot start when chat settings failed to load', () => {
  render(<EnhancedChatInput {...props} />);
  expect(screen.getByRole('button', { name: 'Voice chat disabled' })).toBeDisabled();
  expect(screen.getByText(/needs your chat settings/)).toBeInTheDocument();
});

it('still allows dictation when chat settings failed to load', () => {
  mockSelection = { ...vera, mode: 'dictation' };
  render(<EnhancedChatInput {...props} />);
  expect(screen.getByRole('button', { name: 'Start dictation' })).toBeEnabled();
});
