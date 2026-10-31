import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import EnhancedChatInput from '../EnhancedChatInput';
import useVoiceChat from '../../Hooks/useVoiceChat';

jest.mock('../../Hooks/useVoiceChat');
jest.mock('../../Hooks/useVoiceModels', () => () => ({ data: null, loading: false, error: null }));
jest.mock('../BrandMark', () => () => <span>Brand</span>);
const voice = useVoiceChat as jest.Mock;

beforeEach(() => {
  voice.mockReturnValue({ isRecording: false, isProcessing: false, partialTranscript: '', assistantText: '', status: '', audioLevel: 0, toggleRecording: jest.fn() });
});

it('appends dictation to the draft and waits for explicit Send', () => {
  const onChange = jest.fn();
  const onSubmit = jest.fn();
  render(<EnhancedChatInput value="Existing draft" onChange={onChange} onSubmit={onSubmit} />);
  voice.mock.calls.at(-1)[0].onTranscriptFinal('new words');
  expect(onChange).toHaveBeenCalledWith('Existing draft new words');
  expect(onSubmit).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Send', exact: true }));
  expect(onSubmit).toHaveBeenCalledWith('Existing draft');
});

it('live chat replaces the composer and restores the unsent draft on return', () => {
  render(<EnhancedChatInput value="Keep this draft" onChange={jest.fn()} onSubmit={jest.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: 'Live chat', exact: true }));
  expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Send', exact: true })).not.toBeInTheDocument();
  expect(screen.getByLabelText('Live voice conversation')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Live chat', exact: true }));
  expect(screen.getByRole('textbox')).toHaveValue('Keep this draft');
});

it('Enter cannot submit a half-transcribed message', () => {
  voice.mockReturnValue({ ...voice(), isProcessing: true });
  const onSubmit = jest.fn();
  const handleKeyDown = jest.fn();
  render(<EnhancedChatInput value="Partial draft" onChange={jest.fn()} onSubmit={onSubmit} handleKeyDown={handleKeyDown} />);
  fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
  expect(onSubmit).not.toHaveBeenCalled();
  expect(handleKeyDown).not.toHaveBeenCalled();
});


it('live voice is available when the normal text model is not installed', () => {
  render(<EnhancedChatInput disabled value="" onChange={jest.fn()} onSubmit={jest.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: 'Live chat', exact: true }));
  expect(screen.getByRole('button', { name: 'Start voice call', exact: true })).toBeEnabled();
});
