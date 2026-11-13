import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import EnhancedChatInput from './EnhancedChatInput';

jest.mock('../Hooks/useVoiceChat', () => ({
  __esModule: true,
  default: () => ({ isRecording: false, isProcessing: false, partialTranscript: '', toggleRecording: jest.fn() }),
}));

test.each([
  { key: 'Enter', isComposing: true },
  { key: 'Enter', keyCode: 229 },
  { key: 'ArrowDown', isComposing: true },
])('leaves IME keys to the browser: %j', (event) => {
  const onSubmit = jest.fn();
  const handleKeyDown = jest.fn();
  render(<EnhancedChatInput value="日本語" onChange={jest.fn()} onSubmit={onSubmit}
    handleKeyDown={handleKeyDown} enableVoice={false} />);
  expect(fireEvent.keyDown(screen.getByRole('textbox'), event)).toBe(true);
  expect(onSubmit).not.toHaveBeenCalled();
  expect(handleKeyDown).not.toHaveBeenCalled();
});

test('Enter sends after composition finishes and Shift+Enter remains a newline', () => {
  const onSubmit = jest.fn();
  render(<EnhancedChatInput value="日本語" onChange={jest.fn()} onSubmit={onSubmit} enableVoice={false} />);
  const input = screen.getByRole('textbox');
  fireEvent.compositionStart(input);
  fireEvent.keyDown(input, { key: 'Enter', isComposing: true });
  fireEvent.compositionEnd(input);
  expect(fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })).toBe(true);
  expect(onSubmit).not.toHaveBeenCalled();
  expect(fireEvent.keyDown(input, { key: 'Enter' })).toBe(false);
  expect(onSubmit).toHaveBeenCalledTimes(1);
  expect(onSubmit).toHaveBeenCalledWith('日本語');
});

test('places the badge beside unavailable text and preserves the draft through recovery', () => {
  const onChange = jest.fn();
  const onShow = jest.fn();
  const props = { value: 'Keep my draft', onChange, onSubmit: jest.fn(), enableVoice: false };
  const view = render(<EnhancedChatInput {...props} disabled onShowModelError={onShow} />);
  const badge = screen.getByRole('button', { name: 'See more…' });
  expect(screen.getByRole('textbox', { name: 'Message' }))
    .toHaveAccessibleDescription('Model unavailable See more…');
  expect(screen.getByRole('textbox', { name: 'Message' })).toBeDisabled();
  fireEvent.click(badge);
  expect(onShow).toHaveBeenCalledTimes(1);
  expect(onChange).not.toHaveBeenCalled();
  view.rerender(<EnhancedChatInput {...props} />);
  expect(screen.getByRole('textbox', { name: 'Message' })).toHaveValue('Keep my draft');
  expect(screen.queryByRole('button', { name: 'See more…' })).not.toBeInTheDocument();
});
