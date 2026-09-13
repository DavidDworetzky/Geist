import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import EnhancedChatInput from './EnhancedChatInput';

jest.mock('../Hooks/useVoiceChat', () => ({
  __esModule: true,
  default: () => ({ isRecording: false, isProcessing: false, partialTranscript: '', toggleRecording: jest.fn() }),
}));

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
