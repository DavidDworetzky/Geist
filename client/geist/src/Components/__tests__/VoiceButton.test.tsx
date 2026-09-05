import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import VoiceButton from '../VoiceButton';

it('keeps Stop usable during processing even if the chat input becomes disabled', () => {
  const onClick = jest.fn();
  render(<VoiceButton isRecording isProcessing disabled onClick={onClick} />);
  const stop = screen.getByRole('button', { name: 'Click to stop recording' });
  expect(stop).toBeEnabled();
  fireEvent.click(stop);
  expect(onClick).toHaveBeenCalledTimes(1);
});
