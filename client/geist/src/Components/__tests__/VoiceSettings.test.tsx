import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import VoiceSettings, { DEFAULT_VOICE_SELECTION, VoiceSelection } from '../VoiceSettings';

const catalog = { providers: [
  { provider: 'openai', display_name: 'OpenAI TTS', type: 'api', default_model: 'tts-1', models: [{ id: 'tts-1', display_name: 'TTS 1', voices: [] }] },
  { provider: 'openai_live', display_name: 'OpenAI GPT-Live', type: 'api', mode: 'conversation', default_model: 'gpt-live-1',
    models: [{ id: 'gpt-live-1', display_name: 'GPT-Live 1', voices: [{ id: 'marin', display_name: 'Marin' }, { id: 'quartz', display_name: 'Quartz' }] }] },
  { provider: 'local_live', display_name: 'Local live voice', description: 'Configured local voice engine', type: 'local', mode: 'conversation', default_model: 'kyutai/moshiko-mlx-q4',
    models: [{ id: 'kyutai/moshiko-mlx-q4', display_name: 'Moshiko 7B', voices: [{ id: 'moshiko', display_name: 'Moshiko' }] }] }
] };

function Settings() {
  const [selection, setSelection] = React.useState<VoiceSelection>(DEFAULT_VOICE_SELECTION);
  return <VoiceSettings selection={selection} onChange={setSelection} />;
}

describe('voice modes', () => {
  beforeEach(() => { global.fetch = jest.fn(async () => ({ ok: true, json: async () => catalog })) as any; });
  afterEach(() => jest.restoreAllMocks());

  it('defaults to dictation and only offers transcription settings', () => {
    render(<Settings />);
    expect(screen.getByRole('button', { name: 'Dictation' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByLabelText('Voice settings'));
    expect(screen.getByLabelText('Speech to text')).toHaveValue('mms');
    expect(screen.queryByLabelText('Live voice model')).not.toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('offers only voice-to-voice models for live chat and resets the voice', async () => {
    render(<Settings />);
    fireEvent.click(screen.getByRole('button', { name: 'Live chat' }));
    fireEvent.click(screen.getByLabelText('Voice settings'));
    await waitFor(() => expect(screen.getByLabelText('Live voice model')).toHaveValue('openai_live/gpt-live-1'));
    expect(screen.queryByLabelText('Speech to text')).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /TTS 1/ })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Live voice model'), { target: { value: 'local_live/kyutai/moshiko-mlx-q4' } });
    expect(screen.getByLabelText('Voice')).toHaveValue('moshiko');
    expect(screen.getByText('Configured local voice engine')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Dictation' }));
    expect(screen.getByLabelText('Speech to text')).toHaveValue('mms');
  });

  it('locks mode controls while a session is active', () => {
    render(<VoiceSettings selection={DEFAULT_VOICE_SELECTION} onChange={jest.fn()} disabled />);
    expect(screen.getByRole('button', { name: 'Live chat' })).toBeDisabled();
    expect(screen.getByLabelText('Voice settings')).toBeDisabled();
  });

  it('shows catalog failure without hiding the mode controls', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Catalog unavailable'));
    render(<Settings />);
    fireEvent.click(screen.getByRole('button', { name: 'Live chat' }));
    fireEvent.click(screen.getByLabelText('Voice settings'));
    expect(await screen.findByText('Catalog unavailable')).toBeInTheDocument();
  });
});


it('shows a configured external engine without a client-side voice selector', async () => {
  const externalCatalog = { providers: [{ ...catalog.providers[2], default_model: 'gpt-live-1', models: [{ id: 'gpt-live-1', display_name: 'Custom local engine', voices: [] }] }, catalog.providers[1]] };
  global.fetch = jest.fn(async () => ({ ok: true, json: async () => externalCatalog })) as any;
  render(<Settings />);
  fireEvent.click(screen.getByRole('button', { name: 'Live chat' }));
  fireEvent.click(screen.getByLabelText('Voice settings'));
  await screen.findByRole('option', { name: /Custom local engine/ });
  fireEvent.change(screen.getByLabelText('Live voice model'), { target: { value: 'local_live/gpt-live-1' } });
  expect(screen.getByLabelText('Live voice model')).toHaveValue('local_live/gpt-live-1');
  expect(screen.queryByLabelText('Voice')).not.toBeInTheDocument();
});
