import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import VoiceSettings, { DEFAULT_VOICE_SELECTION } from '../VoiceSettings';

const voiceModelsResponse = {
  default_provider: 'sesame',
  providers: [
    {
      provider: 'sesame',
      display_name: 'Sesame CSM',
      type: 'local',
      default_model: 'sesame/csm-1b',
      models: [
        {
          id: 'sesame/csm-1b',
          display_name: 'Sesame CSM 1B',
          sample_rate: 24000,
          supports_streaming: false,
          streaming_mode: 'chunked_full_audio',
          supports_instruction_control: false,
          supports_voice_cloning: false,
          voices: [{ id: '0', display_name: 'Default Speaker' }],
          languages: [{ code: 'en', display_name: 'English' }],
        },
      ],
    },
    {
      provider: 'qwen3',
      display_name: 'Qwen3 TTS (local MLX)',
      type: 'local',
      default_model: 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
      models: [
        {
          id: 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
          display_name: 'Qwen3 TTS 0.6B Custom Voice',
          sample_rate: 24000,
          supports_streaming: true,
          streaming_mode: 'native_pcm',
          supports_instruction_control: false,
          supports_voice_cloning: false,
          voices: [{ id: 'Aiden', display_name: 'Aiden' }],
          languages: [{ code: 'English', display_name: 'English' }],
          artifact: {
            id: 'qwen3-tts-0.6b-customvoice-mlx-6bit',
            status: 'installed',
            supported: true,
            runtime_ready: true,
          },
        },
      ],
    },
  ],
};

describe('VoiceSettings', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('renders collapsed without fetching voice models', () => {
    render(
      <VoiceSettings selection={DEFAULT_VOICE_SELECTION} onChange={() => {}} />
    );

    expect(screen.getByLabelText('Voice settings')).toBeInTheDocument();
    expect(screen.queryByText('Speech to text')).not.toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('expands, fetches the catalog, and preselects the defaults', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => voiceModelsResponse,
    });

    render(
      <VoiceSettings selection={DEFAULT_VOICE_SELECTION} onChange={() => {}} />
    );

    fireEvent.click(screen.getByLabelText('Voice settings'));

    expect(global.fetch).toHaveBeenCalledWith('/api/v1/voice/models');
    expect(screen.getByLabelText(/Speech to text/i)).toHaveValue('mms');

    await waitFor(() => {
      expect(screen.getByLabelText(/Voice provider/i)).toHaveValue('sesame');
    });

    // Sesame has a single model/voice/language, so those selects stay hidden
    expect(screen.queryByLabelText(/Model/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^Voice$/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Language/)).not.toBeInTheDocument();
  });

  it('selecting a provider resets model, voice, and language to its defaults', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => voiceModelsResponse,
    });
    const onChange = jest.fn();

    render(<VoiceSettings selection={DEFAULT_VOICE_SELECTION} onChange={onChange} />);

    fireEvent.click(screen.getByLabelText('Voice settings'));
    await screen.findByLabelText(/Voice provider/i);

    fireEvent.change(screen.getByLabelText(/Voice provider/i), {
      target: { value: 'qwen3' },
    });

    expect(onChange).toHaveBeenCalledWith({
      sttProvider: 'mms',
      ttsProvider: 'qwen3',
      ttsModel: 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
      ttsVoice: 'Aiden',
      ttsLanguage: 'English',
    });
  });

  it('disables a local provider until its weights and runtime are ready', async () => {
    const unavailableResponse = JSON.parse(JSON.stringify(voiceModelsResponse));
    const artifact = unavailableResponse.providers[1].models[0].artifact;
    artifact.status = 'not_installed';
    artifact.runtime_ready = false;
    artifact.runtime_detail = 'MLX Audio is not installed.';
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => unavailableResponse,
    });
    const onChange = jest.fn();

    render(<VoiceSettings selection={DEFAULT_VOICE_SELECTION} onChange={onChange} />);

    fireEvent.click(screen.getByLabelText('Voice settings'));
    const select = await screen.findByLabelText(/Voice provider/i);
    const qwenOption = screen.getByRole('option', {
      name: /Qwen3 TTS.*download or runtime required/i,
    });

    expect(qwenOption).toBeDisabled();
    fireEvent.change(select, { target: { value: 'qwen3' } });
    expect(onChange).not.toHaveBeenCalled();
  });

  it('surfaces an error when the catalog fails to load', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      statusText: 'Internal Server Error',
    });

    render(
      <VoiceSettings selection={DEFAULT_VOICE_SELECTION} onChange={() => {}} />
    );

    fireEvent.click(screen.getByLabelText('Voice settings'));

    await waitFor(() => {
      expect(
        screen.getByText(/Failed to load voice models/i)
      ).toBeInTheDocument();
    });
  });
});
