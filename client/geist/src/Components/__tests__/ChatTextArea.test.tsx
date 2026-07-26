import React from 'react';
import { render, screen } from '@testing-library/react';
import ChatTextArea from '../ChatTextArea';
import { ChatPair } from '../../chatTypes';

describe('ChatTextArea loading state', () => {
  it('shows the Geist loading indicator while chat is loading', () => {
    render(<ChatTextArea chatHistory={[]} isLoading />);

    expect(screen.getByRole('status', { name: 'Geist is responding' })).toBeInTheDocument();
    expect(screen.queryByText('Start a conversation with Geist.')).not.toBeInTheDocument();
  });

  it('hides the Geist loading indicator when chat is idle', () => {
    render(<ChatTextArea chatHistory={[]} isLoading={false} />);

    expect(screen.queryByRole('status', { name: 'Geist is responding' })).not.toBeInTheDocument();
  });

  it('shows the selected model lifecycle while weights are loading', () => {
    render(<ChatTextArea chatHistory={[{
      user: 'Hello',
      ai: '',
      status: 'model_loading',
      model_load: {
        model_id: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        state: 'loading',
        detail: 'Downloading or loading model files from the Hugging Face cache.',
        started_at: '2026-07-26T00:00:00Z',
        updated_at: '2026-07-26T00:00:01Z',
      },
    }]} isLoading />);

    expect(screen.getByRole('status', { name: 'Model loading status' })).toHaveTextContent(
      'meta-llama/Meta-Llama-3.1-8B-Instruct',
    );
    expect(screen.getByRole('status', { name: 'Model loading status' })).toHaveTextContent(
      'Downloading or loading model files from the Hugging Face cache.',
    );
    expect(screen.queryByText('Turn status: model loading')).not.toBeInTheDocument();
  });

  it('does not claim an unloaded model is actively loading', () => {
    render(<ChatTextArea chatHistory={[{
      user: 'Hello',
      ai: '',
      status: 'connecting',
      model_load: {
        model_id: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        state: 'unloaded',
        detail: 'Model is not loaded in this backend process.',
        started_at: null,
        updated_at: '2026-07-26T00:00:01Z',
      },
    }]} isLoading />);

    expect(screen.queryByRole('status', { name: 'Model loading status' })).not.toBeInTheDocument();
    expect(screen.queryByText('Model is not loaded in this backend process.'))
      .not.toBeInTheDocument();
    expect(screen.getByText('Turn status: connecting')).toBeInTheDocument();
  });
});

describe('ChatTextArea tool activity', () => {
  it('renders live status, arguments, approval readiness, failures, and artifacts', () => {
    const turn: ChatPair = {
      run_id: 'run_1',
      user: 'Research the framework',
      ai: 'Working on it',
      status: 'awaiting_approval',
      tool_calls: [
        {
          id: 'call_approval',
          name: 'filesystem.write',
          arguments: { filename: 'notes.md', content: 'pi framework' },
          status: 'awaiting_approval',
          requires_approval: true,
        },
        {
          id: 'call_running',
          name: 'search',
          arguments: { query: 'pi framework' },
          status: 'running',
        },
        {
          id: 'call_failed',
          name: 'fetch',
          arguments: { url: 'https://example.com' },
          status: 'failed',
          error: 'Request failed',
        },
      ],
      artifacts: [
        {
          id: 'image_1',
          kind: 'image',
          mime_type: 'image/png',
          filename: 'preview.png',
          sha256: 'image-hash',
          data_base64: 'aW1hZ2U=',
        },
        {
          id: 'text_1',
          kind: 'text',
          mime_type: 'text/plain',
          filename: 'notes.txt',
          sha256: 'text-hash',
          url: 'https://example.com/notes.txt',
        },
      ],
    };

    render(<ChatTextArea chatHistory={[turn]} />);

    expect(screen.getByText('Turn status: awaiting approval')).toBeInTheDocument();
    expect(screen.getByTestId('tool-call-call_approval')).toHaveTextContent(
      'filesystem.write (awaiting approval)',
    );
    expect(screen.getByText('Approval required')).toBeInTheDocument();
    expect(screen.getByTestId('tool-call-call_approval')).toHaveTextContent('pi framework');
    expect(screen.getByTestId('tool-call-call_running')).toHaveTextContent('search (running)');
    expect(screen.getByTestId('tool-call-call_failed')).toHaveTextContent('Request failed');

    expect(screen.getByRole('img', { name: 'preview.png' })).toHaveAttribute(
      'src',
      'data:image/png;base64,aW1hZ2U=',
    );
    expect(screen.getByRole('link', { name: 'notes.txt' })).toHaveAttribute(
      'href',
      'https://example.com/notes.txt',
    );
  });
});
