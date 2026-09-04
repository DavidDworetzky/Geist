import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
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

describe('ChatTextArea agentic progress', () => {
  it('renders goal turns, task status, and evidence', () => {
    render(<ChatTextArea chatHistory={[{
      run_id: 'run_agentic',
      user: 'Build a feature',
      ai: 'Done',
      orchestration: {
        agentic_mode: true,
        goal_status: 'complete',
        turns_used: 2,
        max_turns: 8,
        tasks: [{
          id: 'task-1',
          title: 'Implement the UI',
          acceptance_criteria: ['UI test passes'],
          status: 'completed',
          evidence: 'UI test passes',
        }],
      },
    }]} />);

    const progress = screen.getByRole('region', { name: 'Agentic progress' });
    expect(progress).toHaveTextContent('Agentic plan');
    expect(progress).toHaveTextContent('complete');
    expect(progress).toHaveTextContent('Model calls 2/8');
    expect(progress).toHaveTextContent('Implement the UI');
    expect(progress).toHaveTextContent('UI test passes');
  });
});

describe('ChatTextArea approval decisions', () => {
  const awaitingTurn = (): ChatPair => ({
    run_id: 'run_9',
    user: 'Write the file',
    ai: '',
    status: 'awaiting_approval',
    tool_calls: [
      {
        id: 'call_gated',
        name: 'workspace.write_markdown',
        arguments: { path: 'notes.md' },
        status: 'awaiting_approval',
        requires_approval: true,
      },
    ],
  });

  it('offers the four decision tiers and reports the choice', () => {
    const onToolApproval = jest.fn();
    render(<ChatTextArea chatHistory={[awaitingTurn()]} onToolApproval={onToolApproval} />);

    const group = screen.getByRole('group', { name: 'Approve workspace.write_markdown' });
    expect(group).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Allow for session' }));
    expect(onToolApproval).toHaveBeenCalledWith('run_9', 'call_gated', 'session');

    // Buttons disable after a decision to prevent double submission.
    expect(screen.getByRole('button', { name: 'Deny' })).toBeDisabled();
  });

  it('renders no decision buttons without a handler', () => {
    render(<ChatTextArea chatHistory={[awaitingTurn()]} />);
    expect(screen.queryByRole('button', { name: 'Approve once' })).not.toBeInTheDocument();
  });

  it('offers only one-time approval or denial for protected terminal commands', () => {
    const turn = awaitingTurn();
    turn.tool_calls![0] = {
      ...turn.tool_calls![0],
      name: 'terminal.run',
      requires_per_call_approval: true,
    };
    render(<ChatTextArea chatHistory={[turn]} onToolApproval={() => {}} />);

    expect(screen.getByRole('button', { name: 'Approve once' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Deny' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Allow for session' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Always allow' })).not.toBeInTheDocument();
    expect(screen.getByText('Approval applies to this command only.')).toBeInTheDocument();
  });
});
