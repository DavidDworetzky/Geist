import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Chat from './Chat';
import { installResizeObserverMock } from './testUtils/mockResizeObserver';

const mockPrepareNewChat = jest.fn();
const mockSetScope = jest.fn(async () => true);
const mockSetChatFolder = jest.fn(async () => true);
const mockRefreshChatSessions = jest.fn();
const mockRefreshFolders = jest.fn();
const mockRetryLocalRuntime = jest.fn();
const mockDownloadLocalArtifact = jest.fn();
const mockRefreshLocalArtifacts = jest.fn();
let mockUserSettings: any = null;
let mockLocalRuntimeStatus: any = null;
let mockLocalArtifact: any = null;
let mockLocalArtifactsError: string | null = null;
let mockCompletedTurn: {
  run_id: string;
  prompt: string;
  message: string;
  origin_chat_id: number | null;
  chat_id: number;
} | null = null;
let mockActiveTurn: any = null;

jest.mock('./Hooks/useGetChatSessions', () => ({
  __esModule: true,
  default: () => ({
    chatSessions: [
      {
        chat_id: 42,
        create_date: '2026-07-09T10:00:00.000Z',
        chat_history: [{ user: 'Plan Geist drawer behavior', ai: 'Done' }],
        folder_id: null,
        memory_enabled: true,
        memory_mode: 'public',
      }
    ],
    loading: false,
    error: null,
    hasMore: false,
    loadMore: jest.fn(),
    refreshChatSessions: mockRefreshChatSessions,
  })
}));

jest.mock('./Hooks/useCompleteText', () => ({
  __esModule: true,
  default: () => ({
    prompt: null,
    completeText: jest.fn(),
    cancelGeneration: jest.fn(),
    resetChatSession: jest.fn(),
    loading: false,
    error: null,
    completedText: null,
    completedTurn: mockCompletedTurn,
    activeTurn: mockActiveTurn,
    state_chat_id: null
  })
}));

jest.mock('./Hooks/useFileContext', () => ({
  __esModule: true,
  default: () => ({
    processMessage: jest.fn(async (message: string) => ({
      originalMessage: message,
      enhancedMessage: message,
      references: [],
      contexts: [],
      hasUnresolvedReferences: false
    })),
    isProcessing: false,
    error: null,
    clearCache: jest.fn()
  })
}));

jest.mock('./Hooks/useUserSettings', () => ({
  __esModule: true,
  default: () => ({ settings: mockUserSettings })
}));

jest.mock('./Hooks/useLocalRuntimeReadiness', () => ({
  __esModule: true,
  default: () => ({
    status: mockLocalRuntimeStatus,
    retry: mockRetryLocalRuntime,
  })
}));

jest.mock('./Hooks/useLocalArtifacts', () => ({
  __esModule: true,
  default: () => ({
    artifacts: mockLocalArtifact ? [mockLocalArtifact] : [],
    loaded: true,
    error: mockLocalArtifactsError,
    refreshLocalArtifacts: mockRefreshLocalArtifacts,
    downloadArtifact: mockDownloadLocalArtifact,
  }),
  isArtifactInstalling: (artifact: { status?: string } | undefined) => Boolean(
    artifact && ['queued', 'downloading', 'cancelling'].includes(artifact.status ?? ''),
  ),
}));

jest.mock('./Hooks/useChatMemory', () => ({
  __esModule: true,
  getMemoryScope: (settings: {
    memory_enabled: boolean;
    memory_mode: 'public' | 'private';
    folder_id: number | null;
  }) => {
    if (!settings.memory_enabled) return { kind: 'disabled' };
    if (settings.folder_id !== null) {
      return { kind: 'folder', folderId: settings.folder_id };
    }
    return { kind: settings.memory_mode };
  },
  default: () => ({
    settings: {
      memory_enabled: true,
      memory_mode: 'public',
      folder_id: null,
      effective_scope: 'public',
      status: 'ready',
    },
    folders: [
      {
        folder_id: 9,
        name: 'Research',
        color: 'violet',
        chat_count: 0,
      },
    ],
    loading: false,
    error: null,
    refreshFolders: mockRefreshFolders,
    createFolder: jest.fn(),
    renameFolder: jest.fn(),
    deleteFolder: jest.fn(),
    setScope: mockSetScope,
    prepareNewChat: mockPrepareNewChat,
    setChatFolder: mockSetChatFolder,
  }),
}));

jest.mock('./Components/ChatTextArea', () => {
  const React = require('react');
  return {
    __esModule: true,
    default: React.forwardRef((_props: unknown, ref: React.Ref<HTMLDivElement>) => <div ref={ref}>Chat transcript</div>)
  };
});

jest.mock('./Components/EnhancedChatInput', () => ({
  __esModule: true,
  default: ({ value, onChange, disabled, placeholder, modelLoading, onShowModelError }: {
    value: string;
    onChange: (value: string) => void;
    disabled?: boolean;
    placeholder?: string;
    modelLoading?: boolean;
    onShowModelError?: () => void;
  }) => (
    <>
    <textarea
      aria-label="Message"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      disabled={disabled}
      placeholder={modelLoading ? '' : placeholder}
    />
    {modelLoading && <div role="status">Loading model…</div>}
    {onShowModelError && <button onClick={onShowModelError}>See more…</button>}
    </>
  )
}));

jest.mock('./Components/MemoryExplorer', () => ({
  __esModule: true,
  default: () => <div>Memory explorer</div>,
}));

describe('Chat history panel', () => {
  beforeEach(() => {
    window.localStorage.clear();
    jest.clearAllMocks();
    mockCompletedTurn = null;
    mockActiveTurn = null;
    mockUserSettings = null;
    mockLocalRuntimeStatus = null;
    mockLocalArtifact = null;
    mockLocalArtifactsError = null;
  });

  it('offers a retry when the local model catalogue cannot be loaded', () => {
    mockUserSettings = {
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3.8-27B',
    };
    mockLocalArtifactsError = 'Local model status failed';

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Models unavailable');
    expect(screen.getByLabelText('Message')).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(mockRefreshLocalArtifacts).toHaveBeenCalledTimes(1);
  });

  it('blocks chat and surfaces local runtime failures before submission', () => {
    mockUserSettings = {
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3.8-27B',
    };
    mockLocalRuntimeStatus = {
      model_id: 'Qwen/Qwen3.8-27B',
      state: 'failed',
      detail: 'Installed model files are missing.',
    };
    mockLocalArtifact = {
      id: 'qwen3.8-27b-4bit-mlx',
      model_id: 'Qwen/Qwen3.8-27B',
      status: 'installed',
      supported: true,
    };

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    expect(screen.getByRole('dialog')).toHaveTextContent('Model failed to load');
    expect(screen.getByRole('dialog')).toHaveTextContent('Installed model files are missing.');
    expect(screen.getByLabelText('Message')).toBeDisabled();
    expect(screen.getByLabelText('Message')).toHaveAttribute('placeholder', 'Model unavailable');
    expect(screen.getByRole('link', { name: 'Choose model' }))
      .toHaveAttribute('href', '/models');
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(mockRetryLocalRuntime).toHaveBeenCalledTimes(1);
  });

  it('opens memory failures once per attempt and reopens from the composer badge', () => {
    mockUserSettings = { default_agent_type: 'local', default_local_model: 'test/model' };
    mockLocalArtifact = { id: 'artifact', model_id: 'test/model', status: 'installed', supported: true };
    mockLocalRuntimeStatus = {
      model_id: 'test/model', state: 'failed', updated_at: 'attempt-1',
      error_code: 'gpu_memory', detail: 'Enable Allow system RAM in Settings > Models.',
      can_offload_to_system_ram: true,
    };
    const tree = <MemoryRouter><Chat /></MemoryRouter>;
    const view = render(tree);
    expect(screen.getByRole('dialog', { name: 'Not enough GPU memory' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Memory settings' })).toHaveAttribute('href', '/settings#models');
    fireEvent.click(screen.getByRole('button', { name: 'Close model error' }));
    view.rerender(tree);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'See more…' }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent(screen.getByRole('dialog'), new Event('cancel', { cancelable: true }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    mockLocalRuntimeStatus = { ...mockLocalRuntimeStatus, updated_at: 'attempt-2' };
    view.rerender(<MemoryRouter><Chat /></MemoryRouter>);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it.each([
    ['gpu_memory', true, 'Not enough GPU memory'],
    ['gpu_memory', false, 'Not enough GPU memory'],
    ['unified_memory', false, 'Not enough memory'],
    ['system_memory', false, 'Not enough system memory'],
  ])('only offers offloading when the runtime permits it (%s, %s)', (code, canOffload, title) => {
    mockUserSettings = { default_agent_type: 'local', default_local_model: 'test/model' };
    mockLocalArtifact = { id: 'artifact', model_id: 'test/model', status: 'installed', supported: true };
    mockLocalRuntimeStatus = {
      model_id: 'test/model', state: 'failed', updated_at: 'attempt-1',
      error_code: code, can_offload_to_system_ram: canOffload,
      detail: canOffload ? 'Enable Allow system RAM in Settings > Models.'
        : 'Close other applications or choose a smaller model.',
    };
    render(<MemoryRouter><Chat /></MemoryRouter>);
    const dialog = screen.getByRole('dialog', { name: title as string });
    expect(within(dialog).queryAllByRole('link', { name: 'Memory settings' }))
      .toHaveLength(Number(canOffload));
    expect(within(dialog).queryAllByText(/Allow system RAM/)).toHaveLength(Number(canOffload));
  });

  it('shows generation memory failures in the modal and clears the attempt on retry', () => {
    mockUserSettings = { default_agent_type: 'local', default_local_model: 'test/model' };
    mockLocalArtifact = { id: 'artifact', model_id: 'test/model', status: 'installed', supported: true };
    mockLocalRuntimeStatus = { model_id: 'test/model', state: 'ready' };
    mockActiveTurn = {
      run_id: 'oom-run', prompt: 'Hello', message: 'Partial reply',
      origin_chat_id: null, chat_id: null, status: 'failed', tool_calls: [], artifacts: [],
      model_load: {
        model_id: 'test/model', state: 'failed', updated_at: 'attempt-1',
        error_code: 'unified_memory', can_offload_to_system_ram: false,
        detail: 'There is not enough shared memory to continue this response.',
      },
    };
    render(<MemoryRouter><Chat /></MemoryRouter>);
    expect(screen.getByRole('dialog', { name: 'Not enough memory' })).toBeInTheDocument();
    expect(screen.getByLabelText('Message')).toBeDisabled();
    expect(screen.queryByRole('status', { name: 'Model loading status' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close model error' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'See more…' }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(mockRetryLocalRuntime).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'See more…' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Message')).not.toBeDisabled();
  });

  it('retires a generation failure when a newer artifact load becomes ready', () => {
    mockUserSettings = { default_agent_type: 'local', default_local_model: 'test/model', default_local_artifact_id: 'large' };
    mockLocalArtifact = { id: 'large', model_id: 'test/model', status: 'installed', supported: true };
    mockLocalRuntimeStatus = { model_id: 'test/model', state: 'ready', updated_at: '2026-09-12T00:00:00Z' };
    mockActiveTurn = {
      run_id: 'oom-run', prompt: 'Hello', message: '', origin_chat_id: null, chat_id: null,
      status: 'failed', tool_calls: [], artifacts: [],
      model_load: {
        model_id: 'test/model', state: 'failed', updated_at: '2026-09-12T00:00:10Z',
        error_code: 'unified_memory', detail: 'Not enough memory',
      },
    };
    const view = render(<MemoryRouter><Chat /></MemoryRouter>);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    mockUserSettings = { ...mockUserSettings, default_local_artifact_id: 'small' };
    mockLocalArtifact = { ...mockLocalArtifact, id: 'small' };
    mockLocalRuntimeStatus = { ...mockLocalRuntimeStatus, state: 'loading', updated_at: '2026-09-12T00:00:20Z' };
    view.rerender(<MemoryRouter><Chat /></MemoryRouter>);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    mockLocalRuntimeStatus = { ...mockLocalRuntimeStatus, state: 'ready', updated_at: '2026-09-12T00:00:30Z' };
    view.rerender(<MemoryRouter><Chat /></MemoryRouter>);
    expect(screen.getByLabelText('Message')).not.toBeDisabled();
    expect(screen.queryByRole('button', { name: 'See more…' })).not.toBeInTheDocument();
  });

  it.each(['different-model', 'different-chat'])('ignores a generation failure from a %s', (scope) => {
    mockUserSettings = { default_agent_type: 'local', default_local_model: 'test/model' };
    mockLocalArtifact = { id: 'artifact', model_id: 'test/model', status: 'installed', supported: true };
    mockLocalRuntimeStatus = { model_id: 'test/model', state: 'ready' };
    mockActiveTurn = {
      run_id: 'oom-run', prompt: 'Hello', message: '',
      origin_chat_id: scope === 'different-chat' ? 99 : null,
      chat_id: scope === 'different-chat' ? 99 : null,
      status: 'failed', tool_calls: [], artifacts: [],
      model_load: {
        model_id: scope === 'different-model' ? 'previous/model' : 'test/model',
        state: 'failed', updated_at: 'attempt-1', error_code: 'unified_memory',
        detail: 'Not enough memory',
      },
    };
    render(<MemoryRouter><Chat /></MemoryRouter>);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Message')).not.toBeDisabled();
  });

  it('uses concise language when the selected model is not installed', () => {
    mockUserSettings = {
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3.8-27B',
      default_local_artifact_id: 'qwen3.8-27b-4bit-mlx',
    };
    mockLocalArtifact = {
      id: 'qwen3.8-27b-4bit-mlx',
      model_id: 'Qwen/Qwen3.8-27B',
      status: 'not_installed',
      supported: true,
    };

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    expect(screen.getByRole('status')).toHaveTextContent('Model not installed');
    expect(screen.getByLabelText('Message')).toHaveAttribute('placeholder', 'Model not installed');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('does not add a verbose chat notice while the install chip is active', () => {
    mockUserSettings = {
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3.8-27B',
      default_local_artifact_id: 'qwen3.8-27b-4bit-mlx',
    };
    mockLocalArtifact = {
      id: 'qwen3.8-27b-4bit-mlx',
      model_id: 'Qwen/Qwen3.8-27B',
      status: 'downloading',
      supported: true,
    };

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    expect(screen.getByLabelText('Message')).toHaveAttribute('placeholder', 'Installing model…');
    expect(screen.queryByText(/local model is not ready/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('keeps cancellation quiet while the selector chip finishes it', () => {
    mockUserSettings = {
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3.8-27B',
      default_local_artifact_id: 'qwen3.8-27b-4bit-mlx',
    };
    mockLocalArtifact = {
      id: 'qwen3.8-27b-4bit-mlx',
      model_id: 'Qwen/Qwen3.8-27B',
      status: 'cancelling',
      supported: true,
    };

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    expect(screen.getByLabelText('Message')).toHaveAttribute('placeholder', 'Cancelling…');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows one actionable install error', () => {
    mockUserSettings = {
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3.8-27B',
      default_local_artifact_id: 'qwen3.8-27b-4bit-mlx',
    };
    mockLocalArtifact = {
      id: 'qwen3.8-27b-4bit-mlx',
      model_id: 'Qwen/Qwen3.8-27B',
      status: 'failed',
      error: 'Not enough space to finish installing this model.',
      supported: true,
    };

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Install failed');
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Not enough space to finish installing this model.',
    );
    expect(screen.getByLabelText('Message')).toHaveAttribute('placeholder', 'Install failed');
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(mockDownloadLocalArtifact).toHaveBeenCalledTimes(1);
  });

  it('shows loading immediately while an installed model starts', () => {
    mockUserSettings = {
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3.8-27B',
      default_local_artifact_id: 'qwen3.8-27b-4bit-mlx',
    };
    mockLocalArtifact = {
      id: 'qwen3.8-27b-4bit-mlx',
      model_id: 'Qwen/Qwen3.8-27B',
      status: 'installed',
      supported: true,
    };

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    expect(screen.getByText('Loading model…')).toBeInTheDocument();
    expect(screen.getByLabelText('Message')).toHaveAttribute('placeholder', '');
  });

  it('only reserves the external scrollbar rail while the transcript overflows', () => {
    const resizeObserver = installResizeObserverMock();
    const renderResult = render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    try {
      const { container } = renderResult;
      const chat = container.querySelector<HTMLDivElement>('.ChatContainer');
      const transcript = container.querySelector<HTMLDivElement>('.chat-history-scroll');
      expect(chat).not.toBeNull();
      expect(transcript).not.toBeNull();
      expect(chat).not.toHaveClass('chat-scrollbar-visible');

      let scrollHeight = 700;
      Object.defineProperty(transcript, 'clientHeight', {
        configurable: true,
        get: () => 500,
      });
      Object.defineProperty(transcript, 'scrollHeight', {
        configurable: true,
        get: () => scrollHeight,
      });

      act(() => resizeObserver.trigger(transcript as HTMLDivElement));
      expect(chat).toHaveClass('chat-scrollbar-visible');

      scrollHeight = 500;
      act(() => resizeObserver.trigger(transcript as HTMLDivElement));
      expect(chat).not.toHaveClass('chat-scrollbar-visible');
    } finally {
      renderResult.unmount();
      resizeObserver.restore();
    }
  });

  it('only adds an external scrollbar gap while the chat drawer overflows', () => {
    const resizeObserver = installResizeObserverMock();
    const renderResult = render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    try {
      const drawer = screen.getByRole('complementary', { name: 'Chat sessions' });
      const drawerScroll = drawer.querySelector<HTMLDivElement>('.stage-panel-surface');
      expect(drawerScroll).not.toBeNull();
      expect(drawer).not.toHaveClass('stage-panel-scrollbar-visible');

      fireEvent.click(within(drawer).getByRole('button', { name: 'Expand chat history' }));

      let scrollHeight = 700;
      Object.defineProperty(drawerScroll, 'clientHeight', {
        configurable: true,
        get: () => 500,
      });
      Object.defineProperty(drawerScroll, 'scrollHeight', {
        configurable: true,
        get: () => scrollHeight,
      });

      act(() => resizeObserver.trigger(drawerScroll as HTMLDivElement));
      expect(drawer).toHaveClass('stage-panel-scrollbar-visible');

      scrollHeight = 500;
      act(() => resizeObserver.trigger(drawerScroll as HTMLDivElement));
      expect(drawer).not.toHaveClass('stage-panel-scrollbar-visible');
    } finally {
      renderResult.unmount();
      resizeObserver.restore();
    }
  });

  it('morphs from compact chat controls into the chat-sized history panel', () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    const drawer = screen.getByRole('complementary', { name: 'Chat sessions' });
    const composer = screen.getByRole('textbox', { name: 'Message' });
    expect(drawer).toHaveAttribute('data-state', 'minimized');
    expect(drawer).toHaveClass('stage-panel-minimized');
    expect(composer).toBeInTheDocument();
    const newChatButton = within(drawer).getByRole('button', { name: 'New chat' });
    expect(newChatButton).toBeInTheDocument();
    expect(newChatButton).toHaveTextContent('New Chat');
    expect(within(drawer).getByRole('button', { name: 'Expand chat history' })).toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Search chats' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Chats' })).not.toBeInTheDocument();

    fireEvent.click(within(drawer).getByRole('button', { name: 'Expand chat history' }));

    expect(drawer).toHaveAttribute('data-state', 'expanded');
    expect(drawer).toHaveClass('stage-panel-expanded');
    expect(screen.queryByRole('textbox', { name: 'Message' })).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Chats' })).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search chats')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Research/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'New private folder' })).toBeInTheDocument();
    expect(screen.getByText('Recent chats')).toBeInTheDocument();
    expect(screen.getByText(/Plan Geist drawer behavior/i)).toBeInTheDocument();
    expect(window.localStorage.getItem('geist.chatDrawerState')).toBe('expanded');

    fireEvent.click(screen.getByRole('button', { name: 'Close chat history' }));

    expect(drawer).toHaveAttribute('data-state', 'minimized');
    expect(drawer).toHaveClass('stage-panel-minimized');
    expect(screen.getByRole('textbox', { name: 'Message' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Chats' })).not.toBeInTheDocument();
    expect(window.localStorage.getItem('geist.chatDrawerState')).toBe('minimized');
    expect(screen.getByRole('button', { name: /^Memory settings:/ })).toBeInTheDocument();
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
  });

  it('edits a chat name and persists it locally', () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Expand chat history' }));
    fireEvent.click(screen.getByRole('button', {
      name: 'Chat options for Plan Geist drawer behavior',
    }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));

    const titleInput = screen.getByRole('textbox', { name: 'Chat name' });
    fireEvent.change(titleInput, { target: { value: 'Drawer polish' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save chat name' }));

    expect(screen.getByText('Drawer polish')).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem('geist.chatTitles') || '{}')).toEqual({ '42': 'Drawer polish' });
  });

  it('inherits the selected folder when starting a new chat', () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Expand chat history' }));
    fireEvent.click(screen.getByRole('button', { name: /Research/ }));
    fireEvent.click(screen.getByRole('button', { name: 'New Chat' }));

    expect(mockPrepareNewChat).toHaveBeenCalledWith({
      kind: 'folder',
      folderId: 9,
    });
  });

  it('omits folder management from the memory menu when the drawer is already open', () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Expand chat history' }));
    fireEvent.click(screen.getByRole('button', { name: /^Memory settings:/ }));

    expect(screen.queryByRole('menuitem', { name: /Manage folders/ })).not.toBeInTheDocument();
  });

  it('refreshes chat sessions and folder counts when a turn completes', async () => {
    mockCompletedTurn = {
      run_id: 'run-42',
      prompt: 'Remember the launch date',
      message: 'Saved',
      origin_chat_id: null,
      chat_id: 42,
    };

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(mockRefreshChatSessions).toHaveBeenCalledTimes(1);
    });
    expect(mockRefreshFolders).toHaveBeenCalledTimes(1);
  });

  it('moves a chat into a folder from its row menu after confirmation', async () => {
    jest.spyOn(window, 'confirm').mockReturnValue(true);
    mockSetChatFolder.mockResolvedValueOnce(true);
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Expand chat history' }));
    fireEvent.click(screen.getByRole('button', {
      name: 'Chat options for Plan Geist drawer behavior',
    }));
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Research' }));

    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining(
      "existing summary and future saved memory will be available",
    ));
    await waitFor(() => {
      expect(mockSetChatFolder).toHaveBeenCalledWith(42, 9);
    });
    await waitFor(() => {
      expect(mockRefreshChatSessions).toHaveBeenCalledTimes(1);
    });
  });

  it('supports keyboard navigation and focus restoration in a chat row menu', () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Expand chat history' }));
    const trigger = screen.getByRole('button', {
      name: 'Chat options for Plan Geist drawer behavior',
    });
    fireEvent.click(trigger);

    const renameItem = screen.getByRole('menuitem', { name: 'Rename' });
    expect(renameItem).toHaveFocus();
    fireEvent.keyDown(renameItem, { key: 'ArrowDown' });
    expect(screen.getByRole('menuitemradio', { name: 'No folder' })).toHaveFocus();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('menu', {
      name: 'Plan Geist drawer behavior options',
    })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });
});
