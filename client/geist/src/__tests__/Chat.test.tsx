import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import Chat, { turnBelongsToChatSelection } from '../Chat';


const mockCancelGeneration = jest.fn();
const mockSteerRun = jest.fn(async (_text: string) => true);
const mockCompleteText = jest.fn();
let mockLoading = true;
const mockResetChatSession = jest.fn();
const mockPrepareNewChat = jest.fn();
const mockChatSessions: never[] = [];
const mockLoadMore = jest.fn();
const mockNavigate = jest.fn();

jest.mock('../Hooks/useCompleteText', () => ({
  __esModule: true,
  default: () => ({
    completeText: mockCompleteText,
    steerRun: mockSteerRun,
    isSteering: false,
    cancelGeneration: mockCancelGeneration,
    resetChatSession: mockResetChatSession,
    loading: mockLoading,
    error: null,
    completedTurn: null,
    activeTurn: {
      run_id: 'run_1',
      prompt: 'Use a tool',
      message: 'Working',
      chat_id: null,
      origin_chat_id: null,
      status: 'streaming',
      tool_calls: [],
      artifacts: [],
    },
    state_chat_id: null,
  }),
}));

jest.mock('../Hooks/useGetChatSessions', () => ({
  __esModule: true,
  default: () => ({
    chatSessions: mockChatSessions,
    loading: false,
    error: null,
    loadMore: mockLoadMore,
    hasMore: false,
  }),
}));

jest.mock('../Hooks/useFileContext', () => ({
  __esModule: true,
  default: () => ({
    processMessage: async (message: string) => ({
      enhancedMessage: message, references: [], contexts: [], hasUnresolvedReferences: false,
    }),
    isProcessing: false,
    error: null,
  }),
}));

jest.mock('../Hooks/useUserSettings', () => ({
  __esModule: true,
  default: () => ({ settings: null }),
}));

jest.mock('../Hooks/useChatMemory', () => ({
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
    folders: [],
    loading: false,
    error: null,
    refreshFolders: jest.fn(),
    createFolder: jest.fn(),
    renameFolder: jest.fn(),
    deleteFolder: jest.fn(),
    setScope: jest.fn(async () => true),
    prepareNewChat: mockPrepareNewChat,
    setChatFolder: jest.fn(async () => true),
  }),
}));

jest.mock('react-router-dom', () => ({
  NavLink: ({ to, children, ...props }: any) => <a href={to} {...props}>{children}</a>,
  useNavigate: () => mockNavigate,
  useParams: () => ({}),
}));

jest.mock('../Components/LinkList', () => () => null);
jest.mock('../Components/EnhancedChatInput', () => ({
  __esModule: true,
  default: ({ value, onChange, onSubmit, disabled, submitLabel }: {
    value: string;
    onChange: (value: string) => void;
    onSubmit: (value: string) => void;
    disabled: boolean;
    submitLabel: string;
  }) => (
    <div>
      <textarea aria-label="Message" value={value} onChange={e => onChange(e.target.value)} disabled={disabled} />
      <button disabled={disabled} onClick={() => onSubmit(value)}>{submitLabel}</button>
    </div>
  ),
}));

describe('Chat live run controls', () => {
  beforeEach(() => {
    mockCancelGeneration.mockClear();
    mockSteerRun.mockResolvedValue(true);
    mockLoading = true;
    mockResetChatSession.mockClear();
    mockPrepareNewChat.mockClear();
  });

  it('renders a Stop control and cancels the active generation', () => {
    render(<Chat />);

    fireEvent.click(screen.getByRole('button', { name: 'Stop generating' }));

    expect(mockCancelGeneration).toHaveBeenCalledTimes(1);
  });

  it('keeps the composer enabled and sends steering without cancelling', async () => {
    render(<Chat />);
    const input = screen.getByRole('textbox', { name: 'Message' });
    expect(input).toBeEnabled();
    fireEvent.change(input, { target: { value: 'Use local only' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add instructions' }));
    await waitFor(() => expect(mockSteerRun).toHaveBeenCalledWith('Use local only'));
    await waitFor(() => expect(input).toHaveValue(''));
    expect(mockCancelGeneration).not.toHaveBeenCalled();
  });

  it('does not clear an instruction draft when the original stream finishes', async () => {
    let finish: () => void = () => {};
    mockCompleteText.mockImplementation(() => new Promise<void>(resolve => { finish = resolve; }));
    mockLoading = false;
    const view = render(<Chat />);
    const input = screen.getByRole('textbox', { name: 'Message' });
    fireEvent.change(input, { target: { value: 'Build voice notes' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));
    expect(input).toHaveValue('');
    await waitFor(() => expect(mockCompleteText).toHaveBeenCalled());
    mockLoading = true;
    view.rerender(<Chat />);
    fireEvent.change(input, { target: { value: 'Also add tests' } });
    await act(async () => { finish(); });
    expect(input).toHaveValue('Also add tests');
  });

  it('resets the hook session before starting a New Chat', () => {
    render(<Chat />);

    fireEvent.click(screen.getByRole('button', { name: 'New chat' }));

    expect(mockPrepareNewChat).toHaveBeenCalledWith({ kind: 'public' });
    expect(mockResetChatSession).toHaveBeenCalledTimes(1);
    expect(mockNavigate).toHaveBeenCalledWith('/chat');
  });

  it('does not associate a run with a different selected chat', () => {
    const oldRun = { origin_chat_id: 41, chat_id: 41 };

    expect(turnBelongsToChatSelection(oldRun, 42, null)).toBe(false);
    expect(turnBelongsToChatSelection(oldRun, null, 42)).toBe(false);
    expect(turnBelongsToChatSelection(oldRun, 41, null)).toBe(true);
  });
});
