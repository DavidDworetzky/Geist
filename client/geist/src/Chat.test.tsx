import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Chat from './Chat';

const mockPrepareNewChat = jest.fn();
const mockSetScope = jest.fn(async () => true);
const mockSetChatFolder = jest.fn(async () => true);
const mockRefreshChatSessions = jest.fn();
const mockRefreshFolders = jest.fn();
let mockCompletedTurn: {
  run_id: string;
  prompt: string;
  message: string;
  origin_chat_id: number | null;
  chat_id: number;
} | null = null;

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
    activeTurn: null,
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
  default: () => ({ settings: null })
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
  default: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <textarea aria-label="Message" value={value} onChange={(event) => onChange(event.target.value)} />
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
  });

  it('only reserves the external scrollbar rail while the transcript overflows', () => {
    let resizeObserverCallback: ResizeObserverCallback | null = null;
    const originalResizeObserver = window.ResizeObserver;
    class MockResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        resizeObserverCallback = callback;
      }

      observe() {}
      unobserve() {}
      disconnect() {}
    }
    Object.defineProperty(window, 'ResizeObserver', {
      configurable: true,
      writable: true,
      value: MockResizeObserver,
    });

    try {
      const { container } = render(
        <MemoryRouter initialEntries={['/chat']}>
          <Chat />
        </MemoryRouter>
      );
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

      act(() => resizeObserverCallback?.([], {} as ResizeObserver));
      expect(chat).toHaveClass('chat-scrollbar-visible');

      scrollHeight = 500;
      act(() => resizeObserverCallback?.([], {} as ResizeObserver));
      expect(chat).not.toHaveClass('chat-scrollbar-visible');
    } finally {
      Object.defineProperty(window, 'ResizeObserver', {
        configurable: true,
        writable: true,
        value: originalResizeObserver,
      });
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
