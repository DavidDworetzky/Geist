import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Chat from './Chat';

jest.mock('./Hooks/useGetChatSessions', () => ({
  __esModule: true,
  default: () => ({
    chatSessions: [
      {
        chat_id: 42,
        create_date: '2026-07-09T10:00:00.000Z',
        chat_history: [{ user: 'Plan Geist drawer behavior', ai: 'Done' }]
      }
    ],
    loading: false,
    error: null,
    hasMore: false,
    loadMore: jest.fn(),
    refreshChatSessions: jest.fn()
  })
}));

jest.mock('./Hooks/useCompleteText', () => ({
  __esModule: true,
  default: () => ({
    prompt: null,
    completeText: jest.fn(),
    loading: false,
    error: null,
    completedText: null,
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

describe('Chat history panel', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('morphs from compact chat controls into the chat-sized history panel', () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    const drawer = screen.getByRole('complementary', { name: 'Chat sessions' });
    const stage = drawer.closest('.chat-stage') as HTMLElement;
    const composerDock = screen.getByRole('textbox', { name: 'Message' }).closest('.chat-composer-dock') as HTMLElement;
    expect(drawer).toHaveAttribute('data-state', 'minimized');
    expect(drawer).toHaveClass('stage-panel-minimized');
    expect(stage).not.toBeNull();
    expect(stage).toContainElement(composerDock);
    expect(composerDock).toHaveAttribute('aria-hidden', 'false');
    const newChatButton = within(drawer).getByRole('button', { name: 'New chat' });
    expect(newChatButton).toBeInTheDocument();
    expect(newChatButton).toHaveTextContent('New Chat');
    expect(within(drawer).getByRole('button', { name: 'Expand chat history' })).toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Search chats' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Chats' })).not.toBeInTheDocument();

    fireEvent.click(within(drawer).getByRole('button', { name: 'Expand chat history' }));

    expect(drawer).toHaveAttribute('data-state', 'expanded');
    expect(drawer).toHaveClass('stage-panel-expanded');
    expect(composerDock).toHaveAttribute('aria-hidden', 'true');
    expect(drawer.querySelector('.stage-panel-surface')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Chats' })).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search chats')).toBeInTheDocument();
    expect(screen.getByText('Recent chats')).toBeInTheDocument();
    expect(screen.getByText(/Plan Geist drawer behavior/i)).toBeInTheDocument();
    expect(window.localStorage.getItem('geist.chatDrawerState')).toBe('expanded');

    fireEvent.click(screen.getByRole('button', { name: 'Close chat history' }));

    expect(drawer).toHaveAttribute('data-state', 'minimized');
    expect(drawer).toHaveClass('stage-panel-minimized');
    expect(composerDock).toHaveAttribute('aria-hidden', 'false');
    expect(screen.queryByRole('heading', { name: 'Chats' })).not.toBeInTheDocument();
    expect(window.localStorage.getItem('geist.chatDrawerState')).toBe('minimized');
  });

  it('edits a chat name and persists it locally', () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Chat />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Expand chat history' }));
    fireEvent.click(screen.getByRole('button', { name: 'Rename Plan Geist drawer behavior' }));

    const titleInput = screen.getByRole('textbox', { name: 'Chat name' });
    fireEvent.change(titleInput, { target: { value: 'Drawer polish' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save chat name' }));

    expect(screen.getByText('Drawer polish')).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem('geist.chatTitles') || '{}')).toEqual({ '42': 'Drawer polish' });
  });
});
