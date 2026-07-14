import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import Chat, { turnBelongsToChatSelection } from '../Chat';


const mockCancelGeneration = jest.fn();
const mockResetChatSession = jest.fn();
const mockChatSessions: never[] = [];
const mockLoadMore = jest.fn();
const mockNavigate = jest.fn();

jest.mock('../Hooks/useCompleteText', () => ({
  __esModule: true,
  default: () => ({
    completeText: jest.fn(),
    cancelGeneration: mockCancelGeneration,
    resetChatSession: mockResetChatSession,
    loading: true,
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
    processMessage: jest.fn(),
    isProcessing: false,
    error: null,
  }),
}));

jest.mock('../Hooks/useUserSettings', () => ({
  __esModule: true,
  default: () => ({ settings: null }),
}));

jest.mock('react-router-dom', () => ({
  NavLink: ({ to, children, ...props }: any) => <a href={to} {...props}>{children}</a>,
  useNavigate: () => mockNavigate,
  useParams: () => ({}),
}));

jest.mock('../Components/LinkList', () => () => null);
jest.mock('../Components/EnhancedChatInput', () => () => null);

describe('Chat live run controls', () => {
  beforeEach(() => {
    mockCancelGeneration.mockClear();
    mockResetChatSession.mockClear();
  });

  it('renders a Stop control and cancels the active generation', () => {
    render(<Chat />);

    fireEvent.click(screen.getByRole('button', { name: 'Stop generating' }));

    expect(mockCancelGeneration).toHaveBeenCalledTimes(1);
  });

  it('resets the hook session before starting a New Chat', () => {
    render(<Chat />);

    fireEvent.click(screen.getByRole('button', { name: 'New chat' }));

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
