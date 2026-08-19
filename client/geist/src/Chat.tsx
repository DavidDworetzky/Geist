import React, { useState, useEffect, useRef, useLayoutEffect, useCallback, useMemo } from 'react';
import useCompleteText from './Hooks/useCompleteText';
import useGetChatSessions from './Hooks/useGetChatSessions';
import useFileContext from './Hooks/useFileContext';
import useUserSettings from './Hooks/useUserSettings';
import useChatMemory, { MemoryScope } from './Hooks/useChatMemory';
import useOverflowObserver from './Hooks/useOverflowObserver';
import ChatTextArea from './Components/ChatTextArea';
import EnhancedChatInput from './Components/EnhancedChatInput';
import ChatMemoryControls from './Components/ChatMemoryControls';
import MemoryExplorer from './Components/MemoryExplorer';
import StagePanelIcon from './Components/StagePanelIcon';
import { ChatPair, ChatHistory, ChatTurnResult, ToolApprovalDecision } from './chatTypes';
import { NavLink, useNavigate, useParams } from 'react-router-dom';

const PAGE_SIZE = 20;
const CHAT_DRAWER_STORAGE_KEY = 'geist.chatDrawerState';
const CHAT_TITLES_STORAGE_KEY = 'geist.chatTitles';

type ChatDrawerState = 'minimized' | 'expanded';
type ChatTitles = Record<string, string>;

interface ChatSessionListItem {
  id: number;
  name: string;
  link: string;
  date: Date;
  folderId: number | null;
  memoryEnabled: boolean;
  memoryMode: 'public' | 'private';
}

export const turnBelongsToChatSelection = (
  turn: Pick<ChatTurnResult, 'origin_chat_id' | 'chat_id'>,
  routeChatId: number | null,
  stateChatId: number | null,
): boolean => {
  if (routeChatId !== null) {
    return turn.origin_chat_id === routeChatId || turn.chat_id === routeChatId;
  }
  if (stateChatId !== null) {
    return turn.origin_chat_id === stateChatId || turn.chat_id === stateChatId;
  }
  return turn.origin_chat_id === null;
};

function getInitialDrawerState(): ChatDrawerState {
  if (typeof window === 'undefined') {
    return 'minimized';
  }

  return window.localStorage.getItem(CHAT_DRAWER_STORAGE_KEY) === 'expanded' ? 'expanded' : 'minimized';
}
function getInitialChatTitles(): ChatTitles {
  if (typeof window === 'undefined') {
    return {};
  }

  try {
    const savedTitles = JSON.parse(window.localStorage.getItem(CHAT_TITLES_STORAGE_KEY) || '{}');
    return savedTitles && typeof savedTitles === 'object' ? savedTitles : {};
  } catch {
    return {};
  }
}

const Chat = () => {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const [chatHistory, setChatHistory] = useState<ChatHistory>();
  const [chatDrawerState, setChatDrawerState] = useState<ChatDrawerState>(getInitialDrawerState);
  const [chatSearch, setChatSearch] = useState('');
  const [chatTitles, setChatTitles] = useState<ChatTitles>(getInitialChatTitles);
  const [editingChatId, setEditingChatId] = useState<number | null>(null);
  const [chatTitleDraft, setChatTitleDraft] = useState('');
  const [chatTitleError, setChatTitleError] = useState('');
  const [folderFilter, setFolderFilter] = useState<number | 'all'>('all');
  const [isCreatingFolder, setIsCreatingFolder] = useState(false);
  const [folderNameDraft, setFolderNameDraft] = useState('');
  const [folderCreateError, setFolderCreateError] = useState('');
  const [editingFolderId, setEditingFolderId] = useState<number | null>(null);
  const [folderRenameDraft, setFolderRenameDraft] = useState('');
  const [confirmDeleteFolderId, setConfirmDeleteFolderId] = useState<number | null>(null);
  const [openChatMenuId, setOpenChatMenuId] = useState<number | null>(null);
  const [movingChatId, setMovingChatId] = useState<number | null>(null);
  const chatMenuHostRef = useRef<HTMLDivElement>(null);
  const chatMenuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const { chatSessions, loading: isChatSessionLoading, error: chatSessionError, loadMore: loadMoreSessions, hasMore: hasMoreSessions, refreshChatSessions } = useGetChatSessions();
  const [userInput, setUserInput] = useState('');
  const [fileContextInfo, setFileContextInfo] = useState<string>('');
  const { settings: userSettings } = useUserSettings();
  const {
    completeText,
    cancelGeneration,
    resetChatSession,
    loading: isLoading,
    error,
    completedTurn,
    activeTurn,
    state_chat_id,
  } = useCompleteText(userSettings);
  const { processMessage, isProcessing: isProcessingFiles, error: fileError } = useFileContext();
  const routeChatId = chatId ? parseInt(chatId, 10) : null;
  const selectedChatId = routeChatId ?? state_chat_id;
  const {
    settings: memorySettings,
    folders,
    loading: isMemoryLoading,
    error: memoryError,
    refreshFolders,
    createFolder,
    renameFolder,
    deleteFolder,
    setScope,
    prepareNewChat,
    setChatFolder,
  } = useChatMemory(selectedChatId);

  useEffect(() => {
    if (openChatMenuId === null) return undefined;

    const firstItem = chatMenuHostRef.current?.querySelector<HTMLButtonElement>(
      '[role="menu"] button:not(:disabled)',
    );
    firstItem?.focus();

    const handleOutsideClick = (event: MouseEvent) => {
      if (!chatMenuHostRef.current?.contains(event.target as Node)) {
        setOpenChatMenuId(null);
      }
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setOpenChatMenuId(null);
        chatMenuTriggerRef.current?.focus();
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [openChatMenuId]);

  useEffect(() => {
    if (!chatId && state_chat_id !== null) {
      navigate(`/chat/${state_chat_id}`, { replace: true });
    }
  }, [chatId, navigate, state_chat_id]);

  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const {
    ref: chatContainerRef,
    hasOverflow: hasChatScrollbar,
  } = useOverflowObserver<HTMLDivElement>();
  const {
    ref: chatDrawerScrollRef,
    hasOverflow: hasChatDrawerScrollbar,
  } = useOverflowObserver<HTMLDivElement>(chatDrawerState === 'expanded');
  const prevScrollHeightRef = useRef<number>(0);
  const shouldRestoreScrollRef = useRef(false);

  const fetchHistory = useCallback(async (pageNum: number, currentChatId: string) => {
    if (!currentChatId) return;
    setIsLoadingHistory(true);
    if (pageNum > 1) shouldRestoreScrollRef.current = true;

    try {
      const res = await fetch(`/agent/chat_history/${currentChatId}/paginated?page=${pageNum}&page_size=${PAGE_SIZE}`);
      if (!res.ok) throw new Error('Failed to fetch history');

      const data = await res.json();
      const newMessages = data.chat_history.map((h: any) => ({
        run_id: h.run_id,
        user: h.user,
        ai: h.ai,
        status: h.status,
        tool_calls: h.tool_calls,
        artifacts: h.artifacts,
      }));

      setChatHistory(prev => {
        if (pageNum === 1) {
          return { chatHistory: newMessages };
        }
        return {
          chatHistory: [...newMessages, ...(prev?.chatHistory || [])]
        };
      });

      const total = data.total_messages || 0;
      setHasMore(!(newMessages.length < PAGE_SIZE || (pageNum * PAGE_SIZE >= total)));
    } catch (err) {
      console.error('Error fetching chat history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    if (chatId) {
      setPage(1);
      setHasMore(true);
      fetchHistory(1, chatId);
    } else {
      setChatHistory(undefined);
    }
  }, [chatId, fetchHistory]);

  useLayoutEffect(() => {
    if (shouldRestoreScrollRef.current && chatContainerRef.current && prevScrollHeightRef.current) {
      const newScrollHeight = chatContainerRef.current.scrollHeight;
      const diff = newScrollHeight - prevScrollHeightRef.current;
      chatContainerRef.current.scrollTop = diff;
      shouldRestoreScrollRef.current = false;
    } else if (page === 1 && chatContainerRef.current && !shouldRestoreScrollRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatContainerRef, chatHistory, page, isLoading]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (container && chatId && hasMore && !isLoadingHistory && (chatHistory?.chatHistory?.length ?? 0) > 0) {
      if (container.scrollHeight <= container.clientHeight) {
        const nextPage = page + 1;
        setPage(nextPage);
        if (chatId) {
          fetchHistory(nextPage, chatId);
        }
      }
    }
  }, [chatContainerRef, chatHistory, hasMore, isLoadingHistory, page, chatId, fetchHistory]);

  const observer = useRef<IntersectionObserver>();
  const lastItemRef = useCallback(
    (node: any) => {
      if (isChatSessionLoading || chatSessionError || typeof IntersectionObserver === 'undefined') return;
      if (observer.current) observer.current.disconnect();
      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasMoreSessions) {
          loadMoreSessions();
        }
      });
      if (node) observer.current.observe(node);
    },
    [isChatSessionLoading, hasMoreSessions, loadMoreSessions, chatSessionError]
  );

  const loadedHistoryLength = chatHistory?.chatHistory?.length ?? 0;
  const handleScroll = useCallback(() => {
    if (chatContainerRef.current && chatId && loadedHistoryLength > 0) {
      const { scrollTop, scrollHeight } = chatContainerRef.current;
      if (scrollTop <= 10 && hasMore && !isLoadingHistory) {
        prevScrollHeightRef.current = scrollHeight;
        const nextPage = page + 1;
        setPage(nextPage);
        if (chatId) {
          fetchHistory(nextPage, chatId);
        }
      }
    }
  }, [chatContainerRef, hasMore, isLoadingHistory, page, chatId, loadedHistoryLength, fetchHistory]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [chatContainerRef, handleScroll]);

  const chatWithServer = async (input: string) => {
    const parsedChatId = routeChatId ?? state_chat_id;
    try {
      const processedMessage = await processMessage(input);

      if (processedMessage.contexts.length > 0) {
        const contextInfo = `Files referenced: ${processedMessage.contexts.map(c => c.filename).join(', ')}`;
        setFileContextInfo(contextInfo);
      } else {
        setFileContextInfo('');
      }

      const messageToSend = processedMessage.enhancedMessage || input;
      await completeText(messageToSend, parsedChatId, {
        memory_enabled: memorySettings.memory_enabled,
        memory_mode: memorySettings.memory_mode,
        folder_id: memorySettings.folder_id,
      });
    } catch (err) {
      console.error('Error chatting with server:', err);
    }
  };

  const chatSessionLinks = useMemo<ChatSessionListItem[]>(() => {
    if (!chatSessions) {
      return [];
    }

    return chatSessions
      .map((session) => {
        const date = new Date(session.create_date);
        const firstLine = session.chat_history[0]?.user?.split('\n')[0]?.trim() || 'New Chat';
        const defaultName = firstLine.split(/\s+/).slice(0, 6).join(' ');
        return {
          id: session.chat_id,
          name: chatTitles[String(session.chat_id)] || defaultName,
          link: `/chat/${session.chat_id}`,
          date,
          folderId: session.folder_id,
          memoryEnabled: session.memory_enabled,
          memoryMode: session.memory_mode,
        };
      })
      .sort((a, b) => b.date.getTime() - a.date.getTime());
  }, [chatSessions, chatTitles]);
  useEffect(() => {
    if (!completedTurn) return;
    const belongsToCurrentChat = turnBelongsToChatSelection(
      completedTurn,
      routeChatId,
      state_chat_id,
    );
    if (!belongsToCurrentChat) return;

    const newHistory: ChatPair = {
      run_id: completedTurn.run_id,
      user: completedTurn.prompt,
      ai: completedTurn.message,
      tool_calls: completedTurn.tool_calls,
      artifacts: completedTurn.artifacts,
    };
    setChatHistory(previous => {
      const existingHistory = previous?.chatHistory ?? [];
      if (
        completedTurn.run_id &&
        existingHistory.some((turn) => turn.run_id === completedTurn.run_id)
      ) {
        return previous;
      }
      return { chatHistory: [...existingHistory, newHistory] };
    });
    void Promise.all([
      refreshChatSessions(),
      refreshFolders(),
    ]);
  }, [completedTurn, refreshChatSessions, refreshFolders, routeChatId, state_chat_id]);

  const handleSubmit = async (message: string) => {
    if (message.trim() && !isMemoryLoading) {
      await chatWithServer(message);
      setUserInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (userInput.trim() && !isLoading && !isMemoryLoading) {
        void handleSubmit(userInput);
      }
    }
  };

  const setDrawerState = (nextState: ChatDrawerState) => {
    setChatDrawerState(nextState);
    window.localStorage.setItem(CHAT_DRAWER_STORAGE_KEY, nextState);
    if (nextState === 'minimized') {
      setEditingChatId(null);
      setChatTitleDraft('');
      setChatTitleError('');
      setOpenChatMenuId(null);
    }
  };

  const handleNewChat = () => {
    const newChatScope: MemoryScope = folderFilter === 'all'
      ? { kind: 'public' }
      : { kind: 'folder', folderId: folderFilter };
    prepareNewChat(newChatScope);
    resetChatSession();
    setDrawerState('minimized');
    setChatHistory(undefined);
    setUserInput('');
    setFileContextInfo('');
    navigate('/chat');
  };
  const startEditingChatTitle = (session: ChatSessionListItem) => {
    setOpenChatMenuId(null);
    setEditingChatId(session.id);
    setChatTitleDraft(session.name);
    setChatTitleError('');
  };

  const cancelEditingChatTitle = () => {
    setEditingChatId(null);
    setChatTitleDraft('');
    setChatTitleError('');
  };

  const saveChatTitle = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (editingChatId === null) return;

    const normalizedTitle = chatTitleDraft.trim();
    if (!normalizedTitle) {
      setChatTitleError('Enter a name for this chat.');
      return;
    }

    const nextTitles = { ...chatTitles, [String(editingChatId)]: normalizedTitle };
    try {
      window.localStorage.setItem(CHAT_TITLES_STORAGE_KEY, JSON.stringify(nextTitles));
      setChatTitles(nextTitles);
      cancelEditingChatTitle();
    } catch {
      setChatTitleError('Unable to save this chat name locally.');
    }
  };

  const saveFolder = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = folderNameDraft.trim();
    if (!name) {
      setFolderCreateError('Enter a folder name.');
      return;
    }
    try {
      const folder = await createFolder(name);
      setFolderNameDraft('');
      setFolderCreateError('');
      setIsCreatingFolder(false);
      setFolderFilter(folder.folder_id);
    } catch (folderError) {
      setFolderCreateError(
        folderError instanceof Error ? folderError.message : 'Could not create folder.',
      );
    }
  };

  const saveFolderRename = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (editingFolderId === null || !folderRenameDraft.trim()) return;
    try {
      await renameFolder(editingFolderId, folderRenameDraft.trim());
      setEditingFolderId(null);
      setFolderRenameDraft('');
      setFolderCreateError('');
    } catch (folderError) {
      setFolderCreateError(
        folderError instanceof Error ? folderError.message : 'Could not rename folder.',
      );
    }
  };

  const removeFolder = async (folderId: number) => {
    try {
      await deleteFolder(folderId);
      await refreshChatSessions();
      setFolderFilter('all');
      setConfirmDeleteFolderId(null);
      setFolderCreateError('');
    } catch (folderError) {
      setFolderCreateError(
        folderError instanceof Error ? folderError.message : 'Could not delete folder.',
      );
    }
  };

  const changeMemoryScope = async (scope: MemoryScope) => {
    const updated = await setScope(scope);
    if (updated && selectedChatId !== null) {
      await refreshChatSessions();
    }
  };

  const moveChatToFolder = async (
    session: ChatSessionListItem,
    folderId: number | null,
  ) => {
    setOpenChatMenuId(null);
    if (session.folderId === folderId) return;

    const destination = folderId === null
      ? null
      : folders.find(folder => folder.folder_id === folderId) ?? null;
    const confirmed = window.confirm(
      destination
        ? `Move "${session.name}" to "${destination.name}"? While memory is enabled, this chat's existing summary and future saved memory will be available to other chats in this folder.`
        : `Remove "${session.name}" from its folder? If memory is enabled, its saved memory will stop contributing to the folder and remain available only in this chat.`,
    );
    if (!confirmed) return;

    setMovingChatId(session.id);
    try {
      const updated = await setChatFolder(session.id, folderId);
      if (updated) {
        await refreshChatSessions();
      }
    } finally {
      setMovingChatId(null);
    }
  };

  const manageFolders = () => {
    if (memorySettings.folder_id !== null) {
      setFolderFilter(memorySettings.folder_id);
    }
    setDrawerState('expanded');
  };

  const moveChatMenuFocus = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Tab') {
      setOpenChatMenuId(null);
      return;
    }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'),
    );
    if (items.length === 0) return;
    const currentIndex = items.indexOf(document.activeElement as HTMLButtonElement);
    let nextIndex = currentIndex;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = items.length - 1;
    if (event.key === 'ArrowDown') nextIndex = (currentIndex + 1 + items.length) % items.length;
    if (event.key === 'ArrowUp') {
      nextIndex = (currentIndex - 1 + items.length) % items.length;
    }
    items[nextIndex].focus();
  };

  const searchQuery = chatSearch.trim().toLowerCase();
  const selectedFolder = folderFilter === 'all'
    ? null
    : folders.find(folder => folder.folder_id === folderFilter) ?? null;
  const filteredChatSessionLinks = chatSessionLinks.filter((session) => (
    (folderFilter === 'all' || session.folderId === folderFilter)
    && (!searchQuery || session.name.toLowerCase().includes(searchQuery))
  ));
  const sessionCountLabel = `${chatSessionLinks.length} ${chatSessionLinks.length === 1 ? 'chat' : 'chats'}`;
  const activeTurnBelongsToCurrentChat = activeTurn && turnBelongsToChatSelection(
    activeTurn,
    routeChatId,
    state_chat_id,
  );
  const activeTurnAlreadyPersisted = Boolean(
    activeTurn?.run_id &&
    (chatHistory?.chatHistory ?? []).some((turn) => turn.run_id === activeTurn.run_id)
  );
  const displayedHistory = useMemo<ChatPair[]>(
    () => activeTurnBelongsToCurrentChat && !activeTurnAlreadyPersisted
      ? [
          ...(chatHistory?.chatHistory ?? []),
          {
            run_id: activeTurn!.run_id,
            user: activeTurn!.prompt,
            ai: activeTurn!.message,
            status: activeTurn!.status,
            model_load: activeTurn!.model_load,
            tool_calls: activeTurn!.tool_calls,
            artifacts: activeTurn!.artifacts,
          },
        ]
      : (chatHistory?.chatHistory ?? []),
    [
      activeTurn,
      activeTurnAlreadyPersisted,
      activeTurnBelongsToCurrentChat,
      chatHistory?.chatHistory,
    ],
  );

  const handleToolApproval = useCallback(async (
    runId: string,
    callId: string,
    decision: ToolApprovalDecision,
  ) => {
    const response = await fetch(`/agent/runs/${runId}/tool_approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ call_id: callId, decision }),
    });
    if (!response.ok) {
      throw new Error(`Tool approval failed (${response.status})`);
    }
  }, []);

  return (
    <div className={`ChatContainer chat-drawer-${chatDrawerState}${hasChatScrollbar ? ' chat-scrollbar-visible' : ''}`}>
      <section className="ChatContent">
        <div className="chat-stage">
          <div className="chat-transcript-layer" aria-hidden={chatDrawerState === 'expanded'}>
            <div className="chat-history-scroll" ref={chatContainerRef}>
              <ChatTextArea
                chatHistory={displayedHistory}
                isLoading={isLoading}
                onToolApproval={(runId, callId, decision) => {
                  void handleToolApproval(runId, callId, decision);
                }}
              />
            </div>
          </div>

          <aside
            className={`ChatSidebar stage-panel chat-library-panel-host stage-panel-${chatDrawerState}${hasChatDrawerScrollbar ? ' stage-panel-scrollbar-visible' : ''}`}
            aria-label="Chat sessions"
            data-state={chatDrawerState}
          >
            <div className="stage-panel-surface" ref={chatDrawerScrollRef}>
              {chatDrawerState === 'minimized' ? (
                <div className="chat-minimized-controls stage-panel-compact-controls" aria-label="Chat shortcuts">
                  <button className="button chat-minimized-new-button stage-panel-primary-button" type="button" onClick={handleNewChat} aria-label="New chat" title="New chat">
                    <span>New Chat</span>
                    <StagePanelIcon name="plus" />
                  </button>
                  <ChatMemoryControls
                    compact
                    settings={memorySettings}
                    folders={folders}
                    loading={isMemoryLoading}
                    disabled={isLoading}
                    error={memoryError}
                    onScopeChange={scope => void changeMemoryScope(scope)}
                    onManageFolders={manageFolders}
                  />
                  <button
                    className="chat-minimized-open-button stage-panel-open-button"
                    type="button"
                    onClick={() => setDrawerState('expanded')}
                    aria-label="Expand chat history"
                    aria-expanded={false}
                    title="Expand chat history"
                  >
                    <StagePanelIcon name="maximize" />
                  </button>
                </div>
              ) : (
                <div className="chat-drawer-panel stage-panel-content">
                  <div className="stage-panel-content-column">
                    <header className="chat-drawer-header stage-panel-header">
                      <div>
                        <p className="section-eyebrow">Threads</p>
                        <h2>Chats</h2>
                      </div>
                      <button className="icon-action" type="button" onClick={() => setDrawerState('minimized')} aria-label="Close chat history" title="Close chat history">
                        <StagePanelIcon name="close" />
                      </button>
                    </header>

                    <div className="chat-new-toolbar">
                      <button className="button chat-new-button" type="button" onClick={handleNewChat}>
                        <StagePanelIcon name="plus" />
                        <span>New Chat</span>
                      </button>
                      <ChatMemoryControls
                        settings={memorySettings}
                        folders={folders}
                        loading={isMemoryLoading}
                        disabled={isLoading}
                        error={memoryError}
                        onScopeChange={scope => void changeMemoryScope(scope)}
                      />
                    </div>

                    <label className="chat-search" htmlFor="chat-session-search">
                      <StagePanelIcon name="search" />
                      <input
                        id="chat-session-search"
                        value={chatSearch}
                        onChange={(event) => setChatSearch(event.target.value)}
                        placeholder="Search chats"
                        autoFocus
                      />
                    </label>

                    <section className="chat-folder-summary" aria-label="Chat folders">
                      <button
                        className={`chat-folder-row${folderFilter === 'all' ? ' is-selected' : ''}`}
                        type="button"
                        onClick={() => setFolderFilter('all')}
                      >
                        <span className="chat-folder-icon">
                          <StagePanelIcon name="folder" />
                        </span>
                        <span>
                          <strong>All chats</strong>
                          <small>{sessionCountLabel}</small>
                        </span>
                      </button>
                      {folders.map(folder => (
                        <button
                          className={`chat-folder-row folder-${folder.color}${folderFilter === folder.folder_id ? ' is-selected' : ''}`}
                          type="button"
                          key={folder.folder_id}
                          onClick={() => setFolderFilter(folder.folder_id)}
                        >
                          <span className="chat-folder-icon">
                            <StagePanelIcon name="folder" />
                          </span>
                          <span>
                            <strong>{folder.name}</strong>
                            <small>{folder.chat_count} {folder.chat_count === 1 ? 'chat' : 'chats'} · Private</small>
                          </span>
                        </button>
                      ))}
                      {isCreatingFolder ? (
                        <form className="chat-folder-create-form" onSubmit={saveFolder}>
                          <input
                            aria-label="Folder name"
                            value={folderNameDraft}
                            onChange={event => setFolderNameDraft(event.target.value)}
                            placeholder="Folder name"
                            maxLength={120}
                            autoFocus
                          />
                          <button className="icon-action" type="submit" aria-label="Save folder">
                            <StagePanelIcon name="check" />
                          </button>
                          <button
                            className="icon-action"
                            type="button"
                            aria-label="Cancel folder"
                            onClick={() => {
                              setIsCreatingFolder(false);
                              setFolderNameDraft('');
                              setFolderCreateError('');
                            }}
                          >
                            <StagePanelIcon name="close" />
                          </button>
                          {folderCreateError && <span className="chat-title-error" role="alert">{folderCreateError}</span>}
                        </form>
                      ) : (
                        <button
                          className="chat-folder-add"
                          type="button"
                          onClick={() => setIsCreatingFolder(true)}
                        >
                          <StagePanelIcon name="plus" />
                          New private folder
                        </button>
                      )}
                    </section>

                    {selectedFolder && (
                      <section className="chat-folder-memory-preview" aria-label={`${selectedFolder.name} folder details`}>
                        <div className="chat-folder-detail-header">
                          <span>Private folder</span>
                          <span className="chat-folder-detail-actions">
                            <button
                              type="button"
                              aria-label={`Rename ${selectedFolder.name} folder`}
                              onClick={() => {
                                setEditingFolderId(selectedFolder.folder_id);
                                setFolderRenameDraft(selectedFolder.name);
                              }}
                            >
                              <StagePanelIcon name="edit" />
                            </button>
                            <button
                              type="button"
                              aria-label={`Delete ${selectedFolder.name} folder`}
                              onClick={() => setConfirmDeleteFolderId(selectedFolder.folder_id)}
                            >
                              <StagePanelIcon name="close" />
                            </button>
                          </span>
                        </div>
                        {editingFolderId === selectedFolder.folder_id && (
                          <form className="chat-folder-rename-form" onSubmit={saveFolderRename}>
                            <input
                              aria-label="Rename folder"
                              value={folderRenameDraft}
                              onChange={event => setFolderRenameDraft(event.target.value)}
                              autoFocus
                            />
                            <button type="submit" aria-label="Save folder name">
                              <StagePanelIcon name="check" />
                            </button>
                          </form>
                        )}
                        {confirmDeleteFolderId === selectedFolder.folder_id && (
                          <div className="chat-folder-delete-confirm" role="alert">
                            <p>Chats stay private and become unfiled. Shared folder facts and the folder summary will be deleted.</p>
                            <button type="button" onClick={() => setConfirmDeleteFolderId(null)}>Cancel</button>
                            <button
                              className="is-danger"
                              type="button"
                              onClick={() => void removeFolder(selectedFolder.folder_id)}
                            >
                              Delete folder
                            </button>
                          </div>
                        )}
                        {selectedFolder.summary ? (
                          <p>{selectedFolder.summary}</p>
                        ) : (
                          <p>No folder summary yet. Complete a chat here to create one.</p>
                        )}
                      </section>
                    )}

                    <MemoryExplorer
                      scope={selectedFolder ? 'folder' : 'user'}
                      folderId={selectedFolder?.folder_id ?? null}
                    />

                    <div className="chat-drawer-section-title">
                      <span>Recent chats</span>
                    </div>

                    {filteredChatSessionLinks.length > 0 ? (
                      <div className="chat-session-list" role="list" aria-label="Recent chats">
                        {filteredChatSessionLinks.map((session, index) => {
                          const isLast = filteredChatSessionLinks.length === index + 1;
                          const isEditing = editingChatId === session.id;
                          return (
                            <div
                              className={`chat-session-row${isEditing ? ' editing' : ''}`}
                              role="listitem"
                              ref={isLast ? lastItemRef : undefined}
                              key={session.id}
                            >
                              {isEditing ? (
                                <form className="chat-title-edit-form" onSubmit={saveChatTitle}>
                                  <input
                                    className="chat-title-input"
                                    aria-label="Chat name"
                                    value={chatTitleDraft}
                                    onChange={(event) => setChatTitleDraft(event.target.value)}
                                    onKeyDown={(event) => {
                                      if (event.key === 'Escape') {
                                        event.preventDefault();
                                        cancelEditingChatTitle();
                                      }
                                    }}
                                    maxLength={120}
                                    autoFocus
                                  />
                                  <button className="icon-action" type="submit" aria-label="Save chat name" title="Save chat name">
                                    <StagePanelIcon name="check" />
                                  </button>
                                  <button className="icon-action" type="button" onClick={cancelEditingChatTitle} aria-label="Cancel renaming" title="Cancel">
                                    <StagePanelIcon name="close" />
                                  </button>
                                  {chatTitleError && <span className="chat-title-error" role="alert">{chatTitleError}</span>}
                                </form>
                              ) : (
                                <>
                                  <NavLink
                                    to={session.link}
                                    className="list-link chat-session-link"
                                    onClick={() => setDrawerState('minimized')}
                                  >
                                    <span className="chat-history-item">{session.name}</span>
                                    <span className="chat-session-meta">
                                      {!session.memoryEnabled
                                        ? 'Memory off'
                                        : session.folderId !== null
                                          ? folders.find(folder => folder.folder_id === session.folderId)?.name || 'Private folder'
                                          : session.memoryMode === 'private' ? 'Private' : 'Global memory'}
                                    </span>
                                  </NavLink>
                                  <div
                                    className="chat-session-menu-host"
                                    ref={openChatMenuId === session.id ? chatMenuHostRef : undefined}
                                  >
                                    <button
                                      className="chat-session-edit-button"
                                      type="button"
                                      onClick={(event) => {
                                        chatMenuTriggerRef.current = event.currentTarget;
                                        setOpenChatMenuId(
                                          openChatMenuId === session.id ? null : session.id,
                                        );
                                      }}
                                      aria-label={`Chat options for ${session.name}`}
                                      aria-haspopup="menu"
                                      aria-expanded={openChatMenuId === session.id}
                                      title="Chat options"
                                    >
                                      <StagePanelIcon name="more" />
                                    </button>
                                    {openChatMenuId === session.id && (
                                      <div
                                        className="chat-session-row-menu"
                                        role="menu"
                                        aria-label={`${session.name} options`}
                                        onKeyDown={moveChatMenuFocus}
                                      >
                                        <button
                                          type="button"
                                          role="menuitem"
                                          onClick={() => startEditingChatTitle(session)}
                                        >
                                          <StagePanelIcon name="edit" />
                                          <span>Rename</span>
                                        </button>
                                        <span className="chat-session-row-menu-label">Move to</span>
                                        <button
                                          type="button"
                                          role="menuitemradio"
                                          aria-checked={session.folderId === null}
                                          disabled={movingChatId === session.id}
                                          onClick={() => void moveChatToFolder(session, null)}
                                        >
                                          <span className="chat-session-menu-check" aria-hidden="true">
                                            {session.folderId === null ? '✓' : ''}
                                          </span>
                                          <span>No folder</span>
                                        </button>
                                        {folders.map(folder => (
                                          <button
                                            type="button"
                                            role="menuitemradio"
                                            aria-checked={session.folderId === folder.folder_id}
                                            disabled={movingChatId === session.id}
                                            onClick={() => void moveChatToFolder(session, folder.folder_id)}
                                            key={folder.folder_id}
                                          >
                                            <span className="chat-session-menu-check" aria-hidden="true">
                                              {session.folderId === folder.folder_id ? '✓' : ''}
                                            </span>
                                            <span>{folder.name}</span>
                                          </button>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="chat-sessions-empty">
                        <strong>{searchQuery ? 'No matching chats' : 'No chats yet'}</strong>
                        <span>{searchQuery ? 'Try a different search.' : 'Start a new conversation to build your local chat history.'}</span>
                      </div>
                    )}

                    {isChatSessionLoading && <div className="notice">Loading sessions...</div>}
                  </div>
                </div>
              )}
            </div>
          </aside>

          <div className="chat-composer-dock" aria-hidden={chatDrawerState === 'expanded'}>
            {fileContextInfo && (
              <div className="chat-context-info">
                {fileContextInfo}
              </div>
            )}

            <div className="ChatInputForm">
              <EnhancedChatInput
                value={userInput}
                onChange={setUserInput}
                onSubmit={handleSubmit}
                disabled={isLoading || isProcessingFiles || isMemoryLoading}
                placeholder="Type your message..."
                handleKeyDown={handleKeyDown}
                rows={3}
                sessionId={routeChatId ?? state_chat_id ?? 1}
                enableVoice={true}
              />
              {isLoading && (
                <button
                  className="button button-danger"
                  type="button"
                  onClick={() => void cancelGeneration()}
                  disabled={activeTurn?.status === 'cancelling'}
                  aria-label="Stop generating"
                  style={{ marginTop: 8 }}
                >
                  {activeTurn?.status === 'cancelling' ? 'Stopping…' : 'Stop'}
                </button>
              )}
            </div>

            {(error || fileError) && (
              <p className="ErrorMessage">
                Error: {error || fileError}
              </p>
            )}

            {isProcessingFiles && (
              <p className="input-help">Processing file references...</p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Chat;
