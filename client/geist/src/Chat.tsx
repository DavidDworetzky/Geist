import React, { useState, useEffect, useRef, useLayoutEffect, useCallback, useMemo } from 'react';
import useCompleteText from './Hooks/useCompleteText';
import useGetChatSessions from './Hooks/useGetChatSessions';
import useFileContext from './Hooks/useFileContext';
import useUserSettings from './Hooks/useUserSettings';
import ChatTextArea from './Components/ChatTextArea';
import EnhancedChatInput from './Components/EnhancedChatInput';
import StagePanelIcon from './Components/StagePanelIcon';
import { ChatPair, ChatHistory } from './Components/ChatTextArea';
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
}

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
  const { chatSessions, loading: isChatSessionLoading, error: chatSessionError, loadMore: loadMoreSessions, hasMore: hasMoreSessions } = useGetChatSessions();
  const [userInput, setUserInput] = useState('');
  const [fileContextInfo, setFileContextInfo] = useState<string>('');
  const { settings: userSettings } = useUserSettings();
  const { prompt, completeText, loading: isLoading, error, completedText, state_chat_id } = useCompleteText(userSettings);
  const { processMessage, isProcessing: isProcessingFiles, error: fileError } = useFileContext();

  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
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
        user: h.user,
        ai: h.ai
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
  }, [chatHistory, page]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (container && hasMore && !isLoadingHistory && (chatHistory?.chatHistory?.length ?? 0) > 0) {
      if (container.scrollHeight <= container.clientHeight) {
        const nextPage = page + 1;
        setPage(nextPage);
        if (chatId) {
          fetchHistory(nextPage, chatId);
        }
      }
    }
  }, [chatHistory, hasMore, isLoadingHistory, page, chatId, fetchHistory]);

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

  const handleScroll = useCallback(() => {
    if (chatContainerRef.current) {
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
  }, [hasMore, isLoadingHistory, page, chatId, fetchHistory]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll]);

  const chatWithServer = async (input: string) => {
    const parsedChatId = chatId ? parseInt(chatId) : state_chat_id;
    try {
      const processedMessage = await processMessage(input);

      if (processedMessage.contexts.length > 0) {
        const contextInfo = `Files referenced: ${processedMessage.contexts.map(c => c.filename).join(', ')}`;
        setFileContextInfo(contextInfo);
      } else {
        setFileContextInfo('');
      }

      const messageToSend = processedMessage.enhancedMessage || input;
      await completeText(messageToSend, parsedChatId);
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
        };
      })
      .sort((a, b) => b.date.getTime() - a.date.getTime());
  }, [chatSessions, chatTitles]);
  useEffect(() => {
    if (completedText) {
      const newHistory: ChatPair = {
        user: prompt ?? '',
        ai: completedText
      };
      setChatHistory(prev => ({ chatHistory: [...(prev?.chatHistory ?? []), newHistory] }));
    }
  }, [completedText, prompt]);

  const handleSubmit = async (message: string) => {
    if (message.trim()) {
      await chatWithServer(message);
      setUserInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (userInput.trim() && !isLoading) {
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
    }
  };

  const handleNewChat = () => {
    setDrawerState('minimized');
    setChatHistory(undefined);
    setUserInput('');
    setFileContextInfo('');
    navigate('/chat');
  };
  const startEditingChatTitle = (session: ChatSessionListItem) => {
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

  const searchQuery = chatSearch.trim().toLowerCase();
  const filteredChatSessionLinks = searchQuery
    ? chatSessionLinks.filter((session) => session.name.toLowerCase().includes(searchQuery))
    : chatSessionLinks;
  const sessionCountLabel = `${chatSessionLinks.length} ${chatSessionLinks.length === 1 ? 'chat' : 'chats'}`;

  return (
    <div className={`ChatContainer chat-drawer-${chatDrawerState}`}>
      <section className="ChatContent">
        <div className="chat-stage">
          <div className="chat-transcript-layer" aria-hidden={chatDrawerState === 'expanded'}>
            <ChatTextArea chatHistory={chatHistory?.chatHistory ?? []} ref={chatContainerRef} />
          </div>

          <aside
            className={`ChatSidebar stage-panel chat-library-panel-host stage-panel-${chatDrawerState}`}
            aria-label="Chat sessions"
            data-state={chatDrawerState}
          >
            <div className="stage-panel-surface">
              {chatDrawerState === 'minimized' ? (
                <div className="chat-minimized-controls stage-panel-compact-controls" aria-label="Chat shortcuts">
                  <button className="button chat-minimized-new-button stage-panel-primary-button" type="button" onClick={handleNewChat} aria-label="New chat" title="New chat">
                    <span>New Chat</span>
                    <StagePanelIcon name="plus" />
                  </button>
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

                    <button className="button chat-new-button" type="button" onClick={handleNewChat}>
                      <StagePanelIcon name="plus" />
                      <span>New Chat</span>
                    </button>

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
                      <div className="chat-folder-row">
                        <span className="chat-folder-icon">
                          <StagePanelIcon name="folder" />
                        </span>
                        <span>
                          <strong>Recent</strong>
                          <small>{sessionCountLabel}</small>
                        </span>
                      </div>
                    </section>

                    <div className="chat-drawer-section-title">
                      <span>Recent chats</span>
                      <button className="chat-icon-plain" type="button" aria-label="Chat list options" title="Chat list options">
                        <StagePanelIcon name="more" />
                      </button>
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
                                  </NavLink>
                                  <button
                                    className="chat-session-edit-button"
                                    type="button"
                                    onClick={() => startEditingChatTitle(session)}
                                    aria-label={`Rename ${session.name}`}
                                    title="Rename chat"
                                  >
                                    <StagePanelIcon name="edit" />
                                  </button>
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
                disabled={isLoading || isProcessingFiles}
                placeholder="Type your message..."
                handleKeyDown={handleKeyDown}
                rows={3}
                sessionId={state_chat_id || (chatId ? parseInt(chatId) : 1)}
                enableVoice={true}
              />
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
