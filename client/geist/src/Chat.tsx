import React, { useState, useEffect, FormEvent, useRef, useLayoutEffect } from 'react';
import useCompleteText from './Hooks/useCompleteText';
import useGetChatSessions from './Hooks/useGetChatSessions';
import useFileContext from './Hooks/useFileContext';
import {ChatSession}from './Hooks/useGetChatSessions';
import ChatTextArea from './Components/ChatTextArea';
import LinkList from './Components/LinkList';
import EnhancedChatInput from './Components/EnhancedChatInput';
import { ChatPair, ChatHistory } from './Components/ChatTextArea';
import {ListItem} from './Components/LinkList';
import { useParams } from 'react-router-dom';


const Chat = () => {
    const { chatId } = useParams<{ chatId?: string }>();
    const [chatHistory, setChatHistory] = useState<ChatHistory>();
    const [chatSessionData, setChatSessions] = useState<ChatSession[]>([]);
    const [chatSessionLinks, setChatSessionLinks] = useState<ListItem[]>([]);
    const { chatSessions, loading: isChatSessionLoading, error: chatSessionError, loadMore: loadMoreSessions, hasMore: hasMoreSessions } = useGetChatSessions();
    const [userInput, setUserInput] = useState('');
    const [fileContextInfo, setFileContextInfo] = useState<string>('');
    const { prompt, completeText, loading: isLoading, error, completedText, state_chat_id } = useCompleteText();
    const { processMessage, isProcessing: isProcessingFiles, error: fileError } = useFileContext();

    // Pagination state for chat history
    const [page, setPage] = useState(1);
    const [hasMore, setHasMore] = useState(true);
    const [isLoadingHistory, setIsLoadingHistory] = useState(false);
    const PAGE_SIZE = 20;
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const prevScrollHeightRef = useRef<number>(0);
    const shouldRestoreScrollRef = useRef(false);

    const fetchHistory = async (pageNum: number, currentChatId: string) => {
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
            if (newMessages.length < PAGE_SIZE || (pageNum * PAGE_SIZE >= total)) {
                setHasMore(false);
            } else {
                setHasMore(true);
            }
        } catch (err) {
            console.error('Error fetching chat history:', err);
        } finally {
            setIsLoadingHistory(false);
        }
    };

    // Load history when chatId changes
    useEffect(() => {
        if (chatId) {
            setPage(1);
            setHasMore(true);
            fetchHistory(1, chatId);
        }
    }, [chatId]);

    // Restore scroll position after prepending messages
    useLayoutEffect(() => {
        if (shouldRestoreScrollRef.current && chatContainerRef.current && prevScrollHeightRef.current) {
            const newScrollHeight = chatContainerRef.current.scrollHeight;
            const diff = newScrollHeight - prevScrollHeightRef.current;
            chatContainerRef.current.scrollTop = diff;
            shouldRestoreScrollRef.current = false;
        } else if (page === 1 && chatContainerRef.current && !shouldRestoreScrollRef.current) {
             // Scroll to bottom on initial load
             chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory, page]);

    // Auto-load more history if the container is not full
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
    }, [chatHistory, hasMore, isLoadingHistory, page, chatId]);

    const handleScroll = () => {
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
    };

    // Attach scroll listener
    useEffect(() => {
        const container = chatContainerRef.current;
        if (container) {
            container.addEventListener('scroll', handleScroll);
            return () => container.removeEventListener('scroll', handleScroll);
        }
    }, [hasMore, isLoadingHistory, page, chatId]); // Re-bind when dependencies change

    const chatWithServer = async (input: string) => {
        let parsedChatId = chatId ? parseInt(chatId) : state_chat_id;
        try {
            // Process file references if any exist
            const processedMessage = await processMessage(input);
            
            // Set file context info for display
            if (processedMessage.contexts.length > 0) {
                const contextInfo = `Files referenced: ${processedMessage.contexts.map(c => c.filename).join(', ')}`;
                setFileContextInfo(contextInfo);
            } else {
                setFileContextInfo('');
            }
            
            // Use enhanced message (with file content) for the API call
            const messageToSend = processedMessage.enhancedMessage || input;
            await completeText(messageToSend, parsedChatId);
            
        } catch (err) {
            console.error('Error chatting with server:', err);
        }
    };

    useEffect(() => {
        if (chatSessions) {
            setChatSessions(chatSessions);
            const chatSessionListItems = chatSessions.map((session) => {
                const date = new Date(session.create_date);
                const formattedDate = date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
                const firstLine = session.chat_history[0]?.user?.split('\n')[0] || 'New Chat';
                const firstThreeWords = firstLine.split(' ').slice(0, 3).join(' ');
                const summary = `${formattedDate} - ${firstThreeWords}`;
                return {
                    name: summary,
                    link: `/chat/${session.chat_id}`,
                    date: date,
                };
            });
            // Sorting is handled by backend mostly, but safe to keep for accumulated list
            const sortedItems = chatSessionListItems.sort((a, b) => b.date.getTime() - a.date.getTime());
            setChatSessionLinks(sortedItems);
        }
    }, [chatSessions]);

    const handleSidebarScroll = (e: React.UIEvent<HTMLDivElement>) => {
        const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
        // Load more when scrolled to bottom (with small buffer)
        if (scrollHeight - scrollTop <= clientHeight + 50) {
            if (!isChatSessionLoading && hasMoreSessions) {
                loadMoreSessions();
            }
        }
    };

    useEffect(() => {
        if (completedText) {
            const newHistory: ChatPair = {
                user: prompt ?? '',
                ai: completedText
            };
            const amendedHistory: ChatHistory = {chatHistory: [...chatHistory?.chatHistory ?? [], newHistory]};
            setChatHistory(amendedHistory);
        }
    }, [completedText]);

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
                handleSubmit(userInput);
            }
        }
    };

    return (
        <div className="ChatContainer">
            <div className="ChatSidebar" onScroll={handleSidebarScroll} style={{ overflowY: 'auto' }}>
                <LinkList listItems={chatSessionLinks} />
                {isChatSessionLoading && <div style={{ padding: '10px', textAlign: 'center' }}>Loading...</div>}
            </div>
            <div className="ChatContent">
                <ChatTextArea chatHistory={chatHistory?.chatHistory ?? []} ref={chatContainerRef} />
                
                {/* File context info */}
                {fileContextInfo && (
                    <div style={{
                        padding: '8px 12px',
                        backgroundColor: '#d4edda',
                        border: '1px solid #c3e6cb',
                        borderRadius: '4px',
                        fontSize: '12px',
                        color: '#155724',
                        marginBottom: '10px'
                    }}>
                        {fileContextInfo}
                    </div>
                )}
                
                <div className="ChatInputForm">
                    <EnhancedChatInput
                        value={userInput}
                        onChange={setUserInput}
                        onSubmit={handleSubmit}
                        disabled={isLoading || isProcessingFiles}
                        placeholder="Type your message... Use @ to reference files"
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
                    <p style={{ fontSize: '12px', color: '#6c757d', fontStyle: 'italic' }}>
                        Processing file references...
                    </p>
                )}
            </div>
        </div>
    );
};

export default Chat;
