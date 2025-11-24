import { useState, useEffect, useCallback } from 'react';

export interface ChatMessage {
    user: string;
    ai: string;
}

export interface ChatSession {
    chat_history: ChatMessage[];
    chat_id: number;
    create_date: string;
}

const useGetChatSessions = () => {
    const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [hasMore, setHasMore] = useState(true);
    const [page, setPage] = useState(1);

    const fetchChatSessions = useCallback(async (pageNum: number = 1, reset: boolean = false) => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`/agent/chat_sessions/paginated?page=${pageNum}&page_size=20`);
            if (!response.ok) {
                throw new Error('Failed to fetch chat sessions');
            }
            const data: ChatSession[] = await response.json();
            
            setChatSessions(prev => reset ? data : [...prev, ...data]);
            setHasMore(data.length === 20); // Assuming page size 20
            setPage(pageNum);
            
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchChatSessions(1, true);
    }, [fetchChatSessions]);

    const loadMore = useCallback(() => {
        if (!loading && hasMore) {
            fetchChatSessions(page + 1, false);
        }
    }, [loading, hasMore, page, fetchChatSessions]);

    const refreshChatSessions = useCallback(() => fetchChatSessions(1, true), [fetchChatSessions]);

    return {
        chatSessions,
        loading,
        error,
        hasMore,
        loadMore,
        refreshChatSessions
    };
};

export default useGetChatSessions;
