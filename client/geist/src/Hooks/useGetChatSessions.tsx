import { useState, useEffect } from 'react';

export interface ChatMessage {
    user: string;
    ai: string;
}

export interface ChatSession {
    chat_history: ChatMessage[];
    chat_id: number;
}

const useGetChatSessions = () => {
    const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchChatSessions = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch('/agent/chat_sessions');
            if (!response.ok) {
                throw new Error('Failed to fetch chat sessions');
            }
            const data: ChatSession[] = await response.json();
            setChatSessions(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchChatSessions();
    }, []);

    return {
        chatSessions,
        loading,
        error,
        refreshChatSessions: fetchChatSessions
    };
};

export default useGetChatSessions;
