import React, { useState, useEffect, FormEvent } from 'react';
import useCompleteText from './Hooks/useCompleteText';
import useGetChatSessions from './Hooks/useGetChatSessions';
import {ChatSession}from './Hooks/useGetChatSessions';
import ChatTextArea from './Components/ChatTextArea';
import LinkList from './Components/LinkList';
import { ChatPair, ChatHistory } from './Components/ChatTextArea';
import {ListItem} from './Components/LinkList';
import { useParams } from 'react-router-dom';


const Chat = () => {
    const { chatId } = useParams<{ chatId?: string }>();
    const [chatHistory, setChatHistory] = useState<ChatHistory>();
    const [chatSessionData, setChatSessions] = useState<ChatSession[]>([]);
    const [chatSessionLinks, setChatSessionLinks] = useState<ListItem[]>([]);
    const { chatSessions, loading: isChatSessionLoading, error: chatSessionError } = useGetChatSessions();
    const [userInput, setUserInput] = useState('');
    const { prompt, completeText, loading: isLoading, error, completedText } = useCompleteText();

    const chatWithServer = async (input: string) => {
        try {
            await completeText(input);
        }
        catch (err) {
            console.error('Error chatting with server:', err);
        }
    };

    // when chat sessions are done loading, set the chat history
    useEffect(() => {
        if (!isChatSessionLoading && chatSessions) {
            setChatSessions(chatSessions);
            const chatSessionListItems = chatSessions.map((session) => ({
                name: session.chat_id.toString(),
                link: `/chat/${session.chat_id}`
            }));
            setChatSessionLinks(chatSessionListItems);

            // Load chat history for specific chat ID
            if (chatId) {
                const selectedSession = chatSessions.find(
                    session => session.chat_id.toString() === chatId
                );
                if (selectedSession?.chat_history) {
                    const initialHistory: ChatHistory = {
                        chatHistory: selectedSession.chat_history.map(h => ({
                            user: h.user,
                            ai: h.ai
                        }))
                    };
                    setChatHistory(initialHistory);
                }
            }
        }
    }, [isChatSessionLoading, chatId]);


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

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (userInput.trim()) {
            await chatWithServer(userInput);
            setUserInput('');
        }
    };

    return (
        <div className="Wrapper">
        <LinkList listItems={chatSessionLinks} />
        <div className="LinkContent">
            <ChatTextArea chatHistory={chatHistory?.chatHistory ?? []} />
            
            <form onSubmit={handleSubmit}>
                <textarea
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    rows={3}
                    cols={50}
                    style={{ marginBottom: '10px', marginTop: '20px', width: '100%' }}
                />
                <button type="submit" disabled={isLoading} className={isLoading ? 'loading-dots' : ''}>
                    {isLoading ? '' : 'Send'}
                </button>
            </form>
            {error && <p style={{ color: 'red' }}>Error: {error}</p>}
        </div>
        </div>
    );
};

export default Chat;
