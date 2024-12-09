import React, { useState, useEffect, FormEvent } from 'react';
import useCompleteText from './Hooks/useCompleteText';
import useGetChatSessions from './Hooks/useGetChatSessions';
import {ChatSession}from './Hooks/useGetChatSessions';
import ChatTextArea from './Components/ChatTextArea';
import LinkList from './Components/LinkList';
import { ChatPair, ChatHistory } from './Components/ChatTextArea';


const Chat = () => {
    const [chatHistory, setChatHistory] = useState<ChatHistory>();
    const [chatSessionData, setChatSessions] = useState<ChatSession[]>([]);
    const [chatSessionLinks, setChatSessionLinks] = useState<{name: number, link: string}[]>([]);
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
        if (!isChatSessionLoading) {
            setChatSessions(chatSessions);
            const chatSessionListItems = chatSessions.map((session) => ({
                name: session.chat_id,
                link: `/chat/${session.chat_id}`
            }));
            setChatSessionLinks(chatSessionListItems);
        }
    }, [isChatSessionLoading]);


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
        <>
        <LinkList listItems={chatSessionLinks} />
        <div>
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
        </>
    );
};

export default Chat;
