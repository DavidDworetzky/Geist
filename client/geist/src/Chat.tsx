import React, { useState, useEffect, FormEvent } from 'react';
import useCompleteText from './Hooks/useCompleteText';
import ChatTextArea from './Components/ChatTextArea';
import { ChatPair, ChatHistory } from './Components/ChatTextArea';

const Chat = () => {
    const [chatHistory, setChatHistory] = useState<ChatHistory>();
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
        <div>
            <ChatTextArea chatHistory={chatHistory?.chatHistory ?? []} />
            
            <form onSubmit={handleSubmit}>
                <textarea
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    rows={3}
                    cols={50}
                    style={{ marginBottom: '10px', width: '100%' }}
                />
                <button type="submit" disabled={isLoading} className={isLoading ? 'loading-dots' : ''}>
                    {isLoading ? '' : 'Send'}
                </button>
            </form>
            {error && <p style={{ color: 'red' }}>Error: {error}</p>}
        </div>
    );
};

export default Chat;
