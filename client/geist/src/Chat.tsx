import React, { useState, useEffect, FormEvent } from 'react';
import useCompleteText from './Hooks/useCompleteText';

const Chat = () => {
    const [chatHistory, setChatHistory] = useState('');
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
            const previousHistory = '\nUser: ' + prompt + '\nAI: ' + completedText;
            setChatHistory(prev => prev + previousHistory);
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
            <textarea
                value={chatHistory}
                readOnly
                rows={10}
                cols={50}
                style={{ marginBottom: '10px', width: '100%' }}
            />
            <form onSubmit={handleSubmit}>
                <textarea
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    rows={3}
                    cols={50}
                    style={{ marginBottom: '10px', width: '100%' }}
                />
                <button type="submit" disabled={isLoading}>
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </form>
            {error && <p style={{ color: 'red' }}>Error: {error}</p>}
        </div>
    );
};

export default Chat;
