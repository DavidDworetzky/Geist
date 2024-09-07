import { useState, FormEvent } from 'react';
import useCompleteText from './Hooks/useCompleteText';

const Chat = () => {
    const [chatHistory, setChatHistory] = useState('');
    const [userInput, setUserInput] = useState('');
    const { completeText, isLoading, error } = useCompleteText();

    const chatWithServer = async (input: string) => {
        try {
            const response = await completeText(input);
            setChatHistory(prevHistory => prevHistory + '\nUser: ' + input + '\nAI: ' + response);
            return response;
        } catch (err) {
            console.error('Error chatting with server:', err);
            return '';
        }
    };

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
