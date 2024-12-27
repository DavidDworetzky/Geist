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
    const { prompt, completeText, loading: isLoading, error, completedText, state_chat_id } = useCompleteText();

    const chatWithServer = async (input: string) => {
        let parsedChatId = chatId ? parseInt(chatId) : state_chat_id;
        try {
            await completeText(input, parsedChatId);
        }
        catch (err) {
            console.error('Error chatting with server:', err);
        }
    };

    // when chat sessions are done loading, set the chat history
    useEffect(() => {
        if (!isChatSessionLoading && chatSessions) {
            setChatSessions(chatSessions);
            const chatSessionListItems = chatSessions.map((session) => {
                const date = new Date(session.create_date);
                const formattedDate = date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
                const firstLine = session.chat_history[0].user.split('\n')[0];
                const firstThreeWords = firstLine.split(' ').slice(0, 3).join(' ');
                const summary = `${formattedDate} - ${firstThreeWords}`;
                return {
                    name: summary,
                    link: `/chat/${session.chat_id}`,
                    date: date,
                };
            });
            const sortedItems = chatSessionListItems.sort((a, b) => b.date.getTime() - a.date.getTime());
            setChatSessionLinks(sortedItems);

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
        <div className="ChatContainer">
            <div className="ChatSidebar">
                <LinkList listItems={chatSessionLinks} />
            </div>
            <div className="ChatContent">
                <ChatTextArea chatHistory={chatHistory?.chatHistory ?? []} />
                
                <form onSubmit={handleSubmit} className="ChatInputForm">
                    <textarea
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        rows={3}
                        cols={50}
                        className="ChatInput"
                    />
                    <button 
                        type="submit" 
                        disabled={isLoading} 
                        className={`ChatSubmitButton ${isLoading ? 'loading-dots' : ''}`}
                    >
                        {isLoading ? '' : 'Send'}
                    </button>
                </form>
                {error && <p className="ErrorMessage">Error: {error}</p>}
            </div>
        </div>
    );
};

export default Chat;
