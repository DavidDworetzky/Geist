import React, { useState, useEffect, FormEvent } from 'react';
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
    const { chatSessions, loading: isChatSessionLoading, error: chatSessionError } = useGetChatSessions();
    const [userInput, setUserInput] = useState('');
    const [fileContextInfo, setFileContextInfo] = useState<string>('');
    const { prompt, completeText, loading: isLoading, error, completedText, state_chat_id } = useCompleteText();
    const { processMessage, isProcessing: isProcessingFiles, error: fileError } = useFileContext();

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
            <div className="ChatSidebar">
                <LinkList listItems={chatSessionLinks} />
            </div>
            <div className="ChatContent">
                <ChatTextArea chatHistory={chatHistory?.chatHistory ?? []} />
                
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
