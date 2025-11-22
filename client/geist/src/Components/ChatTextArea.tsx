import React, { forwardRef } from 'react';

export interface ChatPair {
    user: string;
    ai: string;
};

export interface ChatHistory {
    chatHistory: ChatPair[];
}

const ChatTextArea = forwardRef<HTMLDivElement, ChatHistory>((props, ref) => {
    return (
        <div ref={ref} style={{
            border: '1px solid #ccc',
            borderRadius: '4px',
            padding: '12px',
            minHeight: '200px',
            width: '100%',
            backgroundColor: 'white',
            overflowY: 'auto',
            fontFamily: 'inherit',
            boxShadow: 'inset 0 1px 2px rgba(0, 0, 0, 0.1)'
        }}>
            {props.chatHistory.map((element, index) => (
                <div key={index}>
                    <p style={{color: 'blue'}}>User: {element.user}</p>
                    <p style={{color: 'purple'}}>AI: {element.ai}</p>
                </div>
            ))}
        </div>
    );
});

export default ChatTextArea;;