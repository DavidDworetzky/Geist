import React, { useState, useEffect, FormEvent } from 'react';

type ChatPair = {
    user: string;
    ai: string;
};

const ChatTextArea = () => {
    const [chatPairs, setChatPairs] = useState<ChatPair[]>([]);
    return (
        <div style={{
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
            {chatPairs.map((pair, index) => (
                <div key={index}>
                    <p style={{color: 'blue'}}>{pair.user}</p>
                    <p style={{color: 'purple'}}>{pair.ai}</p>
                </div>
            ))}
        </div>
    );
};

export default ChatTextArea;;