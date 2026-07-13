import React, { forwardRef } from 'react';

export interface ChatPair {
  user: string;
  ai: string;
}

export interface ChatHistory {
  chatHistory: ChatPair[];
}

const ChatTextArea = forwardRef<HTMLDivElement, ChatHistory>((props, ref) => {
  const isEmpty = props.chatHistory.length === 0;

  return (
    <div ref={ref} className={`chat-history${isEmpty ? ' chat-history-empty' : ''}`}>
      {isEmpty && (
        <div className="chat-empty-state">Start a conversation with Geist.</div>
      )}
      {props.chatHistory.map((element, index) => (
        <div key={index} className="chat-turn">
          <div className="chat-message chat-message-user">
            <span className="chat-speaker">User</span>
            {element.user}
          </div>
          <div className="chat-message chat-message-ai">
            <span className="chat-speaker">Geist</span>
            {element.ai}
          </div>
        </div>
      ))}
    </div>
  );
});

export default ChatTextArea;
