import React, { forwardRef } from 'react';

export interface ChatPair {
  user: string;
  ai: string;
}

export interface ChatHistory {
  chatHistory: ChatPair[];
}

interface ChatTextAreaProps extends ChatHistory {
  isLoading?: boolean;
}

const ChatTextArea = forwardRef<HTMLDivElement, ChatTextAreaProps>((props, ref) => {
  const isEmpty = props.chatHistory.length === 0 && !props.isLoading;

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
      {props.isLoading && (
        <div
          className="chat-loading-indicator"
          role="status"
          aria-label="Geist is responding"
        >
          <span className="chat-loading-dot" aria-hidden="true" />
          <span className="chat-loading-dot" aria-hidden="true" />
          <span className="chat-loading-dot" aria-hidden="true" />
        </div>
      )}
    </div>
  );
});

export default ChatTextArea;
