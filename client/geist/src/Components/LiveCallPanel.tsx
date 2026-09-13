import React from 'react';
import BrandMark from './BrandMark';

interface LiveCallPanelProps {
  active: boolean;
  status: string;
  audioLevel: number;
  userText: string;
  assistantText: string;
}

export default function LiveCallPanel({ active, status, audioLevel, userText, assistantText }: LiveCallPanelProps) {
  return (
    <div className="live-call-panel" aria-label="Live voice conversation">
      <div className={`live-call-mark${active ? ' active' : ''}`}
        style={{ '--voice-level': audioLevel } as React.CSSProperties} aria-hidden="true">
        <BrandMark />
      </div>
      <div className="live-call-status" role="status">{status || 'Ready for a conversation'}</div>
      <div className="live-call-hint">{active ? 'Speak naturally. You can interrupt at any time.' : 'Start a continuous voice conversation.'}</div>
      {(userText || assistantText) && <div className="live-call-captions" aria-label="Live captions">
        {userText && <div><strong>You</strong><p>{userText}</p></div>}
        {assistantText && <div><strong>Geist</strong><p>{assistantText}</p></div>}
      </div>}
    </div>
  );
}
