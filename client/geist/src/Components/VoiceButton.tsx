import React from 'react';

interface VoiceButtonProps {
  isRecording: boolean;
  isProcessing: boolean;
  onClick: () => void;
  disabled?: boolean;
}

const VoiceButton: React.FC<VoiceButtonProps> = ({
  isRecording,
  isProcessing,
  onClick,
  disabled = false
}) => {
  const className = [
    'icon-button',
    isRecording ? 'recording' : '',
    isProcessing ? 'processing' : ''
  ].filter(Boolean).join(' ');

  const getTooltip = () => {
    if (disabled) return 'Voice chat disabled';
    if (isRecording) return 'Click to stop recording';
    if (isProcessing) return 'Processing...';
    return 'Click to start voice chat';
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || isProcessing}
      title={getTooltip()}
      className={className}
    >
      {isRecording ? (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          <rect x="6" y="6" width="8" height="8" />
        </svg>
      ) : (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          <path d="M10 2C8.34 2 7 3.34 7 5v5c0 1.66 1.34 3 3 3s3-1.34 3-3V5c0-1.66-1.34-3-3-3zm0 10c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2s2 .9 2 2v5c0 1.1-.9 2-2 2z" />
          <path d="M15 10c0 2.76-2.24 5-5 5s-5-2.24-5-5H3c0 3.53 2.61 6.43 6 6.92V19h2v-2.08c3.39-.49 6-3.39 6-6.92h-2z" />
        </svg>
      )}
    </button>
  );
};

export default VoiceButton;
