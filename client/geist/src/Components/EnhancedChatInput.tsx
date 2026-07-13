import React, { useState, useRef, KeyboardEvent } from 'react';
import { fileReferenceParser, FileItem } from '../Utils/fileReferenceParser';
import VoiceButton from './VoiceButton';
import useVoiceChat from '../Hooks/useVoiceChat';

interface EnhancedChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  rows?: number;
  handleKeyDown?: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  sessionId?: number;
  enableVoice?: boolean;
}

interface FileSuggestion extends FileItem {
  suggestionText: string;
}

const EnhancedChatInput: React.FC<EnhancedChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder = 'Type your message...',
  rows = 3,
  handleKeyDown: externalHandleKeyDown,
  sessionId = 1,
  enableVoice = true
}) => {
  const [showFileSuggestions, setShowFileSuggestions] = useState(false);
  const [fileSuggestions, setFileSuggestions] = useState<FileSuggestion[]>([]);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const [currentAtPosition, setCurrentAtPosition] = useState(-1);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const {
    isRecording,
    isProcessing,
    partialTranscript,
    toggleRecording
  } = useVoiceChat({
    sessionId,
    onTranscriptFinal: (text) => {
      onChange(text);
    },
    onAssistantText: (text) => {
      console.log('Assistant text chunk:', text);
    },
    onError: (error) => {
      console.error('Voice error:', error);
      alert('Voice error: ' + error);
    }
  });

  const handleInputChange = (newValue: string) => {
    onChange(newValue);

    const caretPosition = textareaRef.current?.selectionStart || 0;
    const textBeforeCaret = newValue.substring(0, caretPosition);
    const atMatch = textBeforeCaret.match(/@([^@\s]*)$/);

    if (atMatch) {
      const partial = atMatch[1];
      const atPosition = caretPosition - partial.length - 1;
      setCurrentAtPosition(atPosition);

      const suggestions = fileReferenceParser.getFileSuggestions(`@${partial}`);
      const enhancedSuggestions: FileSuggestion[] = suggestions.map(file => ({
        ...file,
        suggestionText: fileReferenceParser.generateFileReference(file)
      }));

      setFileSuggestions(enhancedSuggestions);
      setShowFileSuggestions(enhancedSuggestions.length > 0);
      setSelectedSuggestionIndex(-1);
    } else {
      setShowFileSuggestions(false);
      setCurrentAtPosition(-1);
    }
  };

  const insertFileSuggestion = (suggestion: FileSuggestion) => {
    const caretPosition = textareaRef.current?.selectionStart || 0;
    const textBeforeCaret = value.substring(0, currentAtPosition);
    const textAfterCaret = value.substring(caretPosition);
    const newValue = textBeforeCaret + suggestion.suggestionText + ' ' + textAfterCaret;
    onChange(newValue);
    setShowFileSuggestions(false);
    setSelectedSuggestionIndex(-1);

    setTimeout(() => {
      if (textareaRef.current) {
        const newPosition = textBeforeCaret.length + suggestion.suggestionText.length + 1;
        textareaRef.current.setSelectionRange(newPosition, newPosition);
        textareaRef.current.focus();
      }
    }, 0);
  };

  const handleSubmit = () => {
    if (value.trim() && !disabled) {
      onSubmit(value);
    }
  };

  const internalHandleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (externalHandleKeyDown) {
      externalHandleKeyDown(e);
      if (e.defaultPrevented) return;
    }

    if (showFileSuggestions && fileSuggestions.length > 0) {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedSuggestionIndex(prev => prev < fileSuggestions.length - 1 ? prev + 1 : 0);
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedSuggestionIndex(prev => prev > 0 ? prev - 1 : fileSuggestions.length - 1);
          break;
        case 'Tab':
        case 'Enter':
          if (selectedSuggestionIndex >= 0) {
            e.preventDefault();
            insertFileSuggestion(fileSuggestions[selectedSuggestionIndex]);
          } else if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
          }
          break;
        case 'Escape':
          e.preventDefault();
          setShowFileSuggestions(false);
          setSelectedSuggestionIndex(-1);
          break;
      }
    } else if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const parseResult = fileReferenceParser.parseFileReferences(value);
  const hasFileReferences = parseResult.references.length > 0;
  const hasUnresolvedReferences = parseResult.hasUnresolvedReferences;

  return (
    <div className="enhanced-input">
      {partialTranscript && (
        <div className="input-banner">
          <strong>Listening...</strong>
          <div>{partialTranscript}</div>
        </div>
      )}

      {hasFileReferences && (
        <div className={`input-banner ${hasUnresolvedReferences ? 'input-banner-warning' : 'input-banner-success'}`}>
          <strong>File references ({parseResult.references.length})</strong>
          {parseResult.references.map((ref, index) => (
            <div key={index} className="file-reference-line">
              <span>{ref.resolved ? 'Resolved' : 'Missing'}</span>
              <span>{ref.originalText}</span>
              {ref.resolved && <span className="suggestion-meta">to {ref.filename}</span>}
              {!ref.resolved && <span className="suggestion-meta">file not found</span>}
            </div>
          ))}
        </div>
      )}

      <div className="input-row">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => handleInputChange(e.target.value)}
          onKeyDown={internalHandleKeyDown}
          rows={rows}
          disabled={disabled}
          placeholder={placeholder}
          className="chat-textarea"
        />

        {enableVoice && (
          <VoiceButton
            isRecording={isRecording}
            isProcessing={isProcessing}
            onClick={toggleRecording}
            disabled={disabled}
          />
        )}

        <button
          type="button"
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          className="send-button"
        >
          Send
        </button>
      </div>

      {showFileSuggestions && (
        <div className="suggestion-menu">
          {fileSuggestions.map((suggestion, index) => (
            <button
              type="button"
              key={suggestion.file_id}
              onClick={() => insertFileSuggestion(suggestion)}
              className={`suggestion-item${index === selectedSuggestionIndex ? ' active' : ''}`}
              onMouseEnter={() => setSelectedSuggestionIndex(index)}
            >
              <strong>{suggestion.suggestionText}</strong>
              <span className="suggestion-meta">
                {suggestion.original_filename} - {Math.round(suggestion.file_size / 1024)}KB
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="input-help">Use @ to reference files. Press Tab or Enter to accept suggestions.</div>
    </div>
  );
};

export default EnhancedChatInput;
