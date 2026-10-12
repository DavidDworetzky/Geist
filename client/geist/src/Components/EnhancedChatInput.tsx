import React, { useState, useRef, useId, KeyboardEvent } from 'react';
import { fileReferenceParser, FileItem } from '../Utils/fileReferenceParser';
import VoiceButton from './VoiceButton';
import VoiceSettings, { DEFAULT_VOICE_SELECTION, VoiceSelection } from './VoiceSettings';
import useVoiceChat from '../Hooks/useVoiceChat';
import LiveCallPanel from './LiveCallPanel';
import ChatTextArea from './ChatTextArea';
import useLiveTools from '../Hooks/useLiveTools';
import { UserSettings } from '../Hooks/useUserSettings';

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
  submitLabel?: string;
  modelLoading?: boolean;
  onShowModelError?: () => void;
  voiceAgentSettings?: UserSettings | null;
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
  enableVoice = true,
  submitLabel = 'Send',
  modelLoading = false,
  onShowModelError,
  voiceAgentSettings = null,
}) => {
  const inputStatusId = useId();
  const hasInputStatus = modelLoading || Boolean(onShowModelError);
  const [showFileSuggestions, setShowFileSuggestions] = useState(false);
  const [fileSuggestions, setFileSuggestions] = useState<FileSuggestion[]>([]);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const [currentAtPosition, setCurrentAtPosition] = useState(-1);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const [voiceSelection, setVoiceSelection] = useState<VoiceSelection>(DEFAULT_VOICE_SELECTION);
  const [voiceError, setVoiceError] = useState('');
  const liveTools = useLiveTools(voiceAgentSettings);

  const {
    isRecording,
    isProcessing,
    partialTranscript,
    assistantText,
    status,
    audioLevel,
    toggleRecording
  } = useVoiceChat({
    sessionId,
    mode: voiceSelection.mode,
    sttProvider: voiceSelection.sttProvider,
    ttsProvider: voiceSelection.ttsProvider,
    ttsModel: voiceSelection.ttsModel,
    ttsVoice: voiceSelection.ttsVoice,
    onLiveStart: liveTools.reset,
    onLiveDelegate: voiceAgentSettings ? liveTools.delegate : undefined,
    onTranscriptFinal: (text) => {
      if (text) onChange(value + (value && !/\s$/.test(value) ? ' ' : '') + text);
    },
    onError: (error) => {
      setVoiceError(error);
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
    if (value.trim() && !disabled && !isRecording && !isProcessing) {
      onSubmit(value);
    }
  };

  const internalHandleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((isRecording || isProcessing) && e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      return;
    }
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
      {enableVoice && voiceSelection.mode === 'live' && <LiveCallPanel active={isRecording}
        status={status} audioLevel={audioLevel} userText={partialTranscript} assistantText={assistantText} />}
      {voiceSelection.mode === 'live' && liveTools.turn && (
        <section className="live-tool-activity" aria-label="Voice tool activity">
          <strong>Tool activity</strong>
          <ChatTextArea chatHistory={[{ ...liveTools.turn, user: 'Live voice request', ai: liveTools.turn.message }]}
            isLoading={liveTools.loading} onToolApproval={liveTools.approve} />
          {liveTools.error && <p role="alert">{liveTools.error}</p>}
          {liveTools.loading && <button type="button" className="button button-secondary"
            onClick={() => void liveTools.cancel()}>Stop tool task</button>}
        </section>
      )}
      {voiceError && <div className="input-banner input-banner-warning" role="alert">{voiceError}</div>}
      {voiceSelection.mode === 'dictation' && status && <div className="input-banner" role="status">{status}</div>}

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
        {(!enableVoice || voiceSelection.mode === 'dictation') && (
          <div className="chat-input-field">
          <textarea
            ref={textareaRef}
            value={hasInputStatus ? '' : value}
            onChange={(e) => handleInputChange(e.target.value)}
            onKeyDown={internalHandleKeyDown}
            rows={rows}
            disabled={disabled}
            placeholder={hasInputStatus ? '' : placeholder}
            aria-label="Message"
            aria-describedby={hasInputStatus ? inputStatusId : undefined}
            aria-busy={modelLoading || undefined}
            className="chat-textarea"
          />
          {onShowModelError ? (
            <div className="chat-input-status chat-model-unavailable" id={inputStatusId}>
              <span>Model unavailable</span>
              <button className="model-error-badge" type="button" onClick={onShowModelError}>
                See more…
              </button>
            </div>
          ) : modelLoading && (
            <div className="chat-input-status" id={inputStatusId} role="status">
              <span className="chat-runtime-state">
                <span className="runtime-model-spinner" aria-hidden="true" />
                Loading model…
              </span>
            </div>
          )}
        </div>
        )}

        <div className="input-actions">
          {enableVoice && (
            <>
              <VoiceSettings
                selection={voiceSelection}
                onChange={setVoiceSelection}
                disabled={isRecording || isProcessing}
              />
              <VoiceButton
                isRecording={isRecording}
                isProcessing={isProcessing}
                onClick={() => { setVoiceError(''); toggleRecording(); }}
                disabled={voiceSelection.mode === 'dictation' && disabled}
                mode={voiceSelection.mode}
              />
            </>
          )}

          {(!enableVoice || voiceSelection.mode === 'dictation') && <button
            type="button"
            onClick={handleSubmit}
            disabled={disabled || isRecording || isProcessing || !value.trim()}
            className="send-button"
          >
            {submitLabel}
          </button>}
        </div>
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

      {voiceSelection.mode === 'dictation' && <div className="input-help">Dictation adds text to your message. Review it, then Send.</div>}
    </div>
  );
};

export default EnhancedChatInput;
