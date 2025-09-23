import React, { useState, useEffect, useRef, KeyboardEvent } from 'react';
import { fileReferenceParser, FileItem } from '../Utils/fileReferenceParser';

interface EnhancedChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  rows?: number;
  handleKeyDown?: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
}

interface FileSuggestion extends FileItem {
  suggestionText: string;
}

const EnhancedChatInput: React.FC<EnhancedChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder = "Type your message... Use @ to reference files",
  rows = 3,
  handleKeyDown: externalHandleKeyDown
}) => {
  const [showFileSuggestions, setShowFileSuggestions] = useState(false);
  const [fileSuggestions, setFileSuggestions] = useState<FileSuggestion[]>([]);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const [currentAtPosition, setCurrentAtPosition] = useState(-1);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Handle @ symbol detection and file suggestions
  const handleInputChange = (newValue: string) => {
    onChange(newValue);

    // Check for @ symbol followed by partial filename
    const caretPosition = textareaRef.current?.selectionStart || 0;
    const textBeforeCaret = newValue.substring(0, caretPosition);
    const atMatch = textBeforeCaret.match(/@([^@\s]*)$/);

    if (atMatch) {
      const partial = atMatch[1];
      const atPosition = caretPosition - partial.length - 1;
      setCurrentAtPosition(atPosition);

      // Get file suggestions
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

  // Handle keyboard navigation in suggestions
  const internalHandleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Invoke external handler first to allow prevention/overrides
    if (externalHandleKeyDown) {
      externalHandleKeyDown(e);
      if (e.defaultPrevented) return;
    }
    if (showFileSuggestions && fileSuggestions.length > 0) {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedSuggestionIndex(prev => 
            prev < fileSuggestions.length - 1 ? prev + 1 : 0
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedSuggestionIndex(prev => 
            prev > 0 ? prev - 1 : fileSuggestions.length - 1
          );
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

  // Insert selected file suggestion
  const insertFileSuggestion = (suggestion: FileSuggestion) => {
    const caretPosition = textareaRef.current?.selectionStart || 0;
    const textBeforeCaret = value.substring(0, currentAtPosition);
    const textAfterCaret = value.substring(caretPosition);
    
    const newValue = textBeforeCaret + suggestion.suggestionText + ' ' + textAfterCaret;
    onChange(newValue);
    setShowFileSuggestions(false);
    setSelectedSuggestionIndex(-1);

    // Set cursor position after the inserted text
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

  // Parse current message to show file references info
  const parseResult = fileReferenceParser.parseFileReferences(value);
  const hasFileReferences = parseResult.references.length > 0;
  const hasUnresolvedReferences = parseResult.hasUnresolvedReferences;

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      {/* File references info */}
      {hasFileReferences && (
        <div style={{
          marginBottom: '8px',
          padding: '8px 12px',
          backgroundColor: hasUnresolvedReferences ? '#fff3cd' : '#d4edda',
          border: `1px solid ${hasUnresolvedReferences ? '#ffeaa7' : '#c3e6cb'}`,
          borderRadius: '4px',
          fontSize: '12px'
        }}>
          <div style={{ marginBottom: '4px', fontWeight: 'bold' }}>
            File References ({parseResult.references.length}):
          </div>
          {parseResult.references.map((ref, index) => (
            <div key={index} style={{ 
              color: ref.resolved ? '#155724' : '#856404',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <span>{ref.resolved ? '✓' : '⚠'}</span>
              <span>{ref.originalText}</span>
              {ref.resolved && <span style={{ color: '#6c757d' }}>→ {ref.filename}</span>}
              {!ref.resolved && <span style={{ color: '#856404', fontStyle: 'italic' }}>
                (file not found)
              </span>}
            </div>
          ))}
        </div>
      )}

      {/* Main input area */}
      <div style={{ position: 'relative' }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => handleInputChange(e.target.value)}
            onKeyDown={internalHandleKeyDown}
            rows={rows}
            disabled={disabled}
            placeholder={placeholder}
            style={{
              flex: 1,
              padding: '10px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '14px',
              resize: 'vertical',
              minHeight: '60px',
              fontFamily: 'inherit'
            }}
          />
          

          <button
            type="button"
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            style={{
              padding: '10px 20px',
              backgroundColor: disabled || !value.trim() ? '#6c757d' : '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: disabled || !value.trim() ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              height: '40px',
              whiteSpace: 'nowrap'
            }}
          >
            Send
          </button>
        </div>

        {/* File suggestions dropdown */}
        {showFileSuggestions && (
          <div style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            backgroundColor: 'white',
            border: '1px solid #ddd',
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            zIndex: 1000,
            maxHeight: '200px',
            overflowY: 'auto'
          }}>
            {fileSuggestions.map((suggestion, index) => (
              <div
                key={suggestion.file_id}
                onClick={() => insertFileSuggestion(suggestion)}
                style={{
                  padding: '8px 12px',
                  cursor: 'pointer',
                  backgroundColor: index === selectedSuggestionIndex ? '#e3f2fd' : 'white',
                  borderBottom: index < fileSuggestions.length - 1 ? '1px solid #eee' : 'none'
                }}
                onMouseEnter={() => setSelectedSuggestionIndex(index)}
              >
                <div style={{ fontWeight: 'bold', fontSize: '13px' }}>
                  {suggestion.suggestionText}
                </div>
                <div style={{ fontSize: '11px', color: '#666' }}>
                  {suggestion.original_filename} • {Math.round(suggestion.file_size / 1024)}KB
                </div>
              </div>
            ))}
          </div>
        )}
      </div>


      {/* Help text */}
      <div style={{ marginTop: '4px', fontSize: '11px', color: '#6c757d' }}>
        Tips: Use @ to reference files. Press Tab or Enter to accept suggestions.
      </div>
    </div>
  );
};

export default EnhancedChatInput;