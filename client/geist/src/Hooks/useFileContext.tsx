import { useState, useCallback } from 'react';
import { fileReferenceParser, FileContext, FileReference } from '../Utils/fileReferenceParser';

interface FileContextHook {
  processMessage: (message: string) => Promise<ProcessedMessage>;
  isProcessing: boolean;
  error: string | null;
  clearCache: () => void;
}

interface ProcessedMessage {
  originalMessage: string;
  enhancedMessage: string;
  references: FileReference[];
  contexts: FileContext[];
  hasUnresolvedReferences: boolean;
}

const useFileContext = (characterLimit: number = 2000): FileContextHook => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processMessage = useCallback(async (message: string): Promise<ProcessedMessage> => {
    setIsProcessing(true);
    setError(null);

    try {
      const result = await fileReferenceParser.processMessage(message, characterLimit);
      
      return {
        originalMessage: result.originalMessage,
        enhancedMessage: result.enhancedMessage,
        references: result.references,
        contexts: result.contexts,
        hasUnresolvedReferences: result.hasUnresolvedReferences
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process file references';
      setError(errorMessage);
      
      // Return original message if processing fails
      return {
        originalMessage: message,
        enhancedMessage: message,
        references: [],
        contexts: [],
        hasUnresolvedReferences: false
      };
    } finally {
      setIsProcessing(false);
    }
  }, [characterLimit]);

  const clearCache = useCallback(() => {
    fileReferenceParser.clearCache();
  }, []);

  return {
    processMessage,
    isProcessing,
    error,
    clearCache
  };
};

export default useFileContext;