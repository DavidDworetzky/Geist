interface FileReference {
  type: 'file';
  fileId?: number;
  filename?: string;
  startIndex: number;
  endIndex: number;
  originalText: string;
  resolved: boolean;
}

interface FileItem {
  file_id: number;
  filename: string;
  original_filename: string;
  file_size: number;
  mime_type: string;
  upload_date: string;
  is_processed: boolean;
  processing_error?: string;
}

interface ParseResult {
  references: FileReference[];
  plainText: string;
  hasUnresolvedReferences: boolean;
}

interface FileContext {
  fileId: number;
  filename: string;
  content: string;
  characterLimit?: number;
}

class FileReferenceParser {
  private files: FileItem[] = [];
  private fileContentCache: Map<number, string> = new Map();

  constructor() {
    this.loadAvailableFiles();
  }

  /**
   * Load available files from the API
   */
  async loadAvailableFiles(): Promise<void> {
    try {
      const response = await fetch('/api/v1/files/');
      if (response.ok) {
        const data = await response.json();
        this.files = (data.files || []).filter((f: FileItem) => f.is_processed);
      }
    } catch (error) {
      console.error('Failed to load available files:', error);
    }
  }

  /**
   * Parse text for file references using @ syntax
   * Supports formats:
   * - @filename.ext
   * - @file:123 (by file ID)
   * - @"filename with spaces.ext"
   */
  parseFileReferences(text: string): ParseResult {
    const references: FileReference[] = [];
    
    // Pattern to match @filename, @file:id, or @"filename with spaces"
    const patterns = [
      /@file:(\d+)/g,                           // @file:123
      /@"([^"]+)"/g,                            // @"filename with spaces.ext"
      /@([a-zA-Z0-9._-]+\.[a-zA-Z0-9]+)/g      // @filename.ext
    ];

    let plainText = text;
    
    patterns.forEach((pattern, patternIndex) => {
      let match;
      while ((match = pattern.exec(text)) !== null) {
        const fullMatch = match[0];
        const startIndex = match.index;
        const endIndex = startIndex + fullMatch.length;
        
        let fileRef: FileReference = {
          type: 'file',
          startIndex,
          endIndex,
          originalText: fullMatch,
          resolved: false
        };

        if (patternIndex === 0) {
          // @file:id format
          const fileId = parseInt(match[1]);
          const file = this.findFileById(fileId);
          if (file) {
            fileRef.fileId = fileId;
            fileRef.filename = file.original_filename;
            fileRef.resolved = true;
          }
        } else {
          // Filename-based formats
          const filename = match[1];
          const file = this.findFileByName(filename);
          if (file) {
            fileRef.fileId = file.file_id;
            fileRef.filename = file.original_filename;
            fileRef.resolved = true;
          } else {
            fileRef.filename = filename;
          }
        }

        references.push(fileRef);
      }
    });

    // Sort references by start index (descending) to replace from end to beginning
    references.sort((a, b) => b.startIndex - a.startIndex);

    // Remove file references from plain text
    references.forEach(ref => {
      plainText = plainText.substring(0, ref.startIndex) + plainText.substring(ref.endIndex);
    });

    // Re-sort by start index (ascending) for proper order
    references.sort((a, b) => a.startIndex - b.startIndex);

    const hasUnresolvedReferences = references.some(ref => !ref.resolved);

    return {
      references,
      plainText: plainText.trim(),
      hasUnresolvedReferences
    };
  }

  /**
   * Get file suggestions for autocomplete based on partial input
   */
  getFileSuggestions(partial: string): FileItem[] {
    const query = partial.toLowerCase().replace(/^@/, '');
    
    if (!query) {
      return this.files.slice(0, 10); // Return first 10 files
    }

    return this.files
      .filter(file => 
        file.original_filename.toLowerCase().includes(query) ||
        file.filename.toLowerCase().includes(query)
      )
      .slice(0, 10);
  }

  /**
   * Resolve file content for references
   */
  async resolveFileContexts(
    references: FileReference[], 
    characterLimit: number = 2000
  ): Promise<FileContext[]> {
    const contexts: FileContext[] = [];
    
    for (const ref of references) {
      if (ref.resolved && ref.fileId) {
        try {
          let content = this.fileContentCache.get(ref.fileId);

          if (!content) {
            const response = await fetch(`/api/v1/files/${ref.fileId}/content`);
            if (response.ok) {
              const data = await response.json();
              content = data.extracted_text || '';
              this.fileContentCache.set(ref.fileId, content || '');
            } else {
              content = `[Error loading file content]`;
              this.fileContentCache.set(ref.fileId, content);
            }
          }

          // Ensure content is defined
          if (!content) {
            content = '';
          }

          // Truncate content if it exceeds character limit
          if (content.length > characterLimit) {
            content = content.substring(0, characterLimit) + '\n\n[Content truncated...]';
          }

          contexts.push({
            fileId: ref.fileId,
            filename: ref.filename || 'Unknown file',
            content,
            characterLimit
          });
        } catch (error) {
          console.error(`Failed to load content for file ${ref.fileId}:`, error);
          contexts.push({
            fileId: ref.fileId,
            filename: ref.filename || 'Unknown file',
            content: `[Error loading file content: ${error}]`
          });
        }
      }
    }

    return contexts;
  }

  /**
   * Build enhanced prompt with file context
   */
  buildPromptWithFileContext(
    originalPrompt: string,
    fileContexts: FileContext[]
  ): string {
    if (fileContexts.length === 0) {
      return originalPrompt;
    }

    let enhancedPrompt = originalPrompt;

    // Add file contexts at the beginning of the prompt
    const fileContextSection = fileContexts
      .map(context => 
        `\n--- File: ${context.filename} ---\n${context.content}\n--- End of ${context.filename} ---`
      )
      .join('\n');

    enhancedPrompt = `${fileContextSection}\n\nUser Query: ${originalPrompt}`;

    return enhancedPrompt;
  }

  /**
   * Process complete message with file references
   */
  async processMessage(
    message: string, 
    characterLimit: number = 2000
  ): Promise<{
    originalMessage: string;
    enhancedMessage: string;
    references: FileReference[];
    contexts: FileContext[];
    hasUnresolvedReferences: boolean;
  }> {
    await this.loadAvailableFiles(); // Refresh file list
    
    const parseResult = this.parseFileReferences(message);
    const contexts = await this.resolveFileContexts(parseResult.references, characterLimit);
    const enhancedMessage = this.buildPromptWithFileContext(parseResult.plainText, contexts);

    return {
      originalMessage: message,
      enhancedMessage,
      references: parseResult.references,
      contexts,
      hasUnresolvedReferences: parseResult.hasUnresolvedReferences
    };
  }

  /**
   * Validate file reference format
   */
  validateFileReference(text: string): {
    isValid: boolean;
    suggestions: string[];
  } {
    const fileIdPattern = /^@file:\d+$/;
    const quotedFilenamePattern = /^@"[^"]+"$/;
    const filenamePattern = /^@[a-zA-Z0-9._-]+\.[a-zA-Z0-9]+$/;

    const isValid = fileIdPattern.test(text) || 
                   quotedFilenamePattern.test(text) || 
                   filenamePattern.test(text);

    const suggestions = [];
    if (!isValid) {
      suggestions.push(
        'Use @filename.ext for simple filenames',
        'Use @"filename with spaces.ext" for filenames with spaces',
        'Use @file:123 to reference by file ID'
      );
    }

    return { isValid, suggestions };
  }

  /**
   * Generate file reference string
   */
  generateFileReference(file: FileItem): string {
    // Use quoted format if filename contains spaces or special characters
    if (file.original_filename.includes(' ') || 
        /[^a-zA-Z0-9._-]/.test(file.original_filename)) {
      return `@"${file.original_filename}"`;
    }
    
    return `@${file.original_filename}`;
  }

  private findFileById(fileId: number): FileItem | undefined {
    return this.files.find(file => file.file_id === fileId);
  }

  private findFileByName(filename: string): FileItem | undefined {
    return this.files.find(file => 
      file.original_filename === filename || 
      file.filename === filename
    );
  }

  /**
   * Clear file content cache
   */
  clearCache(): void {
    this.fileContentCache.clear();
  }

  /**
   * Get cached file content
   */
  getCachedContent(fileId: number): string | undefined {
    return this.fileContentCache.get(fileId);
  }

  /**
   * Get available files
   */
  getAvailableFiles(): FileItem[] {
    return this.files;
  }
}

// Export singleton instance
export const fileReferenceParser = new FileReferenceParser();

// Export types for use in other components
export type { FileReference, FileItem, ParseResult, FileContext };