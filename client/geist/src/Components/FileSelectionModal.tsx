import React, { useState, useEffect } from 'react';

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

interface FileSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onFilesSelected: (files: FileItem[]) => void;
  multiple?: boolean;
  title?: string;
}

const FileSelectionModal: React.FC<FileSelectionModalProps> = ({
  isOpen,
  onClose,
  onFilesSelected,
  multiple = false,
  title = 'Select Files'
}) => {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [filteredFiles, setFilteredFiles] = useState<FileItem[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<Set<number>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewFile, setPreviewFile] = useState<FileItem | null>(null);
  const [previewContent, setPreviewContent] = useState<string | null>(null);

  const fetchFiles = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/v1/files/');

      if (!response.ok) {
        throw new Error('Failed to fetch files');
      }

      const data = await response.json();
      const processedFiles = (data.files || []).filter((f: FileItem) => f.is_processed);
      setFiles(processedFiles);
      setFilteredFiles(processedFiles);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (!query.trim()) {
      setFilteredFiles(files);
      return;
    }

    const lowercaseQuery = query.toLowerCase();
    const filtered = files.filter(file =>
      file.filename.toLowerCase().includes(lowercaseQuery) ||
      file.original_filename.toLowerCase().includes(lowercaseQuery) ||
      file.mime_type.toLowerCase().includes(lowercaseQuery)
    );
    setFilteredFiles(filtered);
  };

  const handleFileSelection = (file: FileItem) => {
    if (multiple) {
      const newSelected = new Set(selectedFiles);
      if (newSelected.has(file.file_id)) {
        newSelected.delete(file.file_id);
      } else {
        newSelected.add(file.file_id);
      }
      setSelectedFiles(newSelected);
    } else {
      setSelectedFiles(new Set([file.file_id]));
    }
  };

  const handlePreview = async (file: FileItem) => {
    if (previewFile?.file_id === file.file_id) {
      setPreviewFile(null);
      setPreviewContent(null);
      return;
    }

    try {
      setPreviewFile(file);
      setPreviewContent('Loading...');

      const response = await fetch(`/api/v1/files/${file.file_id}/content`);

      if (!response.ok) {
        throw new Error('Failed to fetch file content');
      }

      const data = await response.json();
      setPreviewContent(data.extracted_text || 'No text content available');
    } catch (err) {
      setPreviewContent(`Error loading content: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const handleConfirm = () => {
    const selected = files.filter(file => selectedFiles.has(file.file_id));
    onFilesSelected(selected);
    handleClose();
  };

  const handleClose = () => {
    setSelectedFiles(new Set());
    setSearchQuery('');
    setPreviewFile(null);
    setPreviewContent(null);
    setError(null);
    onClose();
  };

  const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round((bytes / Math.pow(1024, i)) * 100) / 100} ${sizes[i]}`;
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString();
  };

  useEffect(() => {
    if (isOpen) {
      void fetchFiles();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop">
      <div className="file-selection-modal" role="dialog" aria-modal="true" aria-labelledby="file-selection-title">
        <header className="file-selection-header">
          <h2 id="file-selection-title">{title}</h2>
          <button className="icon-action" type="button" onClick={handleClose} aria-label="Close file selector">
            X
          </button>
        </header>

        <div className="file-selection-search">
          <input
            className="form-control"
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search files..."
          />
        </div>

        <div className={`file-selection-content ${previewFile ? 'with-preview' : ''}`}>
          <div className="file-selection-list">
            {loading ? (
              <div className="empty-state compact">Loading files...</div>
            ) : error ? (
              <div className="notice notice-error">{error}</div>
            ) : filteredFiles.length === 0 ? (
              <div className="empty-state compact">
                {searchQuery ? 'No files found matching your search.' : 'No processed files available.'}
              </div>
            ) : (
              filteredFiles.map((file) => {
                const selected = selectedFiles.has(file.file_id);
                const previewing = previewFile?.file_id === file.file_id;
                return (
                  <article
                    key={file.file_id}
                    className={`file-option-card ${selected ? 'selected' : ''}`}
                    onClick={() => handleFileSelection(file)}
                  >
                    <input
                      type={multiple ? 'checkbox' : 'radio'}
                      checked={selected}
                      readOnly
                      aria-label={`Select ${file.original_filename}`}
                    />
                    <div className="file-option-copy">
                      <h4>{file.original_filename}</h4>
                      <p>{formatFileSize(file.file_size)} | {file.mime_type} | {formatDate(file.upload_date)}</p>
                    </div>
                    <button
                      type="button"
                      className={`button button-small ${previewing ? 'button-danger' : 'button-secondary'}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handlePreview(file);
                      }}
                    >
                      {previewing ? 'Hide' : 'Preview'}
                    </button>
                  </article>
                );
              })
            )}
          </div>

          {previewFile && (
            <aside className="file-selection-preview">
              <h3>Preview: {previewFile.original_filename}</h3>
              <pre>{previewContent}</pre>
            </aside>
          )}
        </div>

        <footer className="file-selection-footer">
          <span className="settings-description">
            {selectedFiles.size > 0 ? `${selectedFiles.size} file${selectedFiles.size === 1 ? '' : 's'} selected` : 'No files selected'}
          </span>
          <div className="modal-actions">
            <button className="button button-secondary" onClick={handleClose}>Cancel</button>
            <button className="button" onClick={handleConfirm} disabled={selectedFiles.size === 0}>Select</button>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default FileSelectionModal;
