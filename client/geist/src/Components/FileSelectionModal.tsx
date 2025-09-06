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
      // Single selection - replace current selection
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
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString();
  };

  useEffect(() => {
    if (isOpen) {
      fetchFiles();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'white',
        borderRadius: '8px',
        width: '90%',
        maxWidth: '1000px',
        height: '80%',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
      }}>
        {/* Header */}
        <div style={{
          padding: '20px',
          borderBottom: '1px solid #e0e0e0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h2 style={{ margin: 0, color: '#333' }}>{title}</h2>
          <button
            onClick={handleClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '24px',
              cursor: 'pointer',
              color: '#666'
            }}
          >
            ×
          </button>
        </div>

        {/* Search */}
        <div style={{ padding: '15px 20px', borderBottom: '1px solid #e0e0e0' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search files..."
            style={{
              width: '100%',
              padding: '8px 12px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '14px'
            }}
          />
        </div>

        {/* Content */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* File List */}
          <div style={{ 
            flex: previewFile ? '1' : '2', 
            padding: '15px 20px', 
            overflowY: 'auto',
            borderRight: previewFile ? '1px solid #e0e0e0' : 'none'
          }}>
            {loading ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                Loading files...
              </div>
            ) : error ? (
              <div style={{
                padding: '12px',
                backgroundColor: '#f8d7da',
                color: '#721c24',
                border: '1px solid #f5c6cb',
                borderRadius: '4px',
                marginBottom: '15px'
              }}>
                {error}
              </div>
            ) : filteredFiles.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                {searchQuery ? 'No files found matching your search.' : 'No processed files available.'}
              </div>
            ) : (
              filteredFiles.map((file) => (
                <div
                  key={file.file_id}
                  style={{
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    padding: '12px',
                    marginBottom: '8px',
                    cursor: 'pointer',
                    backgroundColor: selectedFiles.has(file.file_id) ? '#e3f2fd' : 'white',
                    transition: 'background-color 0.2s'
                  }}
                  onClick={() => handleFileSelection(file)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                    <input
                      type={multiple ? 'checkbox' : 'radio'}
                      checked={selectedFiles.has(file.file_id)}
                      onChange={() => {}} // Handled by div click
                      style={{ marginRight: '10px' }}
                    />
                    <div style={{ flex: 1 }}>
                      <h4 style={{ margin: '0 0 4px 0', fontSize: '14px', color: '#333' }}>
                        {file.original_filename}
                      </h4>
                      <p style={{ margin: 0, fontSize: '12px', color: '#666' }}>
                        {formatFileSize(file.file_size)} • {file.mime_type} • {formatDate(file.upload_date)}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePreview(file);
                      }}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: previewFile?.file_id === file.file_id ? '#dc3545' : '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: 'pointer',
                        fontSize: '11px'
                      }}
                    >
                      {previewFile?.file_id === file.file_id ? 'Hide' : 'Preview'}
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Preview Panel */}
          {previewFile && (
            <div style={{ flex: '1', padding: '15px 20px', overflowY: 'auto' }}>
              <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', color: '#333' }}>
                Preview: {previewFile.original_filename}
              </h3>
              <div style={{
                backgroundColor: '#f8f9fa',
                border: '1px solid #e0e0e0',
                borderRadius: '4px',
                padding: '15px',
                fontSize: '12px',
                fontFamily: 'monospace',
                whiteSpace: 'pre-wrap',
                maxHeight: '100%',
                overflow: 'auto'
              }}>
                {previewContent}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '15px 20px',
          borderTop: '1px solid #e0e0e0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ fontSize: '14px', color: '#666' }}>
            {selectedFiles.size > 0 ? `${selectedFiles.size} file${selectedFiles.size === 1 ? '' : 's'} selected` : 'No files selected'}
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={handleClose}
              style={{
                padding: '8px 16px',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={selectedFiles.size === 0}
              style={{
                padding: '8px 16px',
                backgroundColor: selectedFiles.size === 0 ? '#ccc' : '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: selectedFiles.size === 0 ? 'not-allowed' : 'pointer',
                fontSize: '14px'
              }}
            >
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileSelectionModal;