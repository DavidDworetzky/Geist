import React, { useState, useEffect } from 'react';
import FileUpload from './Components/FileUpload';

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

interface FileListResponse {
  success: boolean;
  files: FileItem[];
  total: number;
  skip: number;
  limit: number;
}

const Files: React.FC = () => {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFile, setSelectedFile] = useState<FileItem | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<{ show: boolean; fileId: number | null }>({ show: false, fileId: null });

  const fetchFiles = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/files/');

      if (!response.ok) {
        throw new Error('Failed to fetch files');
      }

      const data: FileListResponse = await response.json();
      setFiles(data.files || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const searchFiles = async (query: string) => {
    if (!query.trim()) {
      void fetchFiles();
      return;
    }

    try {
      setLoading(true);
      const response = await fetch(`/api/v1/files/search?query=${encodeURIComponent(query)}`);

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setFiles(data.files || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (fileId: number) => {
    setDeleteConfirm({ show: true, fileId });
  };

  const confirmDelete = async () => {
    if (deleteConfirm.fileId) {
      await deleteFile(deleteConfirm.fileId);
    }
    setDeleteConfirm({ show: false, fileId: null });
  };

  const cancelDelete = () => {
    setDeleteConfirm({ show: false, fileId: null });
  };

  const deleteFile = async (fileId: number) => {
    try {
      const response = await fetch(`/api/v1/files/${fileId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to delete file');
      }

      void fetchFiles();

      if (selectedFile?.file_id === fileId) {
        setSelectedFile(null);
        setFileContent(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete file');
    }
  };

  const downloadFile = async (fileId: number, filename: string) => {
    try {
      const response = await fetch(`/api/v1/files/${fileId}/download`);

      if (!response.ok) {
        throw new Error('Failed to download file');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download file');
    }
  };

  const viewFileContent = async (file: FileItem) => {
    if (selectedFile?.file_id === file.file_id) {
      setSelectedFile(null);
      setFileContent(null);
      return;
    }

    try {
      setFileContent(null);
      const response = await fetch(`/api/v1/files/${file.file_id}/content`);

      if (!response.ok) {
        throw new Error('Failed to fetch file content');
      }

      const data = await response.json();
      setSelectedFile(file);
      setFileContent(data.extracted_text || 'No text content available');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load file content');
    }
  };

  const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round((bytes / Math.pow(1024, i)) * 100) / 100} ${sizes[i]}`;
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
  };

  const handleUploadComplete = () => {
    void fetchFiles();
    setShowUpload(false);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    void searchFiles(searchQuery);
  };

  useEffect(() => {
    void fetchFiles();
  }, []);

  return (
    <div className="files-page page-surface">
      <header className="page-header">
        <div>
          <p className="section-eyebrow">Workspace</p>
          <h2>Files</h2>
          <p>Upload, search, preview, and manage files available to chats.</p>
        </div>
        <button className="button" onClick={() => setShowUpload(!showUpload)}>
          {showUpload ? 'Hide Upload' : 'Upload Files'}
        </button>
      </header>

      {showUpload && (
        <section className="file-upload-panel">
          <FileUpload
            onUploadComplete={handleUploadComplete}
            onUploadError={(uploadError) => setError(uploadError)}
            multiple={true}
          />
        </section>
      )}

      <form className="file-search" onSubmit={handleSearch}>
        <input
          className="form-control"
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search files by name or content..."
        />
        <button className="button" type="submit">Search</button>
        {searchQuery && (
          <button
            className="button button-secondary"
            type="button"
            onClick={() => {
              setSearchQuery('');
              void fetchFiles();
            }}
          >
            Clear
          </button>
        )}
      </form>

      {error && <div className="notice notice-error">{error}</div>}

      {loading ? (
        <div className="empty-state">Loading files...</div>
      ) : files.length === 0 ? (
        <div className="empty-state">{searchQuery ? 'No files found matching your search.' : 'No files uploaded yet.'}</div>
      ) : (
        <div className={`files-grid ${selectedFile ? 'with-preview' : ''}`}>
          <section className="file-list-panel">
            <div className="panel-title-row">
              <h3>Files ({files.length})</h3>
            </div>
            <div className="file-list">
              {files.map((file) => {
                const selected = selectedFile?.file_id === file.file_id;
                return (
                  <article key={file.file_id} className={`file-card ${selected ? 'selected' : ''}`}>
                    <div className="file-card-main">
                      <h4>{file.original_filename}</h4>
                      <p>
                        {formatFileSize(file.file_size)} | {file.mime_type} | {formatDate(file.upload_date)}
                      </p>
                      <div className="file-status-row">
                        <span className={`status-badge ${file.is_processed ? 'success' : 'warning'}`}>
                          {file.is_processed ? 'Processed' : 'Processing'}
                        </span>
                        {file.processing_error && <span className="status-badge danger">{file.processing_error}</span>}
                      </div>
                    </div>

                    <div className="file-actions">
                      <button className={`button button-small ${selected ? 'button-danger' : ''}`} onClick={() => viewFileContent(file)}>
                        {selected ? 'Hide Content' : 'View Content'}
                      </button>
                      <button className="button button-secondary button-small" onClick={() => downloadFile(file.file_id, file.original_filename)}>
                        Download
                      </button>
                      <button className="button button-danger button-small" onClick={() => handleDeleteClick(file.file_id)}>
                        Delete
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>

          {selectedFile && (
            <section className="file-preview-panel">
              <div className="panel-title-row">
                <h3>Content Preview</h3>
                <span>{selectedFile.original_filename}</span>
              </div>
              <div className="file-preview">
                {fileContent ? <pre>{fileContent}</pre> : <p className="settings-description">Loading content...</p>}
              </div>
            </section>
          )}
        </div>
      )}

      {deleteConfirm.show && (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-panel" role="dialog" aria-modal="true" aria-labelledby="delete-file-title">
            <h3 id="delete-file-title">Confirm Delete</h3>
            <p>Are you sure you want to delete this file?</p>
            <div className="modal-actions">
              <button className="button button-secondary" onClick={cancelDelete}>Cancel</button>
              <button className="button button-danger" onClick={confirmDelete}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Files;
