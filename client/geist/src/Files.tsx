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
      fetchFiles();
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
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete file');
      }
      
      // Refresh file list
      fetchFiles();
      
      // Clear selected file if it was deleted
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
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString() + ' ' + new Date(dateString).toLocaleTimeString();
  };

  const handleUploadComplete = () => {
    fetchFiles();
    setShowUpload(false);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    searchFiles(searchQuery);
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ marginBottom: '30px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '20px' 
        }}>
          <h1 style={{ margin: '0', color: '#333' }}>File Management</h1>
          <button
            onClick={() => setShowUpload(!showUpload)}
            style={{
              padding: '10px 20px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
          >
            {showUpload ? 'Hide Upload' : 'Upload Files'}
          </button>
        </div>

        {showUpload && (
          <div style={{ marginBottom: '30px' }}>
            <FileUpload
              onUploadComplete={handleUploadComplete}
              onUploadError={(error) => setError(error)}
              multiple={true}
            />
          </div>
        )}

        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search files by name or content..."
            style={{
              flex: 1,
              padding: '10px',
              border: '1px solid #ddd',
              borderRadius: '5px',
              fontSize: '14px'
            }}
          />
          <button
            type="submit"
            style={{
              padding: '10px 20px',
              backgroundColor: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            Search
          </button>
          {searchQuery && (
            <button
              type="button"
              onClick={() => {
                setSearchQuery('');
                fetchFiles();
              }}
              style={{
                padding: '10px 15px',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              Clear
            </button>
          )}
        </form>
      </div>

      {error && (
        <div style={{
          padding: '12px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderRadius: '5px',
          marginBottom: '20px'
        }}>
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#6c757d' }}>
          Loading files...
        </div>
      ) : files.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#6c757d' }}>
          {searchQuery ? 'No files found matching your search.' : 'No files uploaded yet.'}
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: selectedFile ? '1fr 1fr' : '1fr', gap: '20px' }}>
          <div>
            <h2 style={{ marginBottom: '15px', color: '#333' }}>
              Files ({files.length})
            </h2>
            <div style={{ maxHeight: '70vh', overflowY: 'auto' }}>
              {files.map((file) => (
                <div
                  key={file.file_id}
                  style={{
                    border: '1px solid #ddd',
                    borderRadius: '5px',
                    padding: '15px',
                    marginBottom: '10px',
                    backgroundColor: selectedFile?.file_id === file.file_id ? '#e3f2fd' : 'white'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                    <div style={{ flex: 1 }}>
                      <h3 style={{ margin: '0 0 5px 0', color: '#333', fontSize: '16px' }}>
                        {file.original_filename}
                      </h3>
                      <p style={{ margin: '0', fontSize: '12px', color: '#6c757d' }}>
                        Size: {formatFileSize(file.file_size)} | 
                        Type: {file.mime_type} | 
                        Uploaded: {formatDate(file.upload_date)}
                      </p>
                      <p style={{ margin: '5px 0 0 0', fontSize: '12px' }}>
                        Status: <span style={{
                          color: file.is_processed ? '#28a745' : '#ffc107',
                          fontWeight: 'bold'
                        }}>
                          {file.is_processed ? 'Processed' : 'Processing...'}
                        </span>
                        {file.processing_error && (
                          <span style={{ color: '#dc3545', marginLeft: '10px' }}>
                            Error: {file.processing_error}
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    <button
                      onClick={() => viewFileContent(file)}
                      style={{
                        padding: '5px 10px',
                        backgroundColor: selectedFile?.file_id === file.file_id ? '#dc3545' : '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      {selectedFile?.file_id === file.file_id ? 'Hide Content' : 'View Content'}
                    </button>
                    
                    <button
                      onClick={() => downloadFile(file.file_id, file.original_filename)}
                      style={{
                        padding: '5px 10px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      Download
                    </button>
                    
                    <button
                      onClick={() => handleDeleteClick(file.file_id)}
                      style={{
                        padding: '5px 10px',
                        backgroundColor: '#dc3545',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedFile && (
            <div>
              <h2 style={{ marginBottom: '15px', color: '#333' }}>
                Content Preview: {selectedFile.original_filename}
              </h2>
              <div style={{
                border: '1px solid #ddd',
                borderRadius: '5px',
                padding: '15px',
                backgroundColor: 'white',
                maxHeight: '70vh',
                overflowY: 'auto'
              }}>
                {fileContent ? (
                  <pre style={{ 
                    whiteSpace: 'pre-wrap', 
                    fontSize: '12px', 
                    lineHeight: '1.4',
                    margin: '0',
                    fontFamily: 'monospace'
                  }}>
                    {fileContent}
                  </pre>
                ) : (
                  <p style={{ color: '#6c757d', fontStyle: 'italic' }}>
                    Loading content...
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {deleteConfirm.show && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            maxWidth: '400px'
          }}>
            <h3 style={{ margin: '0 0 15px 0' }}>Confirm Delete</h3>
            <p style={{ margin: '0 0 20px 0' }}>Are you sure you want to delete this file?</p>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={cancelDelete}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#dc3545',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Files;