import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onFileUpload?: (file: File) => void;
  onUploadComplete?: (response: any) => void;
  onUploadError?: (error: string) => void;
  multiple?: boolean;
  maxSize?: number;
  acceptedFileTypes?: string[];
}

interface UploadStatus {
  status: 'idle' | 'uploading' | 'success' | 'error';
  progress: number;
  message: string;
  fileName?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFileUpload,
  onUploadComplete,
  onUploadError,
  multiple = false,
  maxSize = 50 * 1024 * 1024, // 50MB default
  acceptedFileTypes = ['.pdf', '.txt', '.docx', '.xlsx', '.csv', '.md', '.json', '.xml']
}) => {
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({
    status: 'idle',
    progress: 0,
    message: ''
  });

  const uploadFile = async (file: File) => {
    setUploadStatus({
      status: 'uploading',
      progress: 0,
      message: `Uploading ${file.name}...`,
      fileName: file.name
    });

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/v1/files/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      
      setUploadStatus({
        status: 'success',
        progress: 100,
        message: `${file.name} uploaded successfully!`,
        fileName: file.name
      });

      if (onUploadComplete) {
        onUploadComplete(result);
      }

      // Reset status after 3 seconds
      setTimeout(() => {
        setUploadStatus({
          status: 'idle',
          progress: 0,
          message: ''
        });
      }, 3000);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      
      setUploadStatus({
        status: 'error',
        progress: 0,
        message: errorMessage,
        fileName: file.name
      });

      if (onUploadError) {
        onUploadError(errorMessage);
      }

      // Reset status after 5 seconds
      setTimeout(() => {
        setUploadStatus({
          status: 'idle',
          progress: 0,
          message: ''
        });
      }, 5000);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      if (onFileUpload) {
        onFileUpload(file);
      }
      await uploadFile(file);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple,
    maxSize,
    accept: acceptedFileTypes.reduce((acc, type) => {
      acc[type] = [];
      return acc;
    }, {} as Record<string, string[]>)
  });

  const getStatusColor = () => {
    switch (uploadStatus.status) {
      case 'uploading':
        return '#007bff';
      case 'success':
        return '#28a745';
      case 'error':
        return '#dc3545';
      default:
        return '#6c757d';
    }
  };

  const getStatusIcon = () => {
    switch (uploadStatus.status) {
      case 'uploading':
        return '⏳';
      case 'success':
        return '✅';
      case 'error':
        return '❌';
      default:
        return '📁';
    }
  };

  return (
    <div style={{ width: '100%', maxWidth: '600px', margin: '0 auto' }}>
      <div
        {...getRootProps()}
        style={{
          border: `2px dashed ${isDragActive ? '#007bff' : '#ccc'}`,
          borderRadius: '8px',
          padding: '40px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? '#f8f9fa' : '#fff',
          transition: 'all 0.3s ease',
          minHeight: '150px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        <input {...getInputProps()} />
        
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>
          {getStatusIcon()}
        </div>
        
        {uploadStatus.status === 'idle' ? (
          <>
            <p style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: 'bold' }}>
              {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
            </p>
            <p style={{ margin: '0', color: '#6c757d' }}>
              or click to select files
            </p>
            <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#6c757d' }}>
              Supported formats: {acceptedFileTypes.join(', ')}
            </p>
            <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#6c757d' }}>
              Max file size: {Math.round(maxSize / (1024 * 1024))}MB
            </p>
          </>
        ) : (
          <div style={{ width: '100%' }}>
            <p style={{ 
              margin: '0 0 16px 0', 
              fontSize: '16px', 
              color: getStatusColor(),
              fontWeight: 'bold'
            }}>
              {uploadStatus.message}
            </p>
            
            {uploadStatus.status === 'uploading' && (
              <div style={{ width: '100%', marginBottom: '16px' }}>
                <div style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: '#e9ecef',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${uploadStatus.progress}%`,
                    height: '100%',
                    backgroundColor: '#007bff',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
                <p style={{ 
                  margin: '8px 0 0 0', 
                  fontSize: '14px', 
                  color: '#6c757d',
                  textAlign: 'center'
                }}>
                  {uploadStatus.progress}%
                </p>
              </div>
            )}
          </div>
        )}
      </div>
      
      {uploadStatus.status === 'error' && (
        <div style={{
          marginTop: '16px',
          padding: '12px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          <strong>Upload failed:</strong> {uploadStatus.message}
        </div>
      )}
      
      {uploadStatus.status === 'success' && (
        <div style={{
          marginTop: '16px',
          padding: '12px',
          backgroundColor: '#d4edda',
          color: '#155724',
          border: '1px solid #c3e6cb',
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          <strong>Success:</strong> {uploadStatus.fileName} has been uploaded and is being processed.
        </div>
      )}
    </div>
  );
};

export default FileUpload;