import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { waitForFileJob } from '../Utils/waitForFileJob';

interface FileUploadProps {
  onFileUpload?: (file: File) => void;
  onUploadComplete?: (response: any) => void;
  onUploadError?: (error: string) => void;
  multiple?: boolean;
  maxSize?: number;
  acceptedFileTypes?: string[];
}

interface UploadStatus {
  status: 'idle' | 'uploading' | 'processing' | 'pending' | 'success' | 'error';
  progress: number;
  message: string;
  fileName?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFileUpload,
  onUploadComplete,
  onUploadError,
  multiple = false,
  maxSize = 50 * 1024 * 1024,
  acceptedFileTypes = ['.pdf', '.txt', '.docx', '.xlsx', '.csv', '.md', '.json', '.xml']
}) => {
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({
    status: 'idle',
    progress: 0,
    message: ''
  });

  const uploadController = useRef<AbortController | null>(null);
  const resetTimer = useRef<number | undefined>();
  useEffect(() => () => {
    uploadController.current?.abort();
    window.clearTimeout(resetTimer.current);
  }, []);

  const uploadFile = useCallback(async (file: File) => {
    uploadController.current?.abort();
    window.clearTimeout(resetTimer.current);
    const controller = new AbortController();
    uploadController.current = controller;

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
        signal: controller.signal,
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      if (controller.signal.aborted) return;
      if (result.processing_status?.status === 'failed') {
        throw new Error(result.processing_status.error || 'File processing failed.');
      }
      const jobId = result.processing_status?.job_id;
      if (typeof jobId === 'number') {
        setUploadStatus({ status: 'processing', progress: 100,
          message: `${file.name} uploaded. Processing text...`, fileName: file.name });
        const ready = await waitForFileJob(jobId, controller.signal);
        if (controller.signal.aborted) return;
        if (!ready) {
          setUploadStatus({ status: 'pending', progress: 100,
            message: `${file.name} uploaded. Processing may still be running; check Files later.`,
            fileName: file.name });
          return;
        }
      }
      if (controller.signal.aborted) return;

      setUploadStatus({
        status: 'success',
        progress: 100,
        message: `${file.name} uploaded successfully!`,
        fileName: file.name
      });

      if (onUploadComplete) {
        onUploadComplete(result);
      }

      resetTimer.current = window.setTimeout(() => {
        setUploadStatus({ status: 'idle', progress: 0, message: '' });
      }, 3000);
    } catch (error) {
      if (controller.signal.aborted) return;
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

      resetTimer.current = window.setTimeout(() => {
        setUploadStatus({ status: 'idle', progress: 0, message: '' });
      }, 5000);
    }
  }, [onUploadComplete, onUploadError]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      if (onFileUpload) {
        onFileUpload(file);
      }
      await uploadFile(file);
      if (uploadController.current?.signal.aborted) break;
    }
  }, [onFileUpload, uploadFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple,
    maxSize,
    accept: acceptedFileTypes.reduce((acc, type) => {
      acc[type] = [];
      return acc;
    }, {} as Record<string, string[]>)
  });

  const statusLabel = uploadStatus.status === 'idle' ? 'Upload' : uploadStatus.status;

  return (
    <div className="file-upload">
      <div
        {...getRootProps()}
        className={`file-dropzone ${isDragActive ? 'drag-active' : ''} file-upload-${uploadStatus.status}`}
      >
        <input {...getInputProps()} aria-label="Choose files to upload" />
        <div className="file-upload-status-label">{statusLabel}</div>

        {uploadStatus.status === 'idle' ? (
          <>
            <p className="file-upload-title">{isDragActive ? 'Drop files here' : 'Drag and drop files here'}</p>
            <p className="settings-description">or click to select files</p>
            <p className="settings-description">Supported formats: {acceptedFileTypes.join(', ')}</p>
            <p className="settings-description">Max file size: {Math.round(maxSize / (1024 * 1024))}MB</p>
          </>
        ) : (
          <div className="file-upload-progress-block">
            <p className="file-upload-message">{uploadStatus.message}</p>
            {uploadStatus.status === 'uploading' && (
              <div className="progress-track" aria-label="Upload progress">
                <div className="progress-fill" style={{ width: `${uploadStatus.progress}%` }} />
              </div>
            )}
          </div>
        )}
      </div>

      {uploadStatus.status === 'error' && (
        <div className="notice notice-error file-upload-notice">
          <strong>Upload failed:</strong> {uploadStatus.message}
        </div>
      )}

      {uploadStatus.status === 'success' && (
        <div className="notice notice-success file-upload-notice">
          <strong>Success:</strong> {uploadStatus.fileName} has been uploaded and is being processed.
        </div>
      )}
    </div>
  );
};

export default FileUpload;
