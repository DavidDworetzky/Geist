import React, { useState, useEffect } from 'react';
import SettingsToggle from './SettingsToggle';

interface RAGSettingsSectionProps {
  enableRagByDefault: boolean;
  defaultFileArchives: number[];
  onEnableRagChange: (value: boolean) => void;
  onFileArchivesChange: (value: number[]) => void;
}

interface FileItem {
  file_id: number;
  filename: string;
  original_filename: string;
}

const RAGSettingsSection: React.FC<RAGSettingsSectionProps> = ({
  enableRagByDefault,
  defaultFileArchives,
  onEnableRagChange,
  onFileArchivesChange
}) => {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await fetch('/api/v1/files/');
      if (response.ok) {
        const data = await response.json();
        setFiles(data.files || []);
      }
    } catch (err) {
      console.error('Failed to fetch files:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleFileSelection = (fileId: number) => {
    if (defaultFileArchives.includes(fileId)) {
      onFileArchivesChange(defaultFileArchives.filter(id => id !== fileId));
    } else {
      onFileArchivesChange([...defaultFileArchives, fileId]);
    }
  };

  const selectAll = () => {
    onFileArchivesChange(files.map(f => f.file_id));
  };

  const clearAll = () => {
    onFileArchivesChange([]);
  };

  return (
    <div style={{
      backgroundColor: 'white',
      padding: '25px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      marginBottom: '20px'
    }}>
      <h3 style={{ 
        margin: '0 0 20px 0', 
        color: '#333', 
        fontSize: '18px',
        borderBottom: '2px solid #007bff',
        paddingBottom: '10px'
      }}>
        RAG & File Settings
      </h3>

      <SettingsToggle
        label="Enable RAG by Default"
        checked={enableRagByDefault}
        onChange={onEnableRagChange}
        description="Automatically use Retrieval-Augmented Generation for new conversations"
      />

      <div style={{ marginTop: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <label style={{ fontWeight: '500', color: '#333', fontSize: '14px' }}>
            Default File Archives
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={selectAll}
              disabled={loading || files.length === 0}
              style={{
                padding: '4px 10px',
                fontSize: '12px',
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading || files.length === 0 ? 'not-allowed' : 'pointer',
                opacity: loading || files.length === 0 ? 0.5 : 1
              }}
            >
              Select All
            </button>
            <button
              onClick={clearAll}
              disabled={loading}
              style={{
                padding: '4px 10px',
                fontSize: '12px',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.5 : 1
              }}
            >
              Clear All
            </button>
          </div>
        </div>
        
        <p style={{ fontSize: '12px', color: '#6c757d', margin: '0 0 12px 0' }}>
          Select files to include in RAG context by default ({defaultFileArchives.length} selected)
        </p>

        {loading ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>
            Loading files...
          </div>
        ) : files.length === 0 ? (
          <div style={{ 
            padding: '20px', 
            textAlign: 'center', 
            color: '#6c757d',
            backgroundColor: '#f8f9fa',
            borderRadius: '5px',
            border: '1px dashed #ddd'
          }}>
            No files uploaded yet. Upload files in the Files page to use RAG.
          </div>
        ) : (
          <div style={{
            maxHeight: '300px',
            overflowY: 'auto',
            border: '1px solid #ddd',
            borderRadius: '5px',
            padding: '10px'
          }}>
            {files.map((file) => (
              <div
                key={file.file_id}
                onClick={() => toggleFileSelection(file.file_id)}
                style={{
                  padding: '10px',
                  marginBottom: '5px',
                  backgroundColor: defaultFileArchives.includes(file.file_id) ? '#e3f2fd' : 'white',
                  border: `1px solid ${defaultFileArchives.includes(file.file_id) ? '#007bff' : '#ddd'}`,
                  borderRadius: '4px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  transition: 'all 0.2s'
                }}
              >
                <div style={{
                  width: '18px',
                  height: '18px',
                  borderRadius: '3px',
                  border: `2px solid ${defaultFileArchives.includes(file.file_id) ? '#007bff' : '#6c757d'}`,
                  backgroundColor: defaultFileArchives.includes(file.file_id) ? '#007bff' : 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0
                }}>
                  {defaultFileArchives.includes(file.file_id) && (
                    <span style={{ color: 'white', fontSize: '12px', fontWeight: 'bold' }}>✓</span>
                  )}
                </div>
                <span style={{ fontSize: '14px', color: '#333' }}>
                  {file.original_filename}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default RAGSettingsSection;

