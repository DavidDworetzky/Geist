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
    void fetchFiles();
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
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>RAG & File Settings</h3>
        <p>Choose the archives that should be available to new chats by default.</p>
      </header>

      <SettingsToggle
        label="Enable RAG by Default"
        checked={enableRagByDefault}
        onChange={onEnableRagChange}
        description="Automatically use Retrieval-Augmented Generation for new conversations."
      />

      <div className="settings-subsection">
        <div className="settings-subsection-header">
          <div>
            <span className="settings-label">Default File Archives</span>
            <p className="settings-description">
              Select files to include in RAG context by default ({defaultFileArchives.length} selected).
            </p>
          </div>
          <div className="settings-inline-actions">
            <button className="button button-secondary button-small" onClick={selectAll} disabled={loading || files.length === 0}>
              Select All
            </button>
            <button className="button button-secondary button-small" onClick={clearAll} disabled={loading}>
              Clear All
            </button>
          </div>
        </div>

        {loading ? (
          <div className="empty-state compact">Loading files...</div>
        ) : files.length === 0 ? (
          <div className="empty-state compact">No files uploaded yet. Upload files in the Files page to use RAG.</div>
        ) : (
          <div className="settings-file-list">
            {files.map((file) => {
              const selected = defaultFileArchives.includes(file.file_id);
              return (
                <button
                  key={file.file_id}
                  type="button"
                  className={`settings-file-option ${selected ? 'selected' : ''}`}
                  onClick={() => toggleFileSelection(file.file_id)}
                >
                  <span className="settings-checkbox" aria-hidden="true">

                  </span>
                  <span>{file.original_filename}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
};

export default RAGSettingsSection;
