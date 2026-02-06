import React, { useState, useEffect, useCallback } from 'react';

interface NoteItem {
  note_id: number;
  title: string;
  content: string;
  user_id: number;
  create_date: string;
  update_date: string;
}

// Minimal markdown-to-HTML renderer for preview
function renderMarkdown(md: string): string {
  let html = md;

  // Escape HTML entities
  html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Code blocks (fenced)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, _lang, code) => {
    return `<pre style="background:#f4f4f4;padding:12px;border-radius:4px;overflow-x:auto"><code>${code.trim()}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code style="background:#f4f4f4;padding:2px 4px;border-radius:3px">$1</code>');

  // Headers
  html = html.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
  html = html.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
  html = html.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Strikethrough
  html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');

  // Blockquotes
  html = html.replace(/^&gt;\s+(.+)$/gm, '<blockquote style="border-left:3px solid #ccc;padding-left:12px;color:#666;margin:8px 0">$1</blockquote>');

  // Unordered lists
  html = html.replace(/^[\-\*]\s+(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

  // Ordered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr/>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  // Line breaks: double newlines become paragraphs
  html = html.replace(/\n\n/g, '</p><p>');
  html = '<p>' + html + '</p>';

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '');
  html = html.replace(/<p>(<h[1-6]>)/g, '$1');
  html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
  html = html.replace(/<p>(<pre)/g, '$1');
  html = html.replace(/(<\/pre>)<\/p>/g, '$1');
  html = html.replace(/<p>(<ul>)/g, '$1');
  html = html.replace(/(<\/ul>)<\/p>/g, '$1');
  html = html.replace(/<p>(<blockquote)/g, '$1');
  html = html.replace(/(<\/blockquote>)<\/p>/g, '$1');
  html = html.replace(/<p>(<hr\/>)/g, '$1');

  return html;
}

const Notes: React.FC = () => {
  const [notes, setNotes] = useState<NoteItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Editor state
  const [selectedNote, setSelectedNote] = useState<NoteItem | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [editContent, setEditContent] = useState('');
  const [previewMode, setPreviewMode] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  // New note state
  const [showNewNote, setShowNewNote] = useState(false);
  const [newTitle, setNewTitle] = useState('');

  // Delete confirmation
  const [deleteConfirm, setDeleteConfirm] = useState<{ show: boolean; noteId: number | null }>({ show: false, noteId: null });

  const fetchNotes = useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (searchQuery) params.set('search', searchQuery);
      const response = await fetch(`/api/v1/notes/?${params.toString()}`);
      if (!response.ok) throw new Error('Failed to fetch notes');
      const data = await response.json();
      setNotes(data.notes || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load notes');
    } finally {
      setLoading(false);
    }
  }, [searchQuery]);

  useEffect(() => {
    fetchNotes();
  }, [fetchNotes]);

  const selectNote = (note: NoteItem) => {
    if (dirty && selectedNote) {
      if (!window.confirm('You have unsaved changes. Discard them?')) return;
    }
    setSelectedNote(note);
    setEditTitle(note.title);
    setEditContent(note.content);
    setPreviewMode(false);
    setDirty(false);
  };

  const handleCreateNote = async () => {
    if (!newTitle.trim()) return;
    try {
      const response = await fetch('/api/v1/notes/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle.trim(), content: '' })
      });
      if (!response.ok) throw new Error('Failed to create note');
      const data = await response.json();
      setShowNewNote(false);
      setNewTitle('');
      await fetchNotes();
      selectNote(data.note);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create note');
    }
  };

  const handleSave = async () => {
    if (!selectedNote) return;
    try {
      setSaving(true);
      const response = await fetch(`/api/v1/notes/${selectedNote.note_id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: editTitle, content: editContent })
      });
      if (!response.ok) throw new Error('Failed to save note');
      const data = await response.json();
      setSelectedNote(data.note);
      setDirty(false);
      fetchNotes();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save note');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteConfirm.noteId) return;
    try {
      const response = await fetch(`/api/v1/notes/${deleteConfirm.noteId}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete note');
      if (selectedNote?.note_id === deleteConfirm.noteId) {
        setSelectedNote(null);
        setEditTitle('');
        setEditContent('');
        setDirty(false);
      }
      setDeleteConfirm({ show: false, noteId: null });
      fetchNotes();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete note');
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchNotes();
  };

  const formatDate = (dateString: string): string => {
    if (!dateString) return '';
    const d = new Date(dateString);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
  };

  // Keyboard shortcut for save
  const handleEditorKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      handleSave();
    }
  };

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Notes sidebar/list */}
      <div style={{
        width: '300px',
        borderRight: '1px solid #ddd',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#fafafa',
        flexShrink: 0
      }}>
        <div style={{ padding: '15px', borderBottom: '1px solid #ddd' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h2 style={{ margin: 0, fontSize: '18px', color: '#333' }}>Notes</h2>
            <button
              onClick={() => setShowNewNote(true)}
              style={{
                padding: '6px 12px',
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '13px'
              }}
            >
              + New
            </button>
          </div>

          {showNewNote && (
            <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
              <input
                type="text"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleCreateNote(); if (e.key === 'Escape') setShowNewNote(false); }}
                placeholder="Note title..."
                autoFocus
                style={{
                  flex: 1,
                  padding: '6px 8px',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  fontSize: '13px'
                }}
              />
              <button
                onClick={handleCreateNote}
                style={{
                  padding: '6px 10px',
                  backgroundColor: '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Create
              </button>
            </div>
          )}

          <form onSubmit={handleSearch} style={{ display: 'flex', gap: '5px' }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search notes..."
              style={{
                flex: 1,
                padding: '6px 8px',
                border: '1px solid #ccc',
                borderRadius: '4px',
                fontSize: '13px'
              }}
            />
          </form>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {loading ? (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>Loading...</div>
          ) : notes.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>
              {searchQuery ? 'No notes found.' : 'No notes yet. Create one!'}
            </div>
          ) : (
            notes.map((note) => (
              <div
                key={note.note_id}
                onClick={() => selectNote(note)}
                style={{
                  padding: '12px 15px',
                  borderBottom: '1px solid #eee',
                  cursor: 'pointer',
                  backgroundColor: selectedNote?.note_id === note.note_id ? '#e3f2fd' : 'transparent',
                  transition: 'background-color 0.15s'
                }}
                onMouseEnter={(e) => {
                  if (selectedNote?.note_id !== note.note_id) {
                    e.currentTarget.style.backgroundColor = '#f0f0f0';
                  }
                }}
                onMouseLeave={(e) => {
                  if (selectedNote?.note_id !== note.note_id) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontWeight: 600,
                      fontSize: '14px',
                      color: '#333',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {note.title}
                    </div>
                    <div style={{
                      fontSize: '12px',
                      color: '#999',
                      marginTop: '4px',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {note.content.substring(0, 80) || 'Empty note'}
                    </div>
                    <div style={{ fontSize: '11px', color: '#bbb', marginTop: '4px' }}>
                      {formatDate(note.update_date)}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteConfirm({ show: true, noteId: note.note_id });
                    }}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#ccc',
                      cursor: 'pointer',
                      padding: '2px 6px',
                      fontSize: '16px',
                      lineHeight: 1
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.color = '#dc3545'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = '#ccc'; }}
                    title="Delete note"
                  >
                    ×
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Editor area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {selectedNote ? (
          <>
            {/* Toolbar */}
            <div style={{
              padding: '10px 15px',
              borderBottom: '1px solid #ddd',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              backgroundColor: '#fafafa'
            }}>
              <input
                type="text"
                value={editTitle}
                onChange={(e) => { setEditTitle(e.target.value); setDirty(true); }}
                style={{
                  flex: 1,
                  padding: '6px 10px',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  fontSize: '16px',
                  fontWeight: 600
                }}
              />
              <div style={{ display: 'flex', gap: '5px' }}>
                <button
                  onClick={() => setPreviewMode(false)}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: !previewMode ? '#007bff' : '#e9ecef',
                    color: !previewMode ? 'white' : '#333',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '13px'
                  }}
                >
                  Edit
                </button>
                <button
                  onClick={() => setPreviewMode(true)}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: previewMode ? '#007bff' : '#e9ecef',
                    color: previewMode ? 'white' : '#333',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '13px'
                  }}
                >
                  Preview
                </button>
              </div>
              <button
                onClick={handleSave}
                disabled={saving || !dirty}
                style={{
                  padding: '6px 16px',
                  backgroundColor: dirty ? '#28a745' : '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: dirty ? 'pointer' : 'default',
                  fontSize: '13px',
                  opacity: dirty ? 1 : 0.6
                }}
              >
                {saving ? 'Saving...' : dirty ? 'Save' : 'Saved'}
              </button>
              <span style={{ fontSize: '11px', color: '#999' }}>Ctrl+S</span>
            </div>

            {/* Content area */}
            {previewMode ? (
              <div
                style={{
                  flex: 1,
                  padding: '20px 30px',
                  overflowY: 'auto',
                  fontSize: '14px',
                  lineHeight: '1.6',
                  color: '#333'
                }}
                dangerouslySetInnerHTML={{ __html: renderMarkdown(editContent) }}
              />
            ) : (
              <textarea
                value={editContent}
                onChange={(e) => { setEditContent(e.target.value); setDirty(true); }}
                onKeyDown={handleEditorKeyDown}
                placeholder="Write your note in markdown..."
                style={{
                  flex: 1,
                  padding: '20px 30px',
                  border: 'none',
                  outline: 'none',
                  resize: 'none',
                  fontSize: '14px',
                  lineHeight: '1.6',
                  fontFamily: "'SF Mono', 'Fira Code', 'Fira Mono', Menlo, Consolas, monospace",
                  color: '#333',
                  backgroundColor: '#fff'
                }}
              />
            )}
          </>
        ) : (
          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#999',
            fontSize: '16px'
          }}>
            Select a note or create a new one
          </div>
        )}
      </div>

      {/* Error toast */}
      {error && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          padding: '12px 20px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderRadius: '5px',
          zIndex: 1000,
          cursor: 'pointer'
        }} onClick={() => setError(null)}>
          {error}
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteConfirm.show && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            maxWidth: '400px'
          }}>
            <h3 style={{ margin: '0 0 15px 0' }}>Delete Note</h3>
            <p style={{ margin: '0 0 20px 0' }}>Are you sure you want to delete this note? This cannot be undone.</p>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setDeleteConfirm({ show: false, noteId: null })}
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
                onClick={handleDelete}
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

export default Notes;
