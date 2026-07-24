import { useCallback, useEffect, useState } from 'react';

export interface MemoryFolder {
  folder_id: number;
  name: string;
  color: string;
  chat_count: number;
  summary?: string | null;
}

export interface ChatMemorySettings {
  chat_session_id?: number;
  memory_enabled: boolean;
  memory_mode: 'public' | 'private';
  folder_id: number | null;
  effective_scope?: 'public' | 'private' | 'folder' | 'disabled';
  status?: string;
}

const DEFAULT_SETTINGS: ChatMemorySettings = {
  memory_enabled: true,
  memory_mode: 'public',
  folder_id: null,
  effective_scope: 'public',
  status: 'ready',
};

export default function useChatMemory(chatId: number | null) {
  const [settings, setSettings] = useState<ChatMemorySettings>(DEFAULT_SETTINGS);
  const [folders, setFolders] = useState<MemoryFolder[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshFolders = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/memory/folders');
      if (!response.ok) throw new Error('Could not load folders');
      setFolders(await response.json());
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not load folders');
    }
  }, []);

  useEffect(() => {
    void refreshFolders();
  }, [refreshFolders]);

  const fetchSettings = useCallback(async () => {
    if (chatId === null) {
      return DEFAULT_SETTINGS;
    }
    const response = await fetch(`/api/v1/memory/chats/${chatId}`);
    if (!response.ok) throw new Error('Could not load chat memory settings');
    const data: ChatMemorySettings = await response.json();
    return data;
  }, [chatId]);

  useEffect(() => {
    if (chatId === null) {
      setSettings(DEFAULT_SETTINGS);
      return;
    }
    let cancelled = false;
    setLoading(true);
    fetchSettings()
      .then(data => {
        if (!cancelled) setSettings(data);
      })
      .catch(requestError => {
        if (!cancelled) {
          setError(requestError instanceof Error ? requestError.message : 'Could not load memory settings');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [chatId, fetchSettings]);

  useEffect(() => {
    if (
      chatId === null
      || settings.status === 'ready'
      || settings.status === 'disabled'
      || !settings.status
    ) {
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void fetchSettings()
        .then(data => {
          if (!cancelled) setSettings(data);
        })
        .catch(() => undefined);
    }, 1200);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [chatId, fetchSettings, settings.status]);

  const updateSettings = useCallback(async (changes: Record<string, unknown>) => {
    const next: ChatMemorySettings = {
      ...settings,
      ...changes,
      folder_id: changes.clear_folder ? null : (
        typeof changes.folder_id === 'number' ? changes.folder_id : settings.folder_id
      ),
    };
    if (next.folder_id !== null) next.memory_mode = 'private';
    next.effective_scope = !next.memory_enabled
      ? 'disabled'
      : (next.folder_id !== null ? 'folder' : next.memory_mode);
    setSettings(next);
    setError(null);
    if (chatId === null) return;
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/memory/chats/${chatId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(changes),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || 'Could not update memory settings');
      }
      setSettings(await response.json());
      await refreshFolders();
    } catch (requestError) {
      setSettings(settings);
      setError(requestError instanceof Error ? requestError.message : 'Could not update memory settings');
    } finally {
      setLoading(false);
    }
  }, [chatId, refreshFolders, settings]);

  const createFolder = useCallback(async (name: string) => {
    const response = await fetch('/api/v1/memory/folders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, color: 'violet' }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Could not create folder');
    }
    const folder: MemoryFolder = await response.json();
    await refreshFolders();
    return folder;
  }, [refreshFolders]);

  const renameFolder = useCallback(async (folderId: number, name: string) => {
    const response = await fetch(`/api/v1/memory/folders/${folderId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Could not rename folder');
    }
    await refreshFolders();
  }, [refreshFolders]);

  const deleteFolder = useCallback(async (folderId: number) => {
    const response = await fetch(`/api/v1/memory/folders/${folderId}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Could not delete folder');
    if (settings.folder_id === folderId) {
      setSettings(current => ({
        ...current,
        folder_id: null,
        memory_mode: 'private',
        effective_scope: 'private',
      }));
    }
    await refreshFolders();
  }, [refreshFolders, settings.folder_id]);

  return {
    settings,
    folders,
    loading,
    error,
    refreshFolders,
    createFolder,
    renameFolder,
    deleteFolder,
    setMemoryEnabled: (enabled: boolean) => updateSettings({ memory_enabled: enabled }),
    setPrivate: (isPrivate: boolean) => updateSettings({
      memory_mode: isPrivate ? 'private' : 'public',
      ...(isPrivate ? {} : { clear_folder: true }),
    }),
    setFolder: (folderId: number | null) => updateSettings(
      folderId === null ? { clear_folder: true } : { folder_id: folderId, memory_mode: 'private' },
    ),
  };
}
