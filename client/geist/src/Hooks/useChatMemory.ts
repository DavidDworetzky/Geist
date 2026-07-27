import { useCallback, useEffect, useRef, useState } from 'react';

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

export type MemoryScope =
  | { kind: 'disabled' | 'public' | 'private' }
  | { kind: 'folder'; folderId: number };

const DEFAULT_SETTINGS: ChatMemorySettings = {
  memory_enabled: true,
  memory_mode: 'public',
  folder_id: null,
  effective_scope: 'public',
  status: 'ready',
};

type ChatMemoryChanges = {
  memory_enabled?: boolean;
  memory_mode?: 'public' | 'private';
  folder_id?: number;
  clear_folder?: boolean;
};

export function getMemoryScope(settings: ChatMemorySettings): MemoryScope {
  if (!settings.memory_enabled) {
    return { kind: 'disabled' };
  }
  if (settings.folder_id !== null) {
    return { kind: 'folder', folderId: settings.folder_id };
  }
  return { kind: settings.memory_mode };
}

function changesForScope(scope: MemoryScope): ChatMemoryChanges {
  if (scope.kind === 'disabled') {
    return { memory_enabled: false };
  }
  if (scope.kind === 'folder') {
    return {
      memory_enabled: true,
      memory_mode: 'private',
      folder_id: scope.folderId,
    };
  }
  return {
    memory_enabled: true,
    memory_mode: scope.kind,
    clear_folder: true,
  };
}

function applyChanges(
  current: ChatMemorySettings,
  changes: ChatMemoryChanges,
): ChatMemorySettings {
  const next: ChatMemorySettings = { ...current };
  if (typeof changes.memory_enabled === 'boolean') {
    next.memory_enabled = changes.memory_enabled;
  }
  if (changes.memory_mode) {
    next.memory_mode = changes.memory_mode;
  }
  if (changes.clear_folder) {
    next.folder_id = null;
  } else if (typeof changes.folder_id === 'number') {
    next.folder_id = changes.folder_id;
  }
  if (next.folder_id !== null) {
    next.memory_mode = 'private';
  }
  next.effective_scope = !next.memory_enabled
    ? 'disabled'
    : (next.folder_id !== null ? 'folder' : next.memory_mode);
  next.status = next.memory_enabled ? 'ready' : 'disabled';
  return next;
}

export default function useChatMemory(chatId: number | null) {
  const [settings, setSettings] = useState<ChatMemorySettings>(DEFAULT_SETTINGS);
  const [folders, setFolders] = useState<MemoryFolder[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeChatIdRef = useRef(chatId);
  const settingsRef = useRef(settings);
  const draftSettingsRef = useRef(DEFAULT_SETTINGS);
  const pendingRequestCountRef = useRef(0);
  const settingsRequestVersionRef = useRef(0);
  const errorRequestVersionRef = useRef(0);

  activeChatIdRef.current = chatId;
  settingsRef.current = settings;

  const beginRequest = useCallback(() => {
    pendingRequestCountRef.current += 1;
    setLoading(true);
  }, []);

  const endRequest = useCallback(() => {
    pendingRequestCountRef.current = Math.max(0, pendingRequestCountRef.current - 1);
    if (pendingRequestCountRef.current === 0) {
      setLoading(false);
    }
  }, []);

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

  const fetchSettings = useCallback(async (signal?: AbortSignal) => {
    if (chatId === null) {
      return DEFAULT_SETTINGS;
    }
    const response = await fetch(`/api/v1/memory/chats/${chatId}`, { signal });
    if (!response.ok) throw new Error('Could not load chat memory settings');
    const data: ChatMemorySettings = await response.json();
    return data;
  }, [chatId]);

  useEffect(() => {
    const settingsRequestVersion = ++settingsRequestVersionRef.current;
    const errorRequestVersion = ++errorRequestVersionRef.current;
    if (chatId === null) {
      setSettings(draftSettingsRef.current);
      settingsRef.current = draftSettingsRef.current;
      setError(null);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    beginRequest();
    fetchSettings(controller.signal)
      .then(data => {
        if (
          !cancelled
          && settingsRequestVersionRef.current === settingsRequestVersion
        ) {
          settingsRef.current = data;
          setSettings(data);
          if (errorRequestVersionRef.current === errorRequestVersion) {
            setError(null);
          }
        }
      })
      .catch(requestError => {
        if (
          !cancelled
          && errorRequestVersionRef.current === errorRequestVersion
          && !(requestError instanceof DOMException && requestError.name === 'AbortError')
        ) {
          setError(requestError instanceof Error ? requestError.message : 'Could not load memory settings');
        }
      })
      .finally(() => {
        endRequest();
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [beginRequest, chatId, endRequest, fetchSettings]);

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
      const settingsRequestVersion = ++settingsRequestVersionRef.current;
      void fetchSettings()
        .then(data => {
          if (
            !cancelled
            && settingsRequestVersionRef.current === settingsRequestVersion
          ) {
            settingsRef.current = data;
            setSettings(data);
          }
        })
        .catch(() => undefined);
    }, 1200);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [chatId, fetchSettings, settings.status]);

  const updateChatSettings = useCallback(async (
    targetChatId: number,
    changes: ChatMemoryChanges,
  ): Promise<boolean> => {
    const updatesActiveChat = activeChatIdRef.current === targetChatId;
    const previousSettings = settingsRef.current;
    const settingsRequestVersion = updatesActiveChat
      ? ++settingsRequestVersionRef.current
      : null;
    const errorRequestVersion = ++errorRequestVersionRef.current;
    if (updatesActiveChat) {
      const optimisticSettings = applyChanges(previousSettings, changes);
      settingsRef.current = optimisticSettings;
      setSettings(optimisticSettings);
    }
    setError(null);
    beginRequest();
    try {
      const response = await fetch(`/api/v1/memory/chats/${targetChatId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(changes),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || 'Could not update memory settings');
      }
      const updatedSettings: ChatMemorySettings = await response.json();
      if (
        activeChatIdRef.current === targetChatId
        && settingsRequestVersionRef.current === settingsRequestVersion
      ) {
        settingsRef.current = updatedSettings;
        setSettings(updatedSettings);
      }
      await refreshFolders();
      return true;
    } catch (requestError) {
      if (
        updatesActiveChat
        && activeChatIdRef.current === targetChatId
        && settingsRequestVersionRef.current === settingsRequestVersion
      ) {
        settingsRef.current = previousSettings;
        setSettings(previousSettings);
      }
      if (errorRequestVersionRef.current === errorRequestVersion) {
        setError(requestError instanceof Error ? requestError.message : 'Could not update memory settings');
      }
      return false;
    } finally {
      endRequest();
    }
  }, [beginRequest, endRequest, refreshFolders]);

  const setScope = useCallback(async (scope: MemoryScope): Promise<boolean> => {
    const changes = changesForScope(scope);
    const targetChatId = activeChatIdRef.current;
    if (targetChatId === null) {
      const next = applyChanges(draftSettingsRef.current, changes);
      draftSettingsRef.current = next;
      settingsRef.current = next;
      setSettings(next);
      setError(null);
      return true;
    }
    return updateChatSettings(targetChatId, changes);
  }, [updateChatSettings]);

  const prepareNewChat = useCallback((scope: MemoryScope): void => {
    const next = applyChanges(DEFAULT_SETTINGS, changesForScope(scope));
    draftSettingsRef.current = next;
    if (activeChatIdRef.current === null) {
      settingsRef.current = next;
      setSettings(next);
      setError(null);
    }
  }, []);

  const setChatFolder = useCallback(async (
    targetChatId: number,
    folderId: number | null,
  ): Promise<boolean> => updateChatSettings(
    targetChatId,
    folderId === null
      ? { memory_mode: 'private', clear_folder: true }
      : {
          memory_mode: 'private',
          folder_id: folderId,
        },
  ), [updateChatSettings]);

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
    if (settingsRef.current.folder_id === folderId) {
      setSettings(current => {
        const next: ChatMemorySettings = {
          ...current,
          folder_id: null,
          memory_mode: 'private',
          effective_scope: current.memory_enabled ? 'private' : 'disabled',
        };
        settingsRef.current = next;
        if (activeChatIdRef.current === null) {
          draftSettingsRef.current = next;
        }
        return next;
      });
    }
    await refreshFolders();
  }, [refreshFolders]);

  return {
    settings,
    folders,
    loading,
    error,
    refreshFolders,
    createFolder,
    renameFolder,
    deleteFolder,
    setScope,
    prepareNewChat,
    setChatFolder,
  };
}
