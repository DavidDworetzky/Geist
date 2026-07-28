import { act, renderHook, waitFor } from '@testing-library/react';
import useChatMemory, {
  ChatMemorySettings,
  getMemoryScope,
} from '../useChatMemory';

type MemoryPatch = {
  chatId: number;
  body: Record<string, unknown>;
};

const DEFAULT_CHAT_SETTINGS: ChatMemorySettings = {
  chat_session_id: 42,
  memory_enabled: true,
  memory_mode: 'public',
  folder_id: null,
  effective_scope: 'public',
  status: 'ready',
};

function response(body: unknown, ok = true): Response {
  return {
    ok,
    json: async () => body,
  } as Response;
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>(next => {
    resolve = next;
  });
  return { promise, resolve };
}

function applyServerChanges(
  current: ChatMemorySettings,
  changes: Record<string, unknown>,
): ChatMemorySettings {
  const next = { ...current };
  if (typeof changes.memory_enabled === 'boolean') {
    next.memory_enabled = changes.memory_enabled;
  }
  if (changes.memory_mode === 'public' || changes.memory_mode === 'private') {
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

function installMemoryApi(initialSettings: ChatMemorySettings[]) {
  const settingsById = new Map(
    initialSettings.map(settings => [settings.chat_session_id!, { ...settings }]),
  );
  const patches: MemoryPatch[] = [];
  let patchFailure: string | null = null;
  const folders = [
    {
      folder_id: 7,
      name: 'Research',
      color: 'violet',
      chat_count: 1,
    },
    {
      folder_id: 9,
      name: 'Planning',
      color: 'mint',
      chat_count: 0,
    },
  ];

  const fetchMock = jest.fn(async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    if (url === '/api/v1/memory/folders' && method === 'GET') {
      return response(folders);
    }

    const chatMatch = url.match(/^\/api\/v1\/memory\/chats\/(\d+)$/);
    if (chatMatch) {
      const chatId = Number(chatMatch[1]);
      if (method === 'GET') {
        return response(settingsById.get(chatId));
      }
      if (method === 'PATCH') {
        const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
        patches.push({ chatId, body });
        if (patchFailure) {
          const detail = patchFailure;
          patchFailure = null;
          return response({ detail }, false);
        }
        const current = settingsById.get(chatId) ?? {
          ...DEFAULT_CHAT_SETTINGS,
          chat_session_id: chatId,
        };
        const updated = applyServerChanges(current, body);
        settingsById.set(chatId, updated);
        return response(updated);
      }
    }

    throw new Error(`Unexpected request: ${method} ${url}`);
  });
  global.fetch = fetchMock as unknown as typeof fetch;

  return {
    fetchMock,
    patches,
    failNextPatch(detail: string) {
      patchFailure = detail;
    },
  };
}

describe('getMemoryScope', () => {
  it('uses the effective disabled, folder, private, and public scopes', () => {
    expect(getMemoryScope({
      ...DEFAULT_CHAT_SETTINGS,
      memory_enabled: false,
      folder_id: 7,
    })).toEqual({ kind: 'disabled' });
    expect(getMemoryScope({
      ...DEFAULT_CHAT_SETTINGS,
      folder_id: 7,
      memory_mode: 'private',
    })).toEqual({ kind: 'folder', folderId: 7 });
    expect(getMemoryScope({
      ...DEFAULT_CHAT_SETTINGS,
      memory_mode: 'private',
    })).toEqual({ kind: 'private' });
    expect(getMemoryScope(DEFAULT_CHAT_SETTINGS)).toEqual({ kind: 'public' });
  });
});

describe('useChatMemory', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('updates a blank-chat draft without sending a PATCH', async () => {
    const api = installMemoryApi([]);
    const { result } = renderHook(() => useChatMemory(null));
    await waitFor(() => expect(result.current.folders).toHaveLength(2));

    act(() => result.current.prepareNewChat({ kind: 'folder', folderId: 7 }));
    expect(getMemoryScope(result.current.settings)).toEqual({
      kind: 'folder',
      folderId: 7,
    });

    await act(async () => {
      expect(await result.current.setScope({ kind: 'disabled' })).toBe(true);
    });
    expect(result.current.settings.folder_id).toBe(7);
    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'disabled' });

    await act(async () => {
      expect(await result.current.setScope({ kind: 'public' })).toBe(true);
    });
    expect(result.current.settings.folder_id).toBeNull();
    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'public' });
    expect(api.patches).toEqual([]);
  });

  it('prepares a new-chat scope without changing or patching the active chat', async () => {
    const api = installMemoryApi([DEFAULT_CHAT_SETTINGS]);
    const { result, rerender } = renderHook(
      ({ activeChatId }: { activeChatId: number | null }) => useChatMemory(activeChatId),
      { initialProps: { activeChatId: 42 as number | null } },
    );
    await waitFor(() => expect(result.current.settings.chat_session_id).toBe(42));

    act(() => result.current.prepareNewChat({ kind: 'folder', folderId: 9 }));
    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'public' });
    expect(api.patches).toEqual([]);

    rerender({ activeChatId: null });
    await waitFor(() => expect(getMemoryScope(result.current.settings)).toEqual({
      kind: 'folder',
      folderId: 9,
    }));
    expect(api.patches).toEqual([]);
  });

  it('maps every active-chat scope to the expected PATCH payload', async () => {
    const api = installMemoryApi([{
      ...DEFAULT_CHAT_SETTINGS,
      folder_id: 7,
      memory_mode: 'private',
      effective_scope: 'folder',
    }]);
    const { result } = renderHook(() => useChatMemory(42));
    await waitFor(() => expect(result.current.settings.chat_session_id).toBe(42));

    await act(async () => {
      expect(await result.current.setScope({ kind: 'disabled' })).toBe(true);
    });
    expect(result.current.settings.folder_id).toBe(7);
    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'disabled' });

    await act(async () => {
      expect(await result.current.setScope({ kind: 'public' })).toBe(true);
      expect(await result.current.setScope({ kind: 'private' })).toBe(true);
      expect(await result.current.setScope({ kind: 'folder', folderId: 9 })).toBe(true);
    });

    expect(api.patches).toEqual([
      {
        chatId: 42,
        body: { memory_enabled: false },
      },
      {
        chatId: 42,
        body: {
          memory_enabled: true,
          memory_mode: 'public',
          clear_folder: true,
        },
      },
      {
        chatId: 42,
        body: {
          memory_enabled: true,
          memory_mode: 'private',
          clear_folder: true,
        },
      },
      {
        chatId: 42,
        body: {
          memory_enabled: true,
          memory_mode: 'private',
          folder_id: 9,
        },
      },
    ]);
    expect(getMemoryScope(result.current.settings)).toEqual({
      kind: 'folder',
      folderId: 9,
    });
  });

  it('does not overwrite active settings when updating a different chat folder', async () => {
    const api = installMemoryApi([
      {
        ...DEFAULT_CHAT_SETTINGS,
        folder_id: 7,
        memory_mode: 'private',
        effective_scope: 'folder',
      },
      {
        ...DEFAULT_CHAT_SETTINGS,
        chat_session_id: 99,
      },
    ]);
    const { result } = renderHook(() => useChatMemory(42));
    await waitFor(() => expect(result.current.settings.chat_session_id).toBe(42));

    await act(async () => {
      expect(await result.current.setChatFolder(99, 9)).toBe(true);
    });
    expect(result.current.settings.chat_session_id).toBe(42);
    expect(result.current.settings.folder_id).toBe(7);

    await act(async () => {
      expect(await result.current.setChatFolder(42, null)).toBe(true);
    });
    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'private' });
    expect(api.patches).toEqual([
      {
        chatId: 99,
        body: {
          memory_mode: 'private',
          folder_id: 9,
        },
      },
      {
        chatId: 42,
        body: {
          memory_mode: 'private',
          clear_folder: true,
        },
      },
    ]);
  });

  it('returns false and restores active settings when a PATCH fails', async () => {
    const api = installMemoryApi([DEFAULT_CHAT_SETTINGS]);
    const { result } = renderHook(() => useChatMemory(42));
    await waitFor(() => expect(result.current.settings.chat_session_id).toBe(42));
    api.failNextPatch('Folder not found');

    await act(async () => {
      expect(await result.current.setScope({ kind: 'folder', folderId: 999 })).toBe(false);
    });

    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'public' });
    expect(result.current.error).toBe('Folder not found');
  });

  it('keeps the latest scope and remains loading until overlapping PATCH requests settle', async () => {
    const firstPatch = deferred<Response>();
    const secondPatch = deferred<Response>();
    let patchCount = 0;
    global.fetch = jest.fn((
      input: RequestInfo | URL,
      init?: RequestInit,
    ): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/v1/memory/folders' && method === 'GET') {
        return Promise.resolve(response([]));
      }
      if (url === '/api/v1/memory/chats/42' && method === 'GET') {
        return Promise.resolve(response(DEFAULT_CHAT_SETTINGS));
      }
      if (url === '/api/v1/memory/chats/42' && method === 'PATCH') {
        patchCount += 1;
        return patchCount === 1 ? firstPatch.promise : secondPatch.promise;
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useChatMemory(42));
    await waitFor(() => expect(result.current.settings.chat_session_id).toBe(42));

    let firstResult!: Promise<boolean>;
    let secondResult!: Promise<boolean>;
    act(() => {
      firstResult = result.current.setScope({ kind: 'private' });
    });
    act(() => {
      secondResult = result.current.setScope({ kind: 'public' });
    });
    expect(result.current.loading).toBe(true);

    await act(async () => {
      secondPatch.resolve(response({
        ...DEFAULT_CHAT_SETTINGS,
        memory_mode: 'public',
        effective_scope: 'public',
      }));
      expect(await secondResult).toBe(true);
    });
    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'public' });
    expect(result.current.loading).toBe(true);

    await act(async () => {
      firstPatch.resolve(response({
        ...DEFAULT_CHAT_SETTINGS,
        memory_mode: 'private',
        effective_scope: 'private',
      }));
      expect(await firstResult).toBe(true);
    });
    expect(getMemoryScope(result.current.settings)).toEqual({ kind: 'public' });
    expect(result.current.loading).toBe(false);
  });
});
