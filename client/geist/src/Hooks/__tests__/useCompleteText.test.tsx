import { act, renderHook, waitFor } from '@testing-library/react';
import { TextDecoder } from 'util';
import useCompleteText, {
  chatStreamReducer,
  initialChatStreamState,
} from '../useCompleteText';


Object.defineProperty(global, 'TextDecoder', {
  value: TextDecoder,
  writable: true,
});

const encode = (value: string): Uint8Array => new Uint8Array(Buffer.from(value));

const streamingResponse = (chunks: string[]): Response => {
  let index = 0;
  return {
    ok: true,
    body: {
      getReader: () => ({
        read: jest.fn(async () => {
          if (index >= chunks.length) {
            return { done: true, value: undefined };
          }
          return { done: false, value: encode(chunks[index++]) };
        }),
      }),
    },
  } as unknown as Response;
};

describe('chatStreamReducer', () => {
  it('upserts repeated tool lifecycle events by id', () => {
    let state = chatStreamReducer(initialChatStreamState, {
      type: 'START',
      prompt: 'Search for pi',
      chatId: null,
    });
    state = chatStreamReducer(state, {
      type: 'TOOL_UPSERT',
      toolCall: {
        id: 'call_1',
        name: 'search',
        arguments: { query: 'pi' },
        status: 'proposed',
      },
    });
    state = chatStreamReducer(state, {
      type: 'TOOL_UPSERT',
      toolCall: { id: 'call_1', status: 'running' },
    });
    state = chatStreamReducer(state, {
      type: 'TOOL_UPSERT',
      toolCall: {
        id: 'call_1',
        status: 'succeeded',
        result_summary: 'Found it',
      },
    });

    expect(state.activeTurn?.tool_calls).toHaveLength(1);
    expect(state.activeTurn?.tool_calls[0]).toMatchObject({
      id: 'call_1',
      name: 'search',
      arguments: { query: 'pi' },
      status: 'succeeded',
      result_summary: 'Found it',
    });
  });

  it('upserts repeated artifacts by id', () => {
    let state = chatStreamReducer(initialChatStreamState, {
      type: 'START',
      prompt: 'Create a file',
      chatId: null,
    });
    state = chatStreamReducer(state, {
      type: 'ARTIFACT_UPSERT',
      artifact: {
        id: 'artifact_1',
        kind: 'text',
        mime_type: 'text/plain',
        filename: 'draft.txt',
        sha256: 'draft-hash',
      },
    });
    state = chatStreamReducer(state, {
      type: 'ARTIFACT_UPSERT',
      artifact: {
        id: 'artifact_1',
        kind: 'text',
        mime_type: 'text/plain',
        filename: 'final.txt',
        sha256: 'final-hash',
        url: 'https://example.com/final.txt',
      },
    });

    expect(state.activeTurn?.artifacts).toEqual([
      expect.objectContaining({
        id: 'artifact_1',
        filename: 'final.txt',
        sha256: 'final-hash',
        url: 'https://example.com/final.txt',
      }),
    ]);
  });
});

describe('useCompleteText', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
  });

  it('handles fragmented SSE and returns the authoritative final envelope', async () => {
    const toolCall = {
      id: 'call_1',
      name: 'search',
      arguments: { query: 'pi framework' },
      status: 'succeeded' as const,
      result_summary: 'Found documentation',
      artifact_ids: ['artifact_1'],
    };
    const artifact = {
      id: 'artifact_1',
      kind: 'text',
      mime_type: 'text/plain',
      filename: 'result.txt',
      sha256: 'abc123',
      url: 'https://example.com/result.txt',
    };
    const finalEnvelope = {
      run_id: 'run_1',
      message: ['Hello'],
      chat_id: 7,
      tool_calls: [toolCall],
      artifacts: [artifact],
    };
    const chunks = [
      'event: run_started\r\ndata: {"run_id":"run_1","chat_id":7}\r\n\r\nevent: delta\ndata: {"text":"Dra',
      'ft"}\n\nevent: tool_call\ndata: {"id":"call_1","name":"search","arguments":{"query":"pi framework"},"status":"proposed"}\n\n',
      'event: tool_call\ndata: {"id":"call_1","status":"running"}\n\nevent: artifact\ndata: {"id":"artifact_1","kind":"text",',
      '"mime_type":"text/plain","filename":"result.txt","sha256":"abc123","url":"https://example.com/result.txt"}\n\n',
      `event: tool_call\ndata: ${JSON.stringify(toolCall)}\n\nevent: final\ndata: ${JSON.stringify(finalEnvelope)}\n\nevent: done\ndata: {"run_id":"run_1","chat_id":7}\n\n`,
    ];
    const fetchMock = jest.fn().mockResolvedValue(streamingResponse(chunks));
    global.fetch = fetchMock as typeof fetch;

    const { result } = renderHook(() => useCompleteText());
    await act(async () => {
      await result.current.completeText('Say hello');
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/agent/complete_text_stream',
      expect.objectContaining({ signal: expect.objectContaining({ aborted: false }) }),
    );
    const requestBody = JSON.parse(fetchMock.mock.calls[0][1]?.body as string);
    expect(requestBody).toMatchObject({ prompt: 'Say hello', enable_tools: true });
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.state_chat_id).toBe(7);
    expect(result.current.completedTurn).toMatchObject({
      run_id: 'run_1',
      prompt: 'Say hello',
      message: 'Hello',
      chat_id: 7,
      origin_chat_id: null,
    });
    expect(result.current.completedTurn?.tool_calls).toEqual([toolCall]);
    expect(result.current.completedTurn?.artifacts).toEqual([artifact]);
  });

  it('aborts the stream and posts cancellation for a started run', async () => {
    let readCount = 0;
    let streamSignal: AbortSignal | undefined;
    const fetchMock = jest.fn((url: RequestInfo | URL, options?: RequestInit) => {
      if (String(url) === '/agent/runs/run_cancel/cancel') {
        return Promise.resolve({
          ok: true,
          json: async () => ({ cancelled: true }),
        } as Response);
      }

      streamSignal = options?.signal as AbortSignal;
      const response = {
        ok: true,
        body: {
          getReader: () => ({
            read: jest.fn(() => {
              readCount += 1;
              if (readCount === 1) {
                return Promise.resolve({
                  done: false,
                  value: encode('event: run_started\ndata: {"run_id":"run_cancel"}\n\n'),
                });
              }
              return new Promise((_, reject) => {
                streamSignal?.addEventListener('abort', () => {
                  const error = new Error('Aborted');
                  error.name = 'AbortError';
                  reject(error);
                }, { once: true });
              });
            }),
          }),
        },
      } as unknown as Response;
      return Promise.resolve(response);
    });
    global.fetch = fetchMock as typeof fetch;

    const { result } = renderHook(() => useCompleteText());
    let completionPromise: Promise<void> = Promise.resolve();
    act(() => {
      completionPromise = result.current.completeText('Keep working');
    });

    await waitFor(() => {
      expect(result.current.activeTurn?.run_id).toBe('run_cancel');
    });

    await act(async () => {
      await result.current.cancelGeneration();
      await completionPromise!;
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/agent/runs/run_cancel/cancel',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(result.current.activeTurn?.status).toBe('cancelled');
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('reflects cancellation reported by the server stream', async () => {
    global.fetch = jest.fn().mockResolvedValue(streamingResponse([
      'event: run_started\ndata: {"run_id":"run_server_cancel"}\n\n' +
      'event: cancelled\ndata: {"run_id":"run_server_cancel"}\n\n' +
      'event: done\ndata: {"run_id":"run_server_cancel","chat_id":null}\n\n',
    ])) as typeof fetch;

    const { result } = renderHook(() => useCompleteText());
    await act(async () => {
      await result.current.completeText('Stop on the server');
    });

    expect(result.current.activeTurn?.status).toBe('cancelled');
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('marks an unexpected stream EOF as failed', async () => {
    global.fetch = jest.fn().mockResolvedValue(streamingResponse([
      'event: run_started\ndata: {"run_id":"run_eof"}\n\n' +
      'event: delta\ndata: {"text":"partial"}\n\n',
    ])) as typeof fetch;

    const { result } = renderHook(() => useCompleteText());
    await act(async () => {
      await result.current.completeText('Do not stop early');
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.activeTurn?.status).toBe('failed');
    expect(result.current.error).toBe('Chat stream ended before a final response.');
  });

  it('ignores buffered events from a superseded stream', async () => {
    let staleReadCount = 0;
    let resolveStaleRead: ((result: {
      done: boolean;
      value: Uint8Array | undefined;
    }) => void) | undefined;
    const staleResponse = {
      ok: true,
      body: {
        getReader: () => ({
          read: jest.fn(() => {
            staleReadCount += 1;
            if (staleReadCount === 1) {
              return Promise.resolve({
                done: false,
                value: encode('event: delta\ndata: {"text":"stale'),
              });
            }
            if (staleReadCount === 2) {
              return new Promise((resolve) => {
                resolveStaleRead = resolve;
              });
            }
            return Promise.resolve({ done: true, value: undefined });
          }),
        }),
      },
    } as unknown as Response;
    const currentResponse = streamingResponse([
      'event: run_started\ndata: {"run_id":"run_current","chat_id":22}\n\n' +
      'event: final\ndata: {"run_id":"run_current","message":["current"],"chat_id":22}\n\n' +
      'event: done\ndata: {"run_id":"run_current","chat_id":22}\n\n',
    ]);
    global.fetch = jest.fn()
      .mockResolvedValueOnce(staleResponse)
      .mockResolvedValueOnce(currentResponse) as typeof fetch;

    const { result } = renderHook(() => useCompleteText());
    let staleCompletion: Promise<void> = Promise.resolve();
    act(() => {
      staleCompletion = result.current.completeText('old request');
    });
    await waitFor(() => expect(resolveStaleRead).toBeDefined());

    await act(async () => {
      await result.current.completeText('current request');
    });

    await act(async () => {
      resolveStaleRead?.({
        done: false,
        value: encode('"}\n\nevent: final\ndata: {"run_id":"run_stale","message":["stale"],"chat_id":11}\n\n'),
      });
      await staleCompletion;
    });

    expect(result.current.state_chat_id).toBe(22);
    expect(result.current.completedTurn).toMatchObject({
      run_id: 'run_current',
      prompt: 'current request',
      message: 'current',
      chat_id: 22,
    });
    expect(result.current.error).toBeNull();
  });

  it('keeps the run active when the server rejects cancellation', async () => {
    let streamSignal: AbortSignal | undefined;
    let readCount = 0;
    const fetchMock = jest.fn((url: RequestInfo | URL, options?: RequestInit) => {
      if (String(url) === '/agent/runs/run_reject/cancel') {
        return Promise.resolve({
          ok: true,
          json: async () => ({ cancelled: false }),
        } as Response);
      }

      streamSignal = options?.signal as AbortSignal;
      return Promise.resolve({
        ok: true,
        body: {
          getReader: () => ({
            read: jest.fn(() => {
              readCount += 1;
              if (readCount === 1) {
                return Promise.resolve({
                  done: false,
                  value: encode('event: run_started\ndata: {"run_id":"run_reject"}\n\n'),
                });
              }
              return new Promise((_, reject) => {
                streamSignal?.addEventListener('abort', () => {
                  const error = new Error('Aborted');
                  error.name = 'AbortError';
                  reject(error);
                }, { once: true });
              });
            }),
          }),
        },
      } as unknown as Response);
    });
    global.fetch = fetchMock as typeof fetch;

    const { result } = renderHook(() => useCompleteText());
    let completionPromise: Promise<void> = Promise.resolve();
    act(() => {
      completionPromise = result.current.completeText('Keep running');
    });
    await waitFor(() => expect(result.current.activeTurn?.run_id).toBe('run_reject'));

    await act(async () => {
      await result.current.cancelGeneration();
    });

    expect(streamSignal?.aborted).toBe(false);
    expect(result.current.activeTurn?.status).toBe('streaming');
    expect(result.current.error).toBe('The server did not accept the cancellation request.');

    act(() => result.current.resetChatSession());
    await completionPromise;
  });

  it('resets the local session identity for New Chat', async () => {
    global.fetch = jest.fn().mockResolvedValue(streamingResponse([
      'event: run_started\ndata: {"run_id":"run_7","chat_id":null}\n\n' +
      'event: final\ndata: {"run_id":"run_7","message":["done"],"chat_id":7}\n\n' +
      'event: done\ndata: {"run_id":"run_7","chat_id":7}\n\n',
    ])) as typeof fetch;

    const { result } = renderHook(() => useCompleteText());
    await act(async () => {
      await result.current.completeText('First chat');
    });
    expect(result.current.state_chat_id).toBe(7);

    act(() => result.current.resetChatSession());

    expect(result.current.state_chat_id).toBeNull();
    expect(result.current.activeTurn).toBeNull();
    expect(result.current.completedTurn).toBeNull();
  });
});
