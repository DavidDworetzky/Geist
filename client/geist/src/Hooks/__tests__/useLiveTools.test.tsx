import { act, renderHook } from '@testing-library/react';
import { TextDecoder } from 'util';
import useLiveTools from '../useLiveTools';
import { UserSettings } from '../useUserSettings';

Object.defineProperty(global, 'TextDecoder', { value: TextDecoder, writable: true });
const settings = { default_agent_type: 'online' } as UserSettings;
const response = (events: [string, object][]): Response => {
  let index = 0;
  return { ok: true, body: { getReader: () => ({ read: async () => index < events.length
    ? { done: false, value: new Uint8Array(Buffer.from(`event: ${events[index][0]}\ndata: ${JSON.stringify(events[index++][1])}\n\n`)) }
    : { done: true } }) } } as Response;
};

afterEach(() => jest.restoreAllMocks());

it('uses the real chat stream for catalog tools, retains results, and reuses the voice task chat', async () => {
  global.fetch = jest.fn(async () => response([
    ['run_started', { run_id: 'voice-run', chat_id: 42 }],
    ['tool_call', { id: 'call-1', name: 'catalog.lookup', arguments: {}, status: 'succeeded', result_summary: 'Three files' }],
    ['final', { run_id: 'voice-run', chat_id: 42, message: ['Found three files.'],
      tool_calls: [{ id: 'call-1', name: 'catalog.lookup', arguments: {}, status: 'succeeded' }] }],
    ['done', { run_id: 'voice-run', chat_id: 42 }],
  ])) as jest.Mock;
  const { result } = renderHook(() => useLiveTools(settings));
  let spoken = '';
  await act(async () => { spoken = await result.current.delegate('user: Find the files', new AbortController().signal); });
  expect(spoken).toBe('Found three files.');
  expect(result.current.turn?.tool_calls[0].name).toBe('catalog.lookup');
  const request = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
  expect(request).toMatchObject({ enable_tools: true, agent_type: 'HTTPAGENT', agentic_mode: false,
    memory_enabled: false, memory_mode: 'private' });
  expect(request.prompt).toContain('user: Find the files');
  await act(async () => { await result.current.delegate('user: Which is newest?', new AbortController().signal); });
  expect((global.fetch as jest.Mock).mock.calls[1][0]).toBe('/agent/complete_text_stream/42');
  act(() => result.current.reset());
  await act(async () => { await result.current.delegate('user: A new call', new AbortController().signal); });
  expect((global.fetch as jest.Mock).mock.calls[2][0]).toBe('/agent/complete_text_stream');
});

it('does not announce success or retry when the stream fails', async () => {
  global.fetch = jest.fn(async () => response([['error', { message: 'Backend unavailable' }]])) as jest.Mock;
  const { result } = renderHook(() => useLiveTools(settings));
  await act(async () => {
    await expect(result.current.delegate('user: Find files', new AbortController().signal)).rejects.toThrow('interrupted');
  });
  expect(result.current.error).toBe('Backend unavailable');
  expect(global.fetch).toHaveBeenCalledTimes(1);
});

it('cancels the backend stream when the voice session closes', async () => {
  let streamSignal: AbortSignal | undefined;
  global.fetch = jest.fn((_url, options) => {
    streamSignal = options.signal;
    return new Promise((_resolve, reject) => streamSignal?.addEventListener('abort', () => {
      const error = new Error('Aborted'); error.name = 'AbortError'; reject(error);
    }));
  }) as jest.Mock;
  const { result } = renderHook(() => useLiveTools(settings));
  const controller = new AbortController();
  await act(async () => {
    const task = result.current.delegate('user: Find files', controller.signal);
    controller.abort();
    await expect(task).rejects.toThrow('interrupted');
  });
  expect(streamSignal?.aborted).toBe(true);
  expect(result.current.turn).toMatchObject({ status: 'cancelled' });
});

it('sends explicit approval decisions to the existing authenticated endpoint', async () => {
  global.fetch = jest.fn(async () => ({ ok: true })) as jest.Mock;
  const { result } = renderHook(() => useLiveTools(settings));
  await result.current.approve('voice-run', 'call-1', 'deny');
  expect(global.fetch).toHaveBeenCalledWith('/agent/runs/voice-run/tool_approval', expect.objectContaining({
    method: 'POST', body: JSON.stringify({ call_id: 'call-1', decision: 'deny' }),
  }));
  (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 404 });
  await expect(result.current.approve('voice-run', 'call-1', 'approve')).rejects.toHaveProperty('name', 'ApprovalUnavailable');
});
