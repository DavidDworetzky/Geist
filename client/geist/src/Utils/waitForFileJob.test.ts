import { waitForFileJob } from './waitForFileJob';

const originalFetch = global.fetch;
beforeEach(() => jest.useFakeTimers());
afterEach(() => { global.fetch = originalFetch; jest.useRealTimers(); });
const advance = async (ms: number) => {
  jest.advanceTimersByTime(ms);
  for (let i = 0; i < 10; i++) await Promise.resolve();
};
it('backs off and tolerates transient server and network failures', async () => {
  global.fetch = jest.fn().mockResolvedValueOnce({ ok: false, status: 503 })
    .mockRejectedValueOnce(new TypeError('network'))
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'succeeded' }) });
  const result = waitForFileJob(3, new AbortController().signal);
  await advance(0);
  await advance(1000);
  expect(global.fetch).toHaveBeenCalledTimes(2);
  await advance(1999);
  expect(global.fetch).toHaveBeenCalledTimes(2);
  await advance(1);
  expect(await result).toBe(true);
  expect(jest.getTimerCount()).toBe(0);
});
it('stops at its deadline without claiming that the job failed', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'queued' }) });
  const result = waitForFileJob(3, new AbortController().signal);
  await advance(0);
  await advance(120000);
  expect(await result).toBe(false);
  expect(jest.getTimerCount()).toBe(0);
});
it('stops after three failed observations', async () => {
  global.fetch = jest.fn().mockRejectedValue(new TypeError('offline'));
  const result = waitForFileJob(3, new AbortController().signal);
  await advance(0); await advance(1000); await advance(2000);
  expect(await result).toBe(false);
  expect(global.fetch).toHaveBeenCalledTimes(3);
});
it('aborts an in-flight status request when the deadline expires', async () => {
  global.fetch = jest.fn((_url, options) => new Promise((_resolve, reject) => {
    options?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
  }));
  const result = waitForFileJob(3, new AbortController().signal);
  await advance(120000);
  expect(await result).toBe(false);
  expect(jest.getTimerCount()).toBe(0);
});

it('clears the polling timer on unmount cancellation', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'running' }) });
  const controller = new AbortController();
  const result = waitForFileJob(3, controller.signal);
  await advance(0);
  controller.abort();
  expect(await result).toBe(false);
  expect(jest.getTimerCount()).toBe(0);
});
