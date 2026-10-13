const wait = (milliseconds: number, signal: AbortSignal) => new Promise<void>((resolve, reject) => {
  const abort = () => {
    window.clearTimeout(timer);
    reject(new DOMException('Aborted', 'AbortError'));
  };
  const timer = window.setTimeout(() => {
    signal.removeEventListener('abort', abort);
    resolve();
  }, milliseconds);
  if (signal.aborted) abort();
  else signal.addEventListener('abort', abort, { once: true });
});

// False means observation stopped, not that extraction failed.
export async function waitForFileJob(jobId: number, signal: AbortSignal): Promise<boolean> {
  const polling = new AbortController();
  const abort = () => polling.abort();
  signal.addEventListener('abort', abort, { once: true });
  if (signal.aborted) abort();
  const deadline = window.setTimeout(abort, 120_000);
  let delay = 1000;
  let failures = 0;
  try {
    while (!polling.signal.aborted) {
      let job: { status: string; error?: string } | undefined;
      try {
        const response = await fetch(`/api/v1/jobs/${jobId}`, { signal: polling.signal });
        if (!response.ok) {
          if (response.status < 500 && response.status !== 429) {
            return false;
          }
        } else {
          job = await response.json();
        }
      } catch {
        if (polling.signal.aborted) return false;
        // Network errors and temporarily unavailable responses are retried below.
      }
      if (job) {
        failures = 0;
        if (job.status === 'failed') throw new Error(job.error || 'File processing failed.');
        if (job.status === 'succeeded') return true;
      } else if (++failures >= 3) {
        return false;
      }
      await wait(delay, polling.signal);
      delay = Math.min(delay * 2, 8000);
    }
    return false;
  } catch (error) {
    if (polling.signal.aborted) return false;
    throw error;
  } finally {
    window.clearTimeout(deadline);
    signal.removeEventListener('abort', abort);
    polling.abort();
  }
}
