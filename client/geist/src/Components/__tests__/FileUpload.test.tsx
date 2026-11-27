import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import FileUpload from '../FileUpload';

const originalFetch = global.fetch;
afterEach(() => { global.fetch = originalFetch; jest.restoreAllMocks(); });

it('waits for background extraction before announcing completion', async () => {
  let resolveJob: (value: unknown) => void = () => {};
  const job = new Promise(resolve => { resolveJob = resolve; });
  global.fetch = jest.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ file_id: 1, processing_status: { job_id: 3, status: 'queued' } }) })
    .mockReturnValueOnce(job);
  const complete = jest.fn();
  render(<FileUpload acceptedFileTypes={['text/plain']} onUploadComplete={complete} />);
  fireEvent.change(screen.getByLabelText('Choose files to upload'), { target: { files: [new File(['hello'], 'notes.txt', { type: 'text/plain' })] } });
  await screen.findByText('notes.txt uploaded. Processing text...');
  expect(complete).not.toHaveBeenCalled();
  await act(async () => { resolveJob({ ok: true, json: async () => ({ status: 'succeeded' }) }); });
  await waitFor(() => expect(complete).toHaveBeenCalledTimes(1));
  expect(global.fetch).toHaveBeenLastCalledWith('/api/v1/jobs/3', expect.objectContaining({ signal: expect.any(AbortSignal) }));
});

it('reports extraction failures without marking the file ready', async () => {
  global.fetch = jest.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ processing_status: { job_id: 3 } }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'failed', error: 'Unsupported content' }) });
  const complete = jest.fn();
  const error = jest.fn();
  render(<FileUpload acceptedFileTypes={['text/plain']} onUploadComplete={complete} onUploadError={error} />);
  fireEvent.change(screen.getByLabelText('Choose files to upload'), { target: { files: [new File(['hello'], 'notes.txt', { type: 'text/plain' })] } });
  await waitFor(() => expect(error).toHaveBeenCalledWith('Unsupported content'));
  expect(complete).not.toHaveBeenCalled();
});


it('does not let a superseded batch upload remaining files or cancel the new batch', async () => {
  const posted: string[] = [];
  let newSignal: AbortSignal | null | undefined;
  let finishNew: (value: Response) => void = () => {};
  global.fetch = jest.fn((url, options) => {
    if (url === '/api/v1/files/upload') {
      const name = ((options?.body as FormData).get('file') as File).name;
      posted.push(name);
      return Promise.resolve({ ok: true, json: async () => ({ filename: name, processing_status: { job_id: name === 'a.txt' ? 1 : 2 } }) } as Response);
    }
    if (url === '/api/v1/jobs/1') {
      return new Promise((_resolve, reject) => {
        options?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
      });
    }
    newSignal = options?.signal;
    return new Promise(resolve => { finishNew = resolve; });
  });
  const complete = jest.fn();
  render(<FileUpload multiple acceptedFileTypes={['text/plain']} onUploadComplete={complete} />);
  const file = (name: string) => new File(['hi'], name, { type: 'text/plain' });
  fireEvent.change(screen.getByLabelText('Choose files to upload'), { target: { files: [file('a.txt'), file('a2.txt')] } });
  await screen.findByText('a.txt uploaded. Processing text...');
  fireEvent.change(screen.getByLabelText('Choose files to upload'), { target: { files: [file('b.txt')] } });
  await screen.findByText('b.txt uploaded. Processing text...');
  expect(posted).toEqual(['a.txt', 'b.txt']);
  expect(newSignal?.aborted).toBe(false);
  await act(async () => { finishNew({ ok: true, json: async () => ({ status: 'succeeded' }) } as Response); });
  await waitFor(() => expect(complete).toHaveBeenCalledTimes(1));
  expect(complete).toHaveBeenCalledWith(expect.objectContaining({ filename: 'b.txt' }));
});

it.each(['failed', 'unavailable'])('notifies the file list after storage and %s processing observation', async (status) => {
  global.fetch = jest.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ file_id: 1, processing_status: { job_id: 3 } }) })
    .mockResolvedValueOnce(status === 'failed'
      ? { ok: true, json: async () => ({ status: 'failed', error: 'Cannot extract' }) }
      : { ok: false, status: 404 });
  const stored = jest.fn();
  const complete = jest.fn();
  render(<FileUpload acceptedFileTypes={['text/plain']} onUploadStored={stored} onUploadComplete={complete} />);
  fireEvent.change(screen.getByLabelText('Choose files to upload'), { target: { files: [new File(['hello'], 'notes.txt', { type: 'text/plain' })] } });
  await waitFor(() => expect(stored).toHaveBeenCalledTimes(2));
  expect(complete).not.toHaveBeenCalled();
});
