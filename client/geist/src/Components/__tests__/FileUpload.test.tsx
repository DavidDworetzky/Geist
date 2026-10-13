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
