import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import Files from './Files';

jest.mock('./Components/FileUpload', () => ({ __esModule: true,
  default: ({ onUploadStored }: { onUploadStored: () => void }) => (
    <button onClick={onUploadStored}>Notify stored upload</button>
  ),
}));
const originalFetch = global.fetch;
afterEach(() => { global.fetch = originalFetch; });

it.each([false, true])('refreshes a stored file without closing the upload batch (processed=%s)', async (processed) => {
  global.fetch = jest.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ files: [] }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ files: [{
      file_id: 1, original_filename: 'stored.txt', filename: 'stored.txt',
      file_size: 5, mime_type: 'text/plain', upload_date: '2026-10-13T00:00:00Z',
      is_processed: processed, processing_error: processed ? 'Extraction failed' : null,
    }] }) });
  render(<Files />);
  await screen.findByText('No files uploaded yet.');
  fireEvent.click(screen.getByRole('button', { name: 'Upload Files' }));
  fireEvent.click(screen.getByRole('button', { name: 'Notify stored upload' }));
  await screen.findByRole('heading', { name: 'stored.txt' });
  expect(screen.getByRole('button', { name: 'Hide Upload' })).toBeInTheDocument();
  expect(screen.getByText(processed ? 'Extraction failed' : 'Processing')).toBeInTheDocument();
});
