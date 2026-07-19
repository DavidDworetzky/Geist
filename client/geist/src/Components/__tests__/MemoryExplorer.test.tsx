import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import MemoryExplorer from '../MemoryExplorer';


const response = (body: unknown = {}) => Promise.resolve({
  ok: true,
  json: async () => body,
} as Response);


describe('MemoryExplorer', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('reviews, corrects, and forgets profile facts', async () => {
    const fetchMock = jest.fn()
      .mockImplementationOnce(() => response({
        facts: [{ memory_id: 7, content: 'Remember cobalt.' }],
      }))
      .mockImplementationOnce(() => response({
        memory_id: 7,
        content: 'Remember cerulean.',
      }))
      .mockImplementationOnce(() => response({
        facts: [{ memory_id: 7, content: 'Remember cerulean.' }],
      }))
      .mockImplementationOnce(() => response())
      .mockImplementationOnce(() => response({ facts: [] }));
    global.fetch = fetchMock as typeof fetch;
    render(<MemoryExplorer scope="user" folderId={null} />);

    fireEvent.click(screen.getByText('Profile memory'));
    expect(await screen.findByText('Remember cobalt.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Edit saved memory' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Edit memory' }), {
      target: { value: 'Remember cerulean.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save memory' }));
    expect(await screen.findByText('Remember cerulean.')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Forget saved memory' }));
    await waitFor(() => expect(screen.queryByText('Remember cerulean.')).not.toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/memory/records/7', {
      method: 'DELETE',
    });
  });

  it('runs search inside the selected folder scope', async () => {
    const fetchMock = jest.fn().mockImplementationOnce(() => response({
      results: [{
        memory_id: 9,
        content: 'Topaz launches Friday.',
        record_type: 'folder_summary',
        score: 0.9,
      }],
    }));
    global.fetch = fetchMock as typeof fetch;
    render(<MemoryExplorer scope="folder" folderId={3} />);

    fireEvent.click(screen.getByText('Folder memory'));
    fireEvent.change(screen.getByRole('textbox', { name: 'Search memory' }), {
      target: { value: 'topaz' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run memory search' }));

    expect(await screen.findByText('Topaz launches Friday.')).toBeInTheDocument();
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({
      query: 'topaz',
      scope: 'folder',
      folder_id: 3,
    });
  });
});
