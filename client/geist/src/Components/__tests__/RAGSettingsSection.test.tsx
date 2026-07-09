import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RAGSettingsSection from '../RAGSettingsSection';

const filesResponse = {
  files: [
    { file_id: 101, filename: 'a.txt', original_filename: 'a.txt' },
    { file_id: 102, filename: 'b.txt', original_filename: 'b.txt' },
  ],
};

describe('RAGSettingsSection', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('loads and displays files, allows selection', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => filesResponse });

    const onEnableRagChange = jest.fn();
    const onFileArchivesChange = jest.fn();

    render(
      <RAGSettingsSection
        enableRagByDefault={true}
        defaultFileArchives={[]}
        onEnableRagChange={onEnableRagChange}
        onFileArchivesChange={onFileArchivesChange}
      />
    );

    expect(screen.getByText(/Loading files/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('a.txt')).toBeInTheDocument();
      expect(screen.getByText('b.txt')).toBeInTheDocument();
    });

    // click first item
    fireEvent.click(screen.getByText('a.txt'));
    expect(onFileArchivesChange).toHaveBeenCalledWith([101]);
  });

  it('select all and clear all buttons work', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => filesResponse });

    const onFileArchivesChange = jest.fn();

    render(
      <RAGSettingsSection
        enableRagByDefault={true}
        defaultFileArchives={[]}
        onEnableRagChange={() => {}}
        onFileArchivesChange={onFileArchivesChange}
      />
    );

    await waitFor(() => screen.getByText('a.txt'));

    fireEvent.click(screen.getByText(/Select All/i));
    expect(onFileArchivesChange).toHaveBeenCalledWith([101, 102]);

    fireEvent.click(screen.getByText(/Clear All/i));
    expect(onFileArchivesChange).toHaveBeenCalledWith([]);
  });
});
