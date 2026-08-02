import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RoutinesSection from '../RoutinesSection';

const routinesResponse = [
  {
    routine_id: 1,
    name: 'Morning digest',
    prompt: 'Summarize my documents',
    interval_minutes: 60,
    enabled: true,
    last_run_at: null,
    next_run_at: '2026-08-02T13:00:00Z',
  },
];

describe('RoutinesSection', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('lists routines from the API', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => routinesResponse,
    });

    render(<RoutinesSection />);

    expect(screen.getByText(/Loading routines/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('Morning digest')).toBeInTheDocument();
    });
    expect(screen.getByText(/every 60 min/i)).toBeInTheDocument();
  });

  it('creates a routine and refreshes the list', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: true, json: async () => [] }) // initial list
      .mockResolvedValueOnce({ ok: true, json: async () => routinesResponse[0] }) // create
      .mockResolvedValueOnce({ ok: true, json: async () => routinesResponse }); // refresh

    render(<RoutinesSection />);
    await waitFor(() => screen.getByText(/No routines yet/i));

    fireEvent.change(screen.getByLabelText('Name'), {
      target: { value: 'Morning digest' },
    });
    fireEvent.change(screen.getByLabelText('Prompt'), {
      target: { value: 'Summarize my documents' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Create Routine' }));

    await waitFor(() => {
      expect(screen.getByText('Morning digest')).toBeInTheDocument();
    });

    const createCall = (global.fetch as jest.Mock).mock.calls[1];
    expect(createCall[0]).toBe('/api/v1/routines/');
    expect(JSON.parse(createCall[1].body)).toEqual({
      name: 'Morning digest',
      prompt: 'Summarize my documents',
      interval_minutes: 60,
    });
  });

  it('deletes a routine after confirmation', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: true, json: async () => routinesResponse })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ deleted: 1 }) })
      .mockResolvedValueOnce({ ok: true, json: async () => [] });
    jest.spyOn(window, 'confirm').mockReturnValue(true);

    render(<RoutinesSection />);
    await waitFor(() => screen.getByText('Morning digest'));

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(screen.getByText(/No routines yet/i)).toBeInTheDocument();
    });
    const deleteCall = (global.fetch as jest.Mock).mock.calls[1];
    expect(deleteCall[0]).toBe('/api/v1/routines/1');
    expect(deleteCall[1].method).toBe('DELETE');
  });
});
