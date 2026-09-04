import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import Schedules from './Schedules';

const schedule = {
  prompt_schedule_id: 7,
  user_id: 1,
  name: 'Morning briefing',
  prompt: 'Summarize my priorities.',
  cron_expression: '0 9 * * 1-5',
  timezone: 'America/Chicago',
  enabled: true,
  inference_config: {},
  next_run_at: '2026-09-04T14:00:00',
  last_enqueued_at: null,
  created_at: '2026-09-03T12:00:00',
  updated_at: '2026-09-03T12:00:00',
};

beforeEach(() => {
  (global.fetch as jest.Mock) = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => [schedule],
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

test('lists configured cron tasks and exposes management actions', async () => {
  render(<Schedules />);

  expect(await screen.findByText('Morning briefing')).toBeInTheDocument();
  expect(screen.getByText('0 9 * * 1-5')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Run now' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();
});

test('creates a schedule from the form', async () => {
  (global.fetch as jest.Mock)
    .mockResolvedValueOnce({ ok: true, json: async () => [] })
    .mockResolvedValueOnce({ ok: true, json: async () => schedule })
    .mockResolvedValueOnce({ ok: true, json: async () => [schedule] });

  render(<Schedules />);
  await screen.findByText('No cron tasks configured yet.');
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Morning briefing' } });
  fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: 'Summarize.' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create schedule' }));

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(3));
  expect(global.fetch).toHaveBeenNthCalledWith(
    2,
    '/api/v1/prompt-schedules/',
    expect.objectContaining({ method: 'POST' }),
  );
  expect(await screen.findByText('Morning briefing')).toBeInTheDocument();
});
