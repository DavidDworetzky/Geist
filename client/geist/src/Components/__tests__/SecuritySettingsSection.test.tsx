import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SecuritySettingsSection from '../SecuritySettingsSection';

const policy = {
  mcp_security_policy_id: 1,
  user_id: 1,
  enabled: true,
  inspect_tool_metadata: true,
  inspect_outbound_arguments: true,
  inspect_inbound_results: true,
  deterministic_scanner: true,
  model_mode: 'mirror',
  create_date: '2026-08-19T00:00:00Z',
  update_date: '2026-08-19T00:00:00Z',
};

const response = (body: unknown) => Promise.resolve({
  ok: true,
  status: 200,
  json: () => Promise.resolve(body),
} as Response);

it('shows default inspection surfaces and persists toggles', async () => {
  const fetchMock = jest.fn((_url: string, options?: RequestInit) =>
    response(options?.method === 'PUT' ? { ...policy, deterministic_scanner: false } : policy)
  );
  global.fetch = fetchMock as typeof fetch;

  render(<SecuritySettingsSection />);

  expect(await screen.findByText('Connector Security')).toBeInTheDocument();
  expect(screen.getByText('Mirrors the active chat model; tools disabled')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Instruction-pattern tripwires' }));

  await waitFor(() => {
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/security/policy',
      expect.objectContaining({ method: 'PUT' })
    );
  });
});
