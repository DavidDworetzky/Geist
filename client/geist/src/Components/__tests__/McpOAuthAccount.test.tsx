import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import McpOAuthAccount from '../McpOAuthAccount';

const originalFetch = global.fetch;
const provider = { id: 'google', label: 'Google', scopes: ['gmail.readonly'], ready: true, setup_message: null };
const disconnected = { provider: null, status: 'disconnected', providers: [provider], redirect_uri: 'http://localhost:3000/api/v1/mcp/oauth/callback' };
const jsonResponse = (body: any, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => body } as Response);

afterEach(() => { global.fetch = originalFetch; });

it('shows missing app registration and the registered callback', async () => {
  global.fetch = jest.fn(() => jsonResponse({ ...disconnected, providers: [{ ...provider, ready: false, setup_message: 'Configure GEIST_GOOGLE_OAUTH_CLIENT_ID on the backend' }] })) as any;
  render(<McpOAuthAccount serverId={1} onChange={async () => {}} />);
  expect(await screen.findByText(/Configure GEIST_GOOGLE_OAUTH_CLIENT_ID/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Connect with Google' })).toBeDisabled();
  expect(screen.getByText(disconnected.redirect_uri)).toBeInTheDocument();
});

it('starts OAuth and surfaces a provider error without losing the server', async () => {
  const fetchMock = jest.fn((url: string) => url.endsWith('/start')
    ? jsonResponse({ detail: 'Open Geist at GEIST_PUBLIC_URL before connecting an account' }, 409)
    : jsonResponse(disconnected));
  global.fetch = fetchMock as any;
  render(<McpOAuthAccount serverId={7} onChange={async () => {}} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Connect with Google' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Open Geist at GEIST_PUBLIC_URL');
  expect(fetchMock).toHaveBeenCalledWith('/api/v1/mcp/servers/7/oauth/start', expect.objectContaining({
    method: 'POST', body: JSON.stringify({ provider: 'google' }),
  }));
});

it('offers reconnect after revoked credentials', async () => {
  global.fetch = jest.fn(() => jsonResponse({ ...disconnected, provider: 'google', status: 'reconnect_required' })) as any;
  render(<McpOAuthAccount serverId={1} onChange={async () => {}} />);
  expect(await screen.findByText('Reconnect your account to restore access.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Reconnect with Google' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Disconnect account' })).toBeInTheDocument();
});

it('disconnects the account and refreshes the server list', async () => {
  let connected = true;
  global.fetch = jest.fn((url: string, options?: RequestInit) => {
    if (options?.method === 'DELETE') { connected = false; return jsonResponse({ revocation_complete: false }); }
    return jsonResponse(connected ? { ...disconnected, provider: 'google', status: 'connected' } : disconnected);
  }) as any;
  const onChange = jest.fn(async () => {});
  render(<McpOAuthAccount serverId={1} onChange={onChange} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Disconnect account' }));
  expect(await screen.findByText(/Account disconnected and tools disabled/)).toBeInTheDocument();
  await waitFor(() => expect(onChange).toHaveBeenCalledTimes(1));
  expect(screen.queryByRole('button', { name: 'Disconnect account' })).not.toBeInTheDocument();
});

it('uses the label supplied by another registered OAuth provider', async () => {
  global.fetch = jest.fn(() => jsonResponse({ ...disconnected, providers: [{ ...provider, id: 'work-mail', label: 'Work Mail' }] })) as any;
  render(<McpOAuthAccount serverId={1} onChange={async () => {}} />);
  expect(await screen.findByRole('button', { name: 'Connect with Work Mail' })).toBeInTheDocument();
});
