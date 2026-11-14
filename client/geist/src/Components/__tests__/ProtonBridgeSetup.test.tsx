import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ProtonBridgeSetup from '../ProtonBridgeSetup';

afterEach(() => jest.restoreAllMocks());

it('submits Bridge fields to the bundled connector without a command', async () => {
  const onSaved = jest.fn().mockResolvedValue(undefined);
  const fetchMock = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true }) });
  global.fetch = fetchMock;
  render(<ProtonBridgeSetup onSaved={onSaved} onCancel={() => {}} />);
  for (const [label, value] of [['Bridge username', 'me@example.test'], ['Sender email address', 'me@example.test'], ['Bridge password', 'synthetic-password'], ['Bridge public TLS certificate', 'synthetic-cert']]) {
    fireEvent.change(screen.getByLabelText(label), { target: { value } });
  }
  fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
  expect(await screen.findByRole('status')).toHaveTextContent('No email was sent');
  expect(onSaved).not.toHaveBeenCalled();
  fireEvent.submit(screen.getByRole('form', { name: 'Proton Bridge connection' }));
  await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));
  expect(fetchMock.mock.calls[1][0]).toBe('/api/v1/mcp/proton-bridge');
  const body = JSON.parse(fetchMock.mock.calls[1][1].body);
  expect(body).toMatchObject({ password: 'synthetic-password', imap_port: 1143, smtp_port: 1025 });
  expect(body).not.toHaveProperty('command');
  expect(screen.getByLabelText('Bridge password')).toHaveValue('');
});

it('shows connection failures and keeps the form editable', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: false, json: async () => ({ detail: 'Cannot authenticate with Bridge' }) });
  const onSaved = jest.fn();
  render(<ProtonBridgeSetup onSaved={onSaved} onCancel={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Cannot authenticate with Bridge');
  expect(onSaved).not.toHaveBeenCalled();
  expect(screen.getByLabelText('Bridge username')).toBeEnabled();
});
