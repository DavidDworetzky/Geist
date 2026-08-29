import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ProviderKeysSection from '../ProviderKeysSection';

const openaiStatus = {
  id: 'openai',
  name: 'OpenAI',
  description: 'API key for OpenAI chat completions.',
  api_key_env: 'OPENAI_API_KEY',
  env_configured: false,
  has_stored_key: false,
  key_hint: null,
  supports_base_url: false,
  base_url: null,
  updated_at: null,
};

const selfHostedStatus = {
  id: 'self-hosted',
  name: 'Self-hosted OpenAI-compatible',
  description: 'API key for Self-hosted OpenAI-compatible chat completions.',
  api_key_env: 'API_KEY',
  env_configured: true,
  has_stored_key: false,
  key_hint: null,
  supports_base_url: true,
  base_url: null,
  updated_at: null,
};

describe('ProviderKeysSection', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  const mockFetch = (handlers: Record<string, (init?: RequestInit) => any>) => {
    global.fetch = jest.fn(async (url: any, init?: RequestInit) => {
      const key = `${init?.method || 'GET'} ${url}`;
      const handler = handlers[key];
      if (!handler) {
        throw new Error(`Unexpected fetch: ${key}`);
      }
      return handler(init);
    }) as any;
  };

  const okJson = (body: any) => ({
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
  });

  it('lists providers with their configuration state', async () => {
    mockFetch({
      'GET /api/v1/providers/': () => okJson([openaiStatus, selfHostedStatus]),
    });

    render(<ProviderKeysSection />);

    expect(await screen.findByText('OpenAI')).toBeInTheDocument();
    expect(screen.getByText('Not configured')).toBeInTheDocument();
    expect(screen.getByText('From environment (API_KEY)')).toBeInTheDocument();
    // Self-hosted rows expose a base URL field.
    expect(
      screen.getByLabelText('Self-hosted OpenAI-compatible base URL')
    ).toBeInTheDocument();
    expect(screen.queryByLabelText('OpenAI base URL')).not.toBeInTheDocument();
  });

  it('saves a key and shows the masked stored state', async () => {
    const storedStatus = {
      ...openaiStatus,
      has_stored_key: true,
      key_hint: '****1234',
    };
    mockFetch({
      'GET /api/v1/providers/': () => okJson([openaiStatus]),
      'PUT /api/v1/providers/openai/key': (init) => {
        expect(JSON.parse(String(init?.body))).toEqual({
          api_key: 'sk-test-key-1234',
          base_url: null,
        });
        return okJson(storedStatus);
      },
    });

    render(<ProviderKeysSection />);

    const input = await screen.findByLabelText('OpenAI API key');
    fireEvent.change(input, { target: { value: 'sk-test-key-1234' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save Key' }));

    expect(await screen.findByText('Key saved.')).toBeInTheDocument();
    expect(screen.getByText('Stored key ****1234')).toBeInTheDocument();
    // The password input clears after a successful save.
    expect((screen.getByLabelText('OpenAI API key') as HTMLInputElement).value).toBe('');
  });

  it('surfaces backend validation errors without clearing the input', async () => {
    mockFetch({
      'GET /api/v1/providers/': () => okJson([openaiStatus]),
      'PUT /api/v1/providers/openai/key': () => ({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => ({ detail: 'API key must not be empty' }),
      }),
    });

    render(<ProviderKeysSection />);

    const input = await screen.findByLabelText('OpenAI API key');
    fireEvent.change(input, { target: { value: 'bad-key' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save Key' }));

    expect(await screen.findByText('API key must not be empty')).toBeInTheDocument();
  });

  it('removes a stored key', async () => {
    const storedStatus = { ...openaiStatus, has_stored_key: true, key_hint: '****5678' };
    mockFetch({
      'GET /api/v1/providers/': () => okJson([storedStatus]),
      'DELETE /api/v1/providers/openai/key': () => okJson(openaiStatus),
    });

    render(<ProviderKeysSection />);

    fireEvent.click(await screen.findByRole('button', { name: 'Remove' }));

    await waitFor(() => {
      expect(screen.getByText('Not configured')).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: 'Remove' })).not.toBeInTheDocument();
  });
});
