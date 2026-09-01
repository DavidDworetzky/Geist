import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import Tools from './Tools';

const jsonResponse = (body: unknown, status = 200) =>
  Promise.resolve({
    ok: status < 400,
    status,
    json: () => Promise.resolve(body),
  } as Response);

describe('Tools', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  it('shows the live tool catalogue and its schema', async () => {
    global.fetch = jest.fn(() => jsonResponse({
      tools: [{
        name: 'web.search',
        description: 'Search current public web information.',
        input_schema: { type: 'object', properties: { query: { type: 'string' } } },
        enabled: true,
        enabled_by_default: true,
        requires_approval: false,
        side_effect: 'read',
        source_adapter: 'SearchAdapter.search',
        semantic_tags: ['retrieval'],
      }],
    })) as any;

    render(<Tools />);

    expect(await screen.findByRole('heading', { name: 'web.search' })).toBeInTheDocument();
    expect(screen.getByText('Available')).toBeInTheDocument();
    expect(screen.getByText('Retrieval')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Input schema'));
    expect(screen.getByText(/"query"/)).toBeInTheDocument();
  });

  it('opens MCP configuration from the top-level tab', async () => {
    global.fetch = jest.fn((url: string) => {
      if (url === '/agent/tools') return jsonResponse({ tools: [] });
      if (url === '/api/v1/mcp/servers') return jsonResponse([]);
      return jsonResponse({}, 404);
    }) as any;

    render(<Tools />);
    fireEvent.click(screen.getByRole('tab', { name: 'MCP servers' }));

    expect(await screen.findByText('No MCP servers configured yet.')).toBeInTheDocument();
    expect(screen.getByTestId('mcp-catalogue')).toHaveTextContent('Gmail');
    expect(screen.getByTestId('mcp-catalogue')).toHaveTextContent('Proton Mail');
  });
});
