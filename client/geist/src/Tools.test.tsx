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
      tools: [
        {
          name: 'web.search',
          description: 'Search current public web information.',
          input_schema: { type: 'object', properties: { query: { type: 'string' } } },
          enabled: true,
          enabled_by_default: true,
          requires_approval: false,
          side_effect: 'read',
          source_adapter: 'SearchAdapter.search',
          semantic_tags: ['public_retrieval'],
        },
        {
          name: 'image.generate',
          description: 'Generate an image.',
          input_schema: { type: 'object', properties: {} },
          enabled: false,
          enabled_by_default: true,
          requires_approval: false,
          side_effect: 'external_write',
          semantic_tags: ['image_generation'],
        },
        {
          name: 'workspace.read_markdown',
          description: 'Read a Markdown file.',
          input_schema: { type: 'object', properties: {} },
          enabled: false,
          enabled_by_default: false,
          requires_approval: false,
          side_effect: 'read',
          semantic_tags: ['local_retrieval'],
        },
      ],
    })) as any;

    render(<Tools />);

    expect(await screen.findByRole('heading', { name: 'web.search' })).toBeInTheDocument();
    expect(screen.getAllByText('Available')).not.toHaveLength(0);
    expect(screen.getByText('Public retrieval')).toBeInTheDocument();
    expect(screen.getByText('Image')).toBeInTheDocument();
    expect(screen.getByText('Local retrieval')).toBeInTheDocument();
    expect(screen.getAllByText('Opt-in')).not.toHaveLength(0);
    expect(screen.getAllByText('Unavailable')).not.toHaveLength(0);
    expect(screen.getByText(/provider or local prerequisite is not configured/i)).toBeInTheDocument();
    expect(screen.getByText(/must be explicitly enabled in Geist configuration/i)).toBeInTheDocument();
    fireEvent.click(screen.getAllByText('Input schema')[0]);
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
