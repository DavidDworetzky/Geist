import React from 'react';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
import McpServersSection from '../McpServersSection';

const stdioServer = {
  mcp_server_id: 1,
  user_id: 1,
  name: 'filesystem',
  transport: 'stdio',
  command: 'npx',
  args: ['-y', '@modelcontextprotocol/server-filesystem'],
  env: {},
  url: null,
  headers: {},
  enabled: true,
  timeout_seconds: 30,
  create_date: '2026-07-28T00:00:00Z',
  update_date: '2026-07-28T00:00:00Z',
};

const jsonResponse = (body: any, status = 200) =>
  Promise.resolve({
    ok: status < 400,
    status,
    json: () => Promise.resolve(body),
  } as Response);

describe('McpServersSection', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  it('renders the configured servers returned by the API', async () => {
    global.fetch = jest.fn(() => jsonResponse([stdioServer])) as any;

    render(<McpServersSection />);

    expect(await screen.findByText('filesystem')).toBeInTheDocument();
    expect(
      screen.getByText('npx -y @modelcontextprotocol/server-filesystem')
    ).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/mcp/servers');
  });

  it('shows an empty state when no servers are configured', async () => {
    global.fetch = jest.fn(() => jsonResponse([])) as any;

    render(<McpServersSection />);

    expect(await screen.findByText('No MCP servers configured yet.')).toBeInTheDocument();
  });

  it('creates a server through the form', async () => {
    const fetchMock = jest.fn((url: string, options?: RequestInit) => {
      if (url === '/api/v1/mcp/servers' && options?.method === 'POST') {
        return jsonResponse({ ...stdioServer, mcp_server_id: 2, name: 'my-server' }, 201);
      }
      return jsonResponse([]);
    });
    global.fetch = fetchMock as any;

    render(<McpServersSection />);
    await screen.findByText('No MCP servers configured yet.');

    fireEvent.click(screen.getByText('Add MCP Server'));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'my-server' } });
    fireEvent.change(screen.getByLabelText('Command'), { target: { value: 'npx' } });
    await act(async () => {
      fireEvent.click(screen.getByText('Add Server'));
    });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/v1/mcp/servers',
        expect.objectContaining({ method: 'POST' })
      );
    });
    await waitFor(() => {
      expect(
        screen.queryByRole('heading', { name: 'Add MCP Server' })
      ).not.toBeInTheDocument();
    });
    const [, options] = fetchMock.mock.calls.find(
      ([url, opts]: [string, RequestInit?]) => opts?.method === 'POST'
    )!;
    expect(JSON.parse((options as RequestInit).body as string)).toMatchObject({
      name: 'my-server',
      transport: 'stdio',
      command: 'npx',
    });
  });

  it('requires a command before submitting a stdio server', async () => {
    global.fetch = jest.fn(() => jsonResponse([])) as any;

    render(<McpServersSection />);
    await screen.findByText('No MCP servers configured yet.');

    fireEvent.click(screen.getByText('Add MCP Server'));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'my-server' } });
    fireEvent.click(screen.getByText('Add Server'));

    expect(await screen.findByText('Local servers require a command')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1); // only the initial list fetch
  });

  it('runs a connection test and lists the discovered tools', async () => {
    const fetchMock = jest.fn((url: string, options?: RequestInit) => {
      if (url.endsWith('/test')) {
        return jsonResponse({
          ok: true,
          error: null,
          tools: [
            { name: 'read_file', description: '' },
            { name: 'write_file', description: '' },
          ],
        });
      }
      return jsonResponse([stdioServer]);
    });
    global.fetch = fetchMock as any;

    render(<McpServersSection />);
    await screen.findByText('filesystem');

    fireEvent.click(screen.getByText('Test'));

    const result = await screen.findByTestId('mcp-test-result-1');
    expect(result.textContent).toContain('2 tools available');
    expect(result.textContent).toContain('read_file');
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/mcp/servers/1/test', { method: 'POST' });
  });

  it('surfaces a failed connection test', async () => {
    const fetchMock = jest.fn((url: string) => {
      if (url.endsWith('/test')) {
        return jsonResponse({ ok: false, error: 'Could not start MCP server', tools: [] });
      }
      return jsonResponse([stdioServer]);
    });
    global.fetch = fetchMock as any;

    render(<McpServersSection />);
    await screen.findByText('filesystem');

    fireEvent.click(screen.getByText('Test'));

    const result = await screen.findByTestId('mcp-test-result-1');
    expect(result.textContent).toContain('Connection failed: Could not start MCP server');
  });
});
