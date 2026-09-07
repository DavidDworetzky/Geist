import { useCallback, useEffect, useState } from 'react';

export interface McpServer {
  mcp_server_id: number;
  workspace_id: number;
  name: string;
  transport: 'stdio' | 'http';
  command: string | null;
  args: string[];
  env: Record<string, string>;
  url: string | null;
  headers: Record<string, string>;
  enabled: boolean;
  timeout_seconds: number;
  create_date: string;
  update_date: string;
}

export interface McpServerInput {
  name: string;
  transport: 'stdio' | 'http';
  command?: string | null;
  args?: string[];
  env?: Record<string, string>;
  url?: string | null;
  headers?: Record<string, string>;
  enabled?: boolean;
  timeout_seconds?: number;
}

export interface McpToolInfo {
  name: string;
  description: string;
}

export interface McpTestResult {
  ok: boolean;
  error?: string | null;
  tools: McpToolInfo[];
}

const BASE_URL = '/api/v1/mcp';

const parseError = async (response: Response): Promise<string> => {
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') {
      return body.detail;
    }
    return JSON.stringify(body?.detail ?? body);
  } catch {
    return `Request failed with status ${response.status}`;
  }
};

export interface UseMcpServersReturn {
  servers: McpServer[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  createServer: (input: McpServerInput) => Promise<McpServer>;
  updateServer: (id: number, updates: Partial<McpServerInput>) => Promise<McpServer>;
  deleteServer: (id: number) => Promise<void>;
  testServer: (id: number) => Promise<McpTestResult>;
}

export const useMcpServers = (): UseMcpServersReturn => {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${BASE_URL}/servers`);
      if (!response.ok) {
        throw new Error(await parseError(response));
      }
      setServers(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load MCP servers');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refetch();
  }, [refetch]);

  const createServer = useCallback(
    async (input: McpServerInput): Promise<McpServer> => {
      const response = await fetch(`${BASE_URL}/servers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      });
      if (!response.ok) {
        throw new Error(await parseError(response));
      }
      const created = await response.json();
      await refetch();
      return created;
    },
    [refetch]
  );

  const updateServer = useCallback(
    async (id: number, updates: Partial<McpServerInput>): Promise<McpServer> => {
      const response = await fetch(`${BASE_URL}/servers/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!response.ok) {
        throw new Error(await parseError(response));
      }
      const updated = await response.json();
      await refetch();
      return updated;
    },
    [refetch]
  );

  const deleteServer = useCallback(
    async (id: number): Promise<void> => {
      const response = await fetch(`${BASE_URL}/servers/${id}`, { method: 'DELETE' });
      if (!response.ok) {
        throw new Error(await parseError(response));
      }
      await refetch();
    },
    [refetch]
  );

  const testServer = useCallback(async (id: number): Promise<McpTestResult> => {
    const response = await fetch(`${BASE_URL}/servers/${id}/test`, { method: 'POST' });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  }, []);

  return { servers, loading, error, refetch, createServer, updateServer, deleteServer, testServer };
};
