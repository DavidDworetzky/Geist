import { useCallback, useEffect, useState } from 'react';

export interface McpServer {
  mcp_server_id: number;
  user_id: number;
  name: string;
  transport: 'stdio' | 'http';
  command: string | null;
  args: string[];
  env: Record<string, string>;
  working_directory: string | null;
  url: string | null;
  headers: Record<string, string>;
  connector_kind: ConnectorKind;
  account_label: string | null;
  trusted: boolean;
  security_required: boolean;
  recipient_allowlist: string[];
  max_writes_per_hour: number;
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
  working_directory?: string | null;
  url?: string | null;
  headers?: Record<string, string>;
  connector_kind?: ConnectorKind;
  account_label?: string | null;
  trusted?: boolean;
  recipient_allowlist?: string[];
  max_writes_per_hour?: number;
  enabled?: boolean;
  timeout_seconds?: number;
}

export type EmailConnectorKind = 'gmail' | 'google_workspace' | 'outlook' | 'proton';
export type ConnectorKind = 'custom' | EmailConnectorKind;

export interface EmailConnectorProfile {
  kind: EmailConnectorKind;
  label: string;
  account_scope: string;
  authentication: string;
  requirements: string[];
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
  connectorProfiles: EmailConnectorProfile[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  createServer: (input: McpServerInput) => Promise<McpServer>;
  createEmailConnector: (input: McpServerInput) => Promise<McpServer>;
  updateServer: (id: number, updates: Partial<McpServerInput>) => Promise<McpServer>;
  deleteServer: (id: number) => Promise<void>;
  testServer: (id: number) => Promise<McpTestResult>;
}

export const useMcpServers = (): UseMcpServersReturn => {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [connectorProfiles, setConnectorProfiles] = useState<EmailConnectorProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [serversResponse, profilesResponse] = await Promise.all([
        fetch(`${BASE_URL}/servers`),
        fetch(`${BASE_URL}/email-connectors/profiles`),
      ]);
      if (!serversResponse.ok) {
        throw new Error(await parseError(serversResponse));
      }
      if (!profilesResponse.ok) {
        throw new Error(await parseError(profilesResponse));
      }
      setServers(await serversResponse.json());
      const profiles = await profilesResponse.json();
      const profileKinds = new Set(['gmail', 'google_workspace', 'outlook', 'proton']);
      setConnectorProfiles(
        Array.isArray(profiles)
          ? profiles.filter((profile) => profileKinds.has(profile?.kind))
          : []
      );
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

  const createEmailConnector = useCallback(
    async (input: McpServerInput): Promise<McpServer> => {
      const response = await fetch(`${BASE_URL}/email-connectors`, {
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

  return {
    servers,
    connectorProfiles,
    loading,
    error,
    refetch,
    createServer,
    createEmailConnector,
    updateServer,
    deleteServer,
    testServer,
  };
};
