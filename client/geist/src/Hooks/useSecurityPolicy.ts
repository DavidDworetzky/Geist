import { useCallback, useEffect, useState } from 'react';

export interface SecurityPolicy {
  mcp_security_policy_id: number;
  user_id: number;
  enabled: boolean;
  inspect_tool_metadata: boolean;
  inspect_outbound_arguments: boolean;
  inspect_inbound_results: boolean;
  deterministic_scanner: boolean;
  model_mode: 'mirror';
  create_date: string;
  update_date: string;
}

export type SecurityPolicyUpdate = Partial<Pick<
  SecurityPolicy,
  | 'enabled'
  | 'inspect_tool_metadata'
  | 'inspect_outbound_arguments'
  | 'inspect_inbound_results'
  | 'deterministic_scanner'
>>;

export const useSecurityPolicy = () => {
  const [policy, setPolicy] = useState<SecurityPolicy | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('/api/v1/security/policy');
      if (!response.ok) throw new Error(`Security policy request failed: ${response.status}`);
      setPolicy(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Failed to load security policy');
    } finally {
      setLoading(false);
    }
  }, []);

  const updatePolicy = useCallback(async (updates: SecurityPolicyUpdate) => {
    const response = await fetch('/api/v1/security/policy', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!response.ok) throw new Error(`Security policy update failed: ${response.status}`);
    setPolicy(await response.json());
  }, []);

  useEffect(() => { void refetch(); }, [refetch]);
  return { policy, loading, error, refetch, updatePolicy };
};
