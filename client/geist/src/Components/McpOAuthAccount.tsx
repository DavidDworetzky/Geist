import React, { useEffect, useState } from 'react';

interface OAuthProvider {
  id: string;
  label: string;
  scopes: string[];
  ready: boolean;
  setup_message: string | null;
}

interface OAuthStatus {
  provider: string | null;
  status: 'disconnected' | 'pending' | 'connected' | 'reconnect_required';
  providers: OAuthProvider[];
  redirect_uri: string | null;
}

export async function oauthRequest(path: string, options?: RequestInit): Promise<any> {
  const response = await fetch(`/api/v1/mcp/${path}`, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Account connection failed');
  return body;
}

export async function connectOAuth(serverId: number, provider: string): Promise<void> {
  const result = await oauthRequest(`servers/${serverId}/oauth/start`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  });
  window.location.assign(result.authorization_url);
}

const McpOAuthAccount: React.FC<{ serverId: number; onChange: () => Promise<void> }> = ({ serverId, onChange }) => {
  const [status, setStatus] = useState<OAuthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    oauthRequest(`servers/${serverId}/oauth`).then((value) => {
      if (!cancelled) setStatus(value);
    }).catch((err) => { if (!cancelled) setError(err.message); });
    return () => { cancelled = true; };
  }, [serverId]);

  const connect = async (provider: string) => {
    setBusy(true);
    setError(null);
    try { await connectOAuth(serverId, provider); }
    catch (err) { setError(err instanceof Error ? err.message : 'Could not connect account'); }
    finally { setBusy(false); }
  };

  const disconnect = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await oauthRequest(`servers/${serverId}/oauth`, { method: 'DELETE' });
      setNotice(result.revocation_complete
        ? 'Account disconnected and provider access revoked. Tools are disabled.'
        : 'Account disconnected and tools disabled. You can also remove the app grant in your provider account settings.');
      setStatus(await oauthRequest(`servers/${serverId}/oauth`));
      await onChange();
    } catch (err) { setError(err instanceof Error ? err.message : 'Could not disconnect account'); }
    finally { setBusy(false); }
  };

  if (!status?.providers?.length && !status?.provider && !error) return null;
  return <div className="mcp-oauth-account">
    <h4>Account connection</h4>
    {error && <p role="alert" className="notice notice-error">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    {status?.status === 'connected' && <p role="status">Account connected. Test the server before enabling its tools.</p>}
    {status?.status === 'pending' && <p role="status">Sign-in pending. You can restart sign-in if the previous attempt was interrupted.</p>}
    {status?.status === 'reconnect_required' && <p role="status">Reconnect your account to restore access.</p>}
    {status?.providers?.map((provider) => <div key={provider.id}>
      {provider.setup_message && <p>{provider.setup_message}</p>}
      {!provider.ready && status.redirect_uri && <p>Register this callback URL: <code>{status.redirect_uri}</code></p>}
      <details><summary>Requested permissions</summary><ul>{provider.scopes.map((scope) => <li key={scope}>{scope}</li>)}</ul></details>
      <button type="button" className="button button-small" disabled={busy || !provider.ready}
        onClick={() => void connect(provider.id)}>
        {status.provider ? `Reconnect with ${provider.label}` : `Connect with ${provider.label}`}
      </button>
    </div>)}
    {status?.provider && <button type="button" className="button button-small" disabled={busy} onClick={() => void disconnect()}>
      Disconnect account
    </button>}
  </div>;
};

export default McpOAuthAccount;
