import React, { useState } from 'react';
import { McpServer } from '../Hooks/useMcpServers';

interface Props {
  server?: McpServer;
  onSaved: () => Promise<void>;
  onCancel: () => void;
}

const ProtonBridgeSetup: React.FC<Props> = ({ server, onSaved, onCancel }) => {
  const saved = server?.managed_email;
  const [name, setName] = useState(server?.name ?? 'proton-mail');
  const [host, setHost] = useState(saved?.host ?? '127.0.0.1');
  const [username, setUsername] = useState(saved?.username ?? '');
  const [sender, setSender] = useState(saved?.sender ?? '');
  const [password, setPassword] = useState('');
  const [certificate, setCertificate] = useState(saved?.certificate ?? '');
  const [imapPort, setImapPort] = useState(saved?.imap_port ?? 1143);
  const [smtpPort, setSmtpPort] = useState(saved?.smtp_port ?? 1025);
  const [imapSecurity, setImapSecurity] = useState(saved?.imap_security ?? 'starttls');
  const [smtpSecurity, setSmtpSecurity] = useState(saved?.smtp_security ?? 'starttls');
  const [allowlist, setAllowlist] = useState((server?.recipient_allowlist ?? []).join('\n'));
  const [limit, setLimit] = useState(server?.max_writes_per_hour ?? 20);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const submit = async (testOnly: boolean) => {
    setBusy(true); setStatus(''); setError('');
    try {
      const path = testOnly ? '/test' : server ? `/${server.mcp_server_id}` : '';
      const response = await fetch(`/api/v1/mcp/proton-bridge${path}`, {
        method: !testOnly && server ? 'PUT' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, host, username, sender, password: password || null, certificate,
          imap_port: imapPort, smtp_port: smtpPort, imap_security: imapSecurity, smtp_security: smtpSecurity,
          recipient_allowlist: allowlist.split('\n').map(x => x.trim()).filter(Boolean), max_writes_per_hour: limit }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Check the connection fields and try again.');
      if (testOnly) setStatus('IMAP and SMTP authenticated successfully. No email was sent.');
      else { setPassword(''); await onSaved(); }
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Connection failed'); }
    finally { setBusy(false); }
  };

  return <form className="mcp-server-form" aria-label="Proton Bridge connection" onSubmit={e => { e.preventDefault(); void submit(false); }}>
    <h3>{server ? 'Edit Proton Bridge connection' : 'Connect Proton Mail'}</h3>
    <p><a href="https://proton.me/mail/bridge" target="_blank" rel="noreferrer">Install Proton Mail Bridge</a>, sign in there, and keep it running on the same computer as Geist. A paid Proton Mail plan is required.</p>
    <p>Paste the connection details shown in Bridge below. Geist includes the email connector; no separate MCP installation or command is needed.</p>
    <p>Use the Bridge password, which is separate from your Proton account password. It is stored in macOS Keychain or Linux Secret Service, not in Geist’s database.</p>
    <p>In Bridge’s advanced settings, export the TLS certificate. Paste the contents of <strong>cert.pem</strong> below, never the private key. Match each TLS mode to Bridge’s settings.</p>
    <p>Run Geist natively alongside Bridge. A Docker container’s localhost cannot reach Bridge on your desktop.</p>
    <fieldset disabled={busy} style={{ border: 0, padding: 0 }}>
      <label className="settings-field">Name<input className="form-control" required pattern="[A-Za-z0-9_-]{1,64}" value={name} onChange={e => setName(e.target.value)} /></label>
      <label className="settings-field">Bridge host<select className="form-control" value={host} onChange={e => setHost(e.target.value)}><option>127.0.0.1</option><option>::1</option></select></label>
      <label className="settings-field">Bridge username<input className="form-control" required autoComplete="off" value={username} onChange={e => setUsername(e.target.value)} /></label>
      <label className="settings-field">Sender email address<input className="form-control" type="email" required value={sender} onChange={e => setSender(e.target.value)} /></label>
      <label className="settings-field">Bridge password<input className="form-control" type="password" required={!saved} autoComplete="new-password" value={password} onChange={e => setPassword(e.target.value)} /></label>
      {saved && <p>Leave the password blank to keep the saved credential. Saving tests the connection again and disables it until you re-enable it.</p>}
      <label className="settings-field">IMAP port<input className="form-control" type="number" required min={1} max={65535} value={imapPort} onChange={e => setImapPort(Number(e.target.value))} /></label>
      <label className="settings-field">IMAP encryption<select className="form-control" value={imapSecurity} onChange={e => setImapSecurity(e.target.value)}><option value="starttls">STARTTLS</option><option value="tls">SSL/TLS</option></select></label>
      <label className="settings-field">SMTP port<input className="form-control" type="number" required min={1} max={65535} value={smtpPort} onChange={e => setSmtpPort(Number(e.target.value))} /></label>
      <label className="settings-field">SMTP encryption<select className="form-control" value={smtpSecurity} onChange={e => setSmtpSecurity(e.target.value)}><option value="starttls">STARTTLS</option><option value="tls">SSL/TLS</option></select></label>
      <label className="settings-field">Bridge public TLS certificate<textarea className="form-control" required rows={5} value={certificate} onChange={e => setCertificate(e.target.value)} /></label>
      <label className="settings-field">Allowed recipients (one per line)<textarea className="form-control" rows={3} value={allowlist} onChange={e => setAllowlist(e.target.value)} /></label>
      <label className="settings-field">Sends per hour<input className="form-control" type="number" required min={1} max={1000} value={limit} onChange={e => setLimit(Number(e.target.value))} /></label>
      <p>The connection starts disabled. Enable it after saving; email tools use Geist’s inspection and approval controls. An online model can receive inspected email content; select a local model to keep inference local.</p>
      <div className="mcp-server-actions">
        <button className="button" type="button" disabled={!!saved && !password} onClick={() => void submit(true)}>Test connection</button>
        <button className="button" type="submit">{busy ? 'Connecting…' : 'Test and save connection'}</button>
        <button className="button button-small" type="button" onClick={onCancel}>Cancel</button>
      </div>
    </fieldset>
    {status && <p role="status">{status}</p>}
    {error && <p role="alert" className="notice notice-error">{error}</p>}
  </form>;
};

export default ProtonBridgeSetup;
