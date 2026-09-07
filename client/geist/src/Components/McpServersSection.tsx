import React, { useEffect, useRef, useState } from 'react';
import SettingsSelect from './SettingsSelect';
import SettingsToggle from './SettingsToggle';
import { McpCatalogueEntry, mcpCatalogue } from '../mcpCatalogue';
import '../Tools.css';
import {
  McpServer,
  McpServerInput,
  McpTestResult,
  useMcpServers,
} from '../Hooks/useMcpServers';

const transportOptions = [
  { value: 'stdio', label: 'Local process (stdio)' },
  { value: 'http', label: 'Remote server (HTTP)' },
];

interface FormState {
  name: string;
  transport: 'stdio' | 'http';
  command: string;
  args: string;
  env: string;
  url: string;
  headers: string;
  timeout_seconds: string;
}

const emptyForm: FormState = {
  name: '',
  transport: 'stdio',
  command: '',
  args: '',
  env: '',
  url: '',
  headers: '',
  timeout_seconds: '30',
};

const linesToList = (value: string): string[] =>
  value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

const linesToRecord = (value: string): Record<string, string> => {
  const record: Record<string, string> = {};
  for (const line of linesToList(value)) {
    const separator = line.indexOf('=');
    if (separator > 0) {
      record[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
    }
  }
  return record;
};

const recordToLines = (record: Record<string, string>): string =>
  Object.entries(record)
    .map(([key, value]) => `${key}=${value}`)
    .join('\n');

const formFromServer = (server: McpServer): FormState => ({
  name: server.name,
  transport: server.transport,
  command: server.command ?? '',
  args: (server.args || []).join('\n'),
  env: recordToLines(server.env || {}),
  url: server.url ?? '',
  headers: recordToLines(server.headers || {}),
  timeout_seconds: String(server.timeout_seconds),
});

const formFromCatalogue = (entry: McpCatalogueEntry): FormState => ({
  name: entry.id,
  transport: entry.transport,
  command: entry.command ?? '',
  args: (entry.args ?? []).join('\n'),
  env: '',
  url: entry.url ?? '',
  headers: '',
  timeout_seconds: '30',
});

const inputFromForm = (form: FormState): McpServerInput => ({
  name: form.name.trim(),
  transport: form.transport,
  command: form.transport === 'stdio' ? form.command.trim() : null,
  args: form.transport === 'stdio' ? linesToList(form.args) : [],
  env: form.transport === 'stdio' ? linesToRecord(form.env) : {},
  url: form.transport === 'http' ? form.url.trim() : null,
  headers: form.transport === 'http' ? linesToRecord(form.headers) : {},
  timeout_seconds: Number(form.timeout_seconds) || 30,
});

const McpServersSection: React.FC = () => {
  const {
    servers,
    loading,
    error,
    createServer,
    updateServer,
    deleteServer,
    testServer,
  } = useMcpServers();

  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [selectedCatalogueId, setSelectedCatalogueId] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(emptyForm);
  const [formError, setFormError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [testingId, setTestingId] = useState<number | null>(null);
  const [testResults, setTestResults] = useState<Record<number, McpTestResult>>({});
  const formRef = useRef<HTMLDivElement>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!formOpen) return;
    formRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
    nameInputRef.current?.focus({ preventScroll: true });
  }, [editingId, formOpen, selectedCatalogueId]);

  const updateForm = (key: keyof FormState, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const openCreateForm = () => {
    setForm(emptyForm);
    setEditingId(null);
    setSelectedCatalogueId(null);
    setFormError(null);
    setFormOpen(true);
  };

  const openCatalogueForm = (entry: McpCatalogueEntry, server?: McpServer) => {
    setForm(server ? formFromServer(server) : formFromCatalogue(entry));
    setEditingId(server?.mcp_server_id ?? null);
    setSelectedCatalogueId(entry.id);
    setFormError(null);
    setFormOpen(true);
  };

  const openEditForm = (server: McpServer) => {
    setForm(formFromServer(server));
    setEditingId(server.mcp_server_id);
    setSelectedCatalogueId(null);
    setFormError(null);
    setFormOpen(true);
  };

  const closeForm = () => {
    setFormOpen(false);
    setEditingId(null);
    setSelectedCatalogueId(null);
    setForm(emptyForm);
    setFormError(null);
  };

  const handleSubmit = async () => {
    try {
      setSaving(true);
      setFormError(null);
      const input = inputFromForm(form);
      if (!input.name) {
        throw new Error('Name is required');
      }
      if (input.transport === 'stdio' && !input.command) {
        throw new Error('Local servers require a command');
      }
      if (input.transport === 'http' && !input.url) {
        throw new Error('Remote servers require a URL');
      }
      if (editingId !== null) {
        await updateServer(editingId, input);
      } else {
        await createServer(input);
      }
      closeForm();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to save MCP server');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (server: McpServer) => {
    if (!window.confirm(`Delete MCP server "${server.name}"? Its tools will disappear from chat.`)) {
      return;
    }
    try {
      await deleteServer(server.mcp_server_id);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to delete MCP server');
    }
  };

  const handleToggleEnabled = async (server: McpServer, enabled: boolean) => {
    try {
      await updateServer(server.mcp_server_id, { enabled });
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to update MCP server');
    }
  };

  const handleTest = async (server: McpServer) => {
    try {
      setTestingId(server.mcp_server_id);
      const result = await testServer(server.mcp_server_id);
      setTestResults((prev) => ({ ...prev, [server.mcp_server_id]: result }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [server.mcp_server_id]: {
          ok: false,
          error: err instanceof Error ? err.message : 'Connection test failed',
          tools: [],
        },
      }));
    } finally {
      setTestingId(null);
    }
  };

  const selectedCatalogueEntry = mcpCatalogue.find(
    (entry) => entry.id === selectedCatalogueId
  );

  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>MCP Servers</h3>
        <p>
          Connect Model Context Protocol servers to add their tools to chat. Tools from
          enabled servers appear to the model as mcp.&lt;server&gt;.&lt;tool&gt;.
        </p>
      </header>

      <div className="mcp-catalogue-block">
        <div className="mcp-catalogue-heading">
          <div>
            <h4>Starter catalogue</h4>
            <p className="settings-description">
              Prefill a reviewed server profile, then inspect its endpoint, permissions, and
              credentials before enabling it.
            </p>
          </div>
          <span className="mcp-catalogue-count">{mcpCatalogue.length} profiles</span>
        </div>
        <div className="mcp-catalogue-grid" data-testid="mcp-catalogue">
          {mcpCatalogue.map((entry) => {
            const configuredServer = servers.find((server) => server.name === entry.id);
            return (
              <article className="mcp-catalogue-card" key={entry.id}>
                <div className="mcp-catalogue-card-heading">
                  <div>
                    <span className="mcp-catalogue-category">{entry.category}</span>
                    <h4>{entry.name}</h4>
                  </div>
                  <span className="mcp-catalogue-publisher">{entry.publisher}</span>
                </div>
                <p>{entry.description}</p>
                {entry.accountScope && (
                  <p className="mcp-catalogue-scope">{entry.accountScope}</p>
                )}
                <div className="mcp-catalogue-actions">
                  <a href={entry.sourceUrl} target="_blank" rel="noreferrer">
                    Source
                  </a>
                  {configuredServer && (
                    <span className="mcp-catalogue-configured">Configured</span>
                  )}
                  <button
                    className="button button-small"
                    type="button"
                    aria-label={`${configuredServer ? 'Edit' : 'Configure'} ${entry.name}`}
                    onClick={() => openCatalogueForm(entry, configuredServer)}
                  >
                    {configuredServer ? 'Edit' : 'Configure'}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </div>

      {error && <div className="notice notice-error">{error}</div>}
      {formError && !formOpen && <div className="notice notice-error">{formError}</div>}

      {loading ? (
        <div className="empty-state">Loading MCP servers...</div>
      ) : servers.length === 0 && !formOpen ? (
        <div className="empty-state">No MCP servers configured yet.</div>
      ) : (
        <ul className="mcp-server-list" data-testid="mcp-server-list">
          {servers.map((server) => {
            const testResult = testResults[server.mcp_server_id];
            return (
              <li key={server.mcp_server_id} className="mcp-server-item settings-field">
                <div className="mcp-server-row">
                  <div className="mcp-server-copy">
                    <span className="settings-label">{server.name}</span>
                    <p className="settings-description">
                      {server.transport === 'stdio'
                        ? `${server.command ?? ''} ${(server.args || []).join(' ')}`.trim()
                        : server.url}
                    </p>
                  </div>
                  <div className="mcp-server-actions">
                    <button
                      className="button button-small"
                      type="button"
                      disabled={testingId === server.mcp_server_id}
                      onClick={() => void handleTest(server)}
                    >
                      {testingId === server.mcp_server_id ? 'Testing...' : 'Test'}
                    </button>
                    <button
                      className="button button-small"
                      type="button"
                      onClick={() => openEditForm(server)}
                    >
                      Edit
                    </button>
                    <button
                      className="button button-small"
                      type="button"
                      onClick={() => void handleDelete(server)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
                <SettingsToggle
                  label="Enabled"
                  checked={server.enabled}
                  onChange={(checked) => void handleToggleEnabled(server, checked)}
                  description="Only enabled servers expose tools to chat."
                />
                {testResult && (
                  <div
                    className={`notice ${testResult.ok ? 'notice-success' : 'notice-error'}`}
                    data-testid={`mcp-test-result-${server.mcp_server_id}`}
                  >
                    {testResult.ok ? (
                      <span>
                        Connected. {testResult.tools.length} tool
                        {testResult.tools.length === 1 ? '' : 's'} available
                        {testResult.tools.length > 0 && (
                          <>: {testResult.tools.map((tool) => tool.name).join(', ')}</>
                        )}
                      </span>
                    ) : (
                      <span>Connection failed: {testResult.error}</span>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {formOpen ? (
        <div className="mcp-server-form" data-testid="mcp-server-form" ref={formRef}>
          <header className="settings-section-header">
            <h3>{editingId !== null ? 'Edit MCP Server' : 'Add MCP Server'}</h3>
          </header>

          {selectedCatalogueEntry && (
            <div className="mcp-profile-summary" data-testid="mcp-profile-summary">
              <span className="mcp-catalogue-category">Catalogue profile</span>
              <h4>{selectedCatalogueEntry.name}</h4>
              {selectedCatalogueEntry.accountScope && (
                <p><strong>Account:</strong> {selectedCatalogueEntry.accountScope}</p>
              )}
              {selectedCatalogueEntry.authentication && (
                <p><strong>Authentication:</strong> {selectedCatalogueEntry.authentication}</p>
              )}
              {selectedCatalogueEntry.requirements.length > 0 && (
                <ul>
                  {selectedCatalogueEntry.requirements.map((requirement) => (
                    <li key={requirement}>{requirement}</li>
                  ))}
                </ul>
              )}
              {selectedCatalogueEntry.configurationNote && (
                <p className="mcp-profile-warning">{selectedCatalogueEntry.configurationNote}</p>
              )}
            </div>
          )}

          {formError && <div className="notice notice-error">{formError}</div>}

          <div className="settings-field">
            <label className="settings-label" htmlFor="mcp-name">Name</label>
            <p className="settings-description">
              Letters, numbers, dashes, and underscores. Used to namespace the server's tools.
            </p>
            <input
              id="mcp-name"
              ref={nameInputRef}
              className="form-control"
              type="text"
              value={form.name}
              onChange={(e) => updateForm('name', e.target.value)}
            />
          </div>

          <SettingsSelect
            label="Transport"
            value={form.transport}
            options={transportOptions}
            onChange={(value) => updateForm('transport', value)}
            description="Local servers are spawned as a process; remote servers speak streamable HTTP."
          />

          {form.transport === 'stdio' ? (
            <>
              <div className="settings-field">
                <label className="settings-label" htmlFor="mcp-command">Command</label>
                <input
                  id="mcp-command"
                  className="form-control"
                  type="text"
                  placeholder="npx"
                  value={form.command}
                  onChange={(e) => updateForm('command', e.target.value)}
                />
              </div>
              <div className="settings-field">
                <label className="settings-label" htmlFor="mcp-args">Arguments (one per line)</label>
                <textarea
                  id="mcp-args"
                  className="form-control"
                  rows={3}
                  placeholder={'-y\n@modelcontextprotocol/server-filesystem'}
                  value={form.args}
                  onChange={(e) => updateForm('args', e.target.value)}
                />
              </div>
              <div className="settings-field">
                <label className="settings-label" htmlFor="mcp-env">
                  Environment variables (KEY=value, one per line)
                </label>
                <textarea
                  id="mcp-env"
                  className="form-control"
                  rows={2}
                  value={form.env}
                  onChange={(e) => updateForm('env', e.target.value)}
                />
              </div>
            </>
          ) : (
            <>
              <div className="settings-field">
                <label className="settings-label" htmlFor="mcp-url">Server URL</label>
                <input
                  id="mcp-url"
                  className="form-control"
                  type="text"
                  placeholder="https://example.com/mcp"
                  value={form.url}
                  onChange={(e) => updateForm('url', e.target.value)}
                />
              </div>
              <div className="settings-field">
                <label className="settings-label" htmlFor="mcp-headers">
                  Headers (KEY=value, one per line)
                </label>
                <textarea
                  id="mcp-headers"
                  className="form-control"
                  rows={2}
                  placeholder="Authorization=Bearer my-token"
                  value={form.headers}
                  onChange={(e) => updateForm('headers', e.target.value)}
                />
              </div>
            </>
          )}

          <div className="settings-field">
            <label className="settings-label" htmlFor="mcp-timeout">Timeout (seconds)</label>
            <input
              id="mcp-timeout"
              className="form-control"
              type="number"
              min={1}
              max={600}
              value={form.timeout_seconds}
              onChange={(e) => updateForm('timeout_seconds', e.target.value)}
            />
          </div>

          <div className="mcp-server-actions">
            <button className="button" type="button" disabled={saving} onClick={() => void handleSubmit()}>
              {saving ? 'Saving...' : editingId !== null ? 'Save Server' : 'Add Server'}
            </button>
            <button className="button button-small" type="button" onClick={closeForm}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="mcp-server-actions">
          <button className="button" type="button" onClick={openCreateForm}>
            Add MCP Server
          </button>
        </div>
      )}
    </section>
  );
};

export default McpServersSection;
