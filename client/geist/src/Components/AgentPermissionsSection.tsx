import React, { useState, useEffect } from 'react';
import SettingsSelect from './SettingsSelect';
import { AgentPermissions, AgentPermissionMode } from '../Hooks/useUserSettings';

interface AgentPermissionsSectionProps {
  agentPermissions: AgentPermissions | null | undefined;
  onChange: (value: AgentPermissions) => void;
}

interface ChatTool {
  name: string;
  description: string;
  requires_approval: boolean;
  requires_per_call_approval?: boolean;
  side_effect: string;
  enabled?: boolean;
  allows_standing_grant?: boolean;
}

const DEFAULT_PERMISSIONS: AgentPermissions = { mode: 'default', always_allow: [] };

const modeOptions = [
  { value: 'default', label: 'Balanced (side-effecting tools ask)' },
  { value: 'require_approval', label: 'Require approval for every tool' },
  { value: 'auto_approve', label: 'Auto-approve all tools' }
];

const modeDescriptions: Record<AgentPermissionMode, string> = {
  default:
    'Read-only tools run automatically; tools that send messages or write files wait for your approval.',
  require_approval:
    'Every agent tool call waits for approval unless it has an eligible always-allow grant. Runtime-discovered tools and tools requiring fresh approval cannot use standing grants.',
  auto_approve:
    'The agent runs tools without asking, except tools requiring fresh approval. This includes runtime-discovered MCP/plugin tools whose definitions can change without notice. Only enable this if you trust all connected tools.'
};

const AgentPermissionsSection: React.FC<AgentPermissionsSectionProps> = ({
  agentPermissions,
  onChange
}) => {
  const permissions = agentPermissions ?? DEFAULT_PERMISSIONS;
  const [tools, setTools] = useState<ChatTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const unavailableGrants = loading || loadError ? [] : permissions.always_allow.filter(name => !tools.some(tool => tool.name === name));

  useEffect(() => {
    void fetchTools();
  }, []);

  const fetchTools = async () => {
    try {
      const response = await fetch('/agent/tools');
      if (!response.ok) {
        throw new Error('Tool catalog request failed');
      }
      const data = await response.json();
      if (!Array.isArray(data.tools) || data.tools.some((tool: unknown) =>
        !tool || typeof tool !== 'object' || !('name' in tool) || typeof tool.name !== 'string'
      )) {
        throw new Error('Invalid tool catalog response');
      }
      setTools(data.tools);
    } catch (err) {
      setLoadError(true);
      console.error('Failed to fetch agent tools:', err);
    } finally {
      setLoading(false);
    }
  };

  const setMode = (mode: string) => {
    onChange({ ...permissions, mode: mode as AgentPermissionMode });
  };

  const toggleAlwaysAllow = (toolName: string) => {
    const alwaysAllow = permissions.always_allow.includes(toolName)
      ? permissions.always_allow.filter((name) => name !== toolName)
      : [...permissions.always_allow, toolName];
    onChange({ ...permissions, always_allow: alwaysAllow });
  };

  const clearAll = () => {
    onChange({ ...permissions, always_allow: [] });
  };

  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>Agent Permissions</h3>
        <p>Control when agent tool calls need your approval before they run.</p>
      </header>

      <SettingsSelect
        label="Approval Mode"
        value={permissions.mode}
        options={modeOptions}
        onChange={setMode}
        description={modeDescriptions[permissions.mode] ?? modeDescriptions.default}
      />

      <div className="settings-subsection">
        <div className="settings-subsection-header">
          <div>
            <span className="settings-label">Always-Allowed Tools</span>
            <p className="settings-description">
              Eligible built-in tools on this list skip approval. Runtime-discovered tools and tools requiring fresh approval cannot receive standing grants (
              {permissions.always_allow.length} selected).
            </p>
          </div>
          <div className="settings-inline-actions">
            <button
              className="button button-secondary button-small"
              onClick={clearAll}
              disabled={loading || permissions.always_allow.length === 0}
            >
              Clear All
            </button>
          </div>
        </div>

        {permissions.mode === 'auto_approve' && (
          <p className="settings-description">
            Auto-approve is on, except for tools requiring fresh approval; this list applies when
            you switch back to a stricter mode.
          </p>
        )}

        {!loading && unavailableGrants.length > 0 && (
          <div>
            <p className="settings-description">Unavailable saved grants — these names are not in the current tool catalog.</p>
            {unavailableGrants.map(name => (
              <button key={name} type="button" onClick={() => toggleAlwaysAllow(name)}>
                Remove unavailable grant: {name}
              </button>
            ))}
          </div>
        )}
        {loading ? (
          <div className="empty-state compact">Loading tools...</div>
        ) : loadError ? (
          <div role="alert" className="empty-state compact">
            Could not load the tool catalog. Your saved grants are unchanged. Switch tabs and return to try again.
          </div>
        ) : tools.length === 0 ? (
          <div className="empty-state compact">No agent tools are available.</div>
        ) : (
          <div className="settings-file-list">
            {tools.map((tool) => {
              const selected = permissions.always_allow.includes(tool.name);
              return (
                <button
                  key={tool.name}
                  type="button"
                  className={`settings-file-option ${selected ? 'selected' : ''}`}
                  onClick={() => toggleAlwaysAllow(tool.name)}
                  aria-pressed={selected}
                  disabled={(tool.enabled === false || tool.requires_per_call_approval === true || tool.allows_standing_grant === false) && !selected}
                  title={tool.description}
                >
                  <span className="settings-checkbox" aria-hidden="true">
                  </span>
                  <span>
                    {tool.name}
                    {tool.enabled === false && <span className="settings-description"> — unavailable</span>}
                    {tool.allows_standing_grant === false && !tool.requires_per_call_approval && <span className="settings-description"> — runtime-discovered; {selected ? 'stored grant is not honored; click to revoke' : 'standing grants unavailable'}</span>}
                    {(tool.side_effect === 'external_write' || tool.side_effect === 'process') && (
                      <span className="settings-description"> — can affect external systems or run commands</span>
                    )}
                    {tool.requires_per_call_approval ? (
                      <span className="settings-description"> — fresh approval required; always-allow does not apply</span>
                    ) : tool.requires_approval && (
                      <span className="settings-description"> — asks by default</span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
};

export default AgentPermissionsSection;
