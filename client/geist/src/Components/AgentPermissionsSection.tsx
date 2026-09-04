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
}

const DEFAULT_PERMISSIONS: AgentPermissions = { mode: 'default', always_allow: [] };

const modeOptions = [
  { value: 'default', label: 'Balanced (side-effecting tools ask)' },
  { value: 'require_approval', label: 'Require approval for every tool' },
  { value: 'auto_approve', label: 'Auto-approve eligible tools' }
];

const modeDescriptions: Record<AgentPermissionMode, string> = {
  default:
    'Read-only tools run automatically; tools that send messages or write files wait for your approval.',
  require_approval:
    'Every agent tool call waits for your approval unless the tool is on your always-allow list. Protected terminal configurations always ask.',
  auto_approve:
    'The agent runs tools without asking except protected terminal configurations, which require approval for every command.'
};

const AgentPermissionsSection: React.FC<AgentPermissionsSectionProps> = ({
  agentPermissions,
  onChange
}) => {
  const permissions = agentPermissions ?? DEFAULT_PERMISSIONS;
  const [tools, setTools] = useState<ChatTool[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void fetchTools();
  }, []);

  const fetchTools = async () => {
    try {
      const response = await fetch('/agent/tools');
      if (response.ok) {
        const data = await response.json();
        setTools(data.tools || []);
      }
    } catch (err) {
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
              Tools on this list never wait for approval, in any mode (
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
            Auto-approve is on for eligible tools. Protected terminal configurations still ask
            for every command; this list applies when you switch back to a stricter mode.
          </p>
        )}

        {loading ? (
          <div className="empty-state compact">Loading tools...</div>
        ) : tools.length === 0 ? (
          <div className="empty-state compact">No agent tools are available.</div>
        ) : (
          <div className="settings-file-list">
            {tools.map((tool) => {
              const grantable = !tool.requires_per_call_approval;
              const selected = grantable && permissions.always_allow.includes(tool.name);
              return (
                <button
                  key={tool.name}
                  type="button"
                  className={`settings-file-option ${selected ? 'selected' : ''}`}
                  onClick={() => toggleAlwaysAllow(tool.name)}
                  disabled={!grantable}
                  title={tool.description}
                >
                  <span className="settings-checkbox" aria-hidden="true">
                  </span>
                  <span>
                    {tool.name}
                    {tool.requires_per_call_approval ? (
                      <span className="settings-description"> — always asks</span>
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
