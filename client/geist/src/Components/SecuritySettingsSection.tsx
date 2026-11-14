import React, { useState } from 'react';
import SettingsToggle from './SettingsToggle';
import { SecurityPolicy, useSecurityPolicy } from '../Hooks/useSecurityPolicy';

type ToggleKey = keyof Pick<
  SecurityPolicy,
  | 'enabled'
  | 'inspect_tool_metadata'
  | 'inspect_outbound_arguments'
  | 'inspect_inbound_results'
  | 'deterministic_scanner'
>;

const SecuritySettingsSection: React.FC = () => {
  const { policy, loading, error, updatePolicy } = useSecurityPolicy();
  const [saving, setSaving] = useState<ToggleKey | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const toggle = async (key: ToggleKey, value: boolean) => {
    try {
      setSaving(key);
      setSaveError(null);
      await updatePolicy({ [key]: value });
    } catch (reason) {
      setSaveError(reason instanceof Error ? reason.message : 'Failed to update security policy');
    } finally {
      setSaving(null);
    }
  };

  if (loading && !policy) return <div className="empty-state">Loading security policy...</div>;
  if (!policy) return <div className="notice notice-error">{error || 'Security policy unavailable'}</div>;

  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>Connector Security</h3>
        <p>
          Inspect untrusted MCP boundaries with a tool-free security agent. Email content remains
          untrusted even when its connector is promoted.
        </p>
      </header>

      {(error || saveError) && <div className="notice notice-error">{saveError || error}</div>}

      <SettingsToggle
        label="Security model"
        checked={policy.enabled}
        disabled={saving !== null}
        onChange={(value) => void toggle('enabled', value)}
        description="Mirror the active local or online chat model. Fail closed on every inspector error."
      />
      <SettingsToggle
        label="Inspect tool definitions"
        checked={policy.inspect_tool_metadata}
        disabled={saving !== null}
        onChange={(value) => void toggle('inspect_tool_metadata', value)}
        description="Inspect discovered names, descriptions, and argument schemas before exposure."
      />
      <SettingsToggle
        label="Inspect outgoing arguments"
        checked={policy.inspect_outbound_arguments}
        disabled={saving !== null}
        onChange={(value) => void toggle('inspect_outbound_arguments', value)}
        description="Inspect every call before the existing approval and execution layers."
      />
      <SettingsToggle
        label="Inspect returned content"
        checked={policy.inspect_inbound_results}
        disabled={saving !== null}
        onChange={(value) => void toggle('inspect_inbound_results', value)}
        description="Inspect results before they enter model context; HTML is reduced to safe text."
      />
      <SettingsToggle
        label="Instruction-pattern tripwires"
        checked={policy.deterministic_scanner}
        disabled={saving !== null}
        onChange={(value) => void toggle('deterministic_scanner', value)}
        description="Use a small deterministic scanner in addition to the security model."
      />

      <div className="settings-readonly-grid">
        <div className="settings-readonly-item">
          <span className="settings-label">Inspector model</span>
          <span className="settings-description">Mirrors the active chat model; tools disabled</span>
        </div>
        <div className="settings-readonly-item">
          <span className="settings-label">Audit log</span>
          <span className="settings-description">security-audit.log (metadata and hashes only)</span>
        </div>
      </div>
    </section>
  );
};

export default SecuritySettingsSection;
