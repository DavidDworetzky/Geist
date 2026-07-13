import React from 'react';

interface SettingsToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  description?: string;
}

const SettingsToggle: React.FC<SettingsToggleProps> = ({
  label,
  checked,
  onChange,
  description
}) => {
  return (
    <div className="settings-field settings-toggle-field">
      <div className="settings-toggle-copy">
        <span className="settings-label">{label}</span>
        {description && <p className="settings-description">{description}</p>}
      </div>

      <button
        type="button"
        className={`settings-toggle ${checked ? 'settings-toggle-on' : ''}`}
        aria-pressed={checked}
        onClick={() => onChange(!checked)}
      >
        <span className="settings-toggle-thumb" />
      </button>
    </div>
  );
};

export default SettingsToggle;
