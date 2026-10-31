import React from 'react';

interface SettingsToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  description?: string;
  disabled?: boolean;
}

const SettingsToggle: React.FC<SettingsToggleProps> = ({
  label,
  checked,
  onChange,
  description,
  disabled = false
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
        aria-label={label}
        aria-pressed={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
      >
        <span className="settings-toggle-thumb" />
      </button>
    </div>
  );
};

export default SettingsToggle;
