import React from 'react';

interface SettingsSelectProps {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  description?: string;
}

const SettingsSelect: React.FC<SettingsSelectProps> = ({
  label,
  value,
  options,
  onChange,
  description
}) => {
  const selectId = `settings-select-${label.toLowerCase().replace(/\s+/g, '-')}`;

  return (
    <div className="settings-field">
      <label className="settings-label" htmlFor={selectId}>
        {label}
      </label>

      {description && <p className="settings-description">{description}</p>}

      <select
        id={selectId}
        className="form-control settings-select-control"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default SettingsSelect;
