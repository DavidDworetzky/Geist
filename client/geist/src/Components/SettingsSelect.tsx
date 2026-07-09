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
    <div style={{ marginBottom: '20px' }}>
      <label
        htmlFor={selectId}
        style={{
          fontWeight: '500',
          color: '#333',
          fontSize: '14px',
          display: 'block',
          marginBottom: '8px'
        }}
      >
        {label}
      </label>

      {description && (
        <p style={{
          fontSize: '12px',
          color: '#6c757d',
          margin: '0 0 8px 0'
        }}>
          {description}
        </p>
      )}

      <select
        id={selectId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%',
          padding: '10px',
          border: '1px solid #ddd',
          borderRadius: '5px',
          fontSize: '14px',
          backgroundColor: 'white',
          cursor: 'pointer',
          color: '#333'
        }}
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

