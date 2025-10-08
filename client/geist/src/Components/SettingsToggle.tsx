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
    <div style={{ marginBottom: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ flex: 1 }}>
          <label style={{ fontWeight: '500', color: '#333', fontSize: '14px', display: 'block' }}>
            {label}
          </label>
          {description && (
            <p style={{ fontSize: '12px', color: '#6c757d', margin: '4px 0 0 0' }}>
              {description}
            </p>
          )}
        </div>
        
        <div
          onClick={() => onChange(!checked)}
          style={{
            width: '50px',
            height: '26px',
            borderRadius: '13px',
            backgroundColor: checked ? '#28a745' : '#6c757d',
            position: 'relative',
            cursor: 'pointer',
            transition: 'background-color 0.2s',
            flexShrink: 0,
            marginLeft: '15px'
          }}
        >
          <div
            style={{
              width: '22px',
              height: '22px',
              borderRadius: '50%',
              backgroundColor: 'white',
              position: 'absolute',
              top: '2px',
              left: checked ? '26px' : '2px',
              transition: 'left 0.2s',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default SettingsToggle;

