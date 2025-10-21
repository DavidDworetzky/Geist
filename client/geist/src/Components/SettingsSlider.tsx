import React from 'react';

interface SettingsSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  description?: string;
}

const SettingsSlider: React.FC<SettingsSliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  description
}) => {
  const sliderId = `settings-slider-${label.toLowerCase().replace(/\s+/g, '-')}`;

  return (
    <div style={{ marginBottom: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <label htmlFor={sliderId} style={{ fontWeight: '500', color: '#333', fontSize: '14px' }}>
          {label}
        </label>
        <span style={{
          backgroundColor: '#e9ecef',
          padding: '4px 12px',
          borderRadius: '4px',
          fontSize: '14px',
          fontWeight: 'bold',
          color: '#495057'
        }}>
          {value}
        </span>
      </div>

      {description && (
        <p style={{
          fontSize: '12px',
          color: '#6c757d',
          margin: '0 0 8px 0'
        }}>
          {description}
        </p>
      )}

      <input
        id={sliderId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{
          width: '100%',
          height: '6px',
          borderRadius: '3px',
          backgroundColor: '#ddd',
          outline: 'none',
          cursor: 'pointer'
        }}
      />

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
        <span style={{ fontSize: '11px', color: '#6c757d' }}>{min}</span>
        <span style={{ fontSize: '11px', color: '#6c757d' }}>{max}</span>
      </div>
    </div>
  );
};

export default SettingsSlider;

