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
    <div className="settings-field">
      <div className="settings-field-header">
        <label className="settings-label" htmlFor={sliderId}>
          {label}
        </label>
        <span className="settings-value-pill">{value}</span>
      </div>

      {description && <p className="settings-description">{description}</p>}

      <input
        id={sliderId}
        className="settings-slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />

      <div className="settings-range-labels">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
};

export default SettingsSlider;
