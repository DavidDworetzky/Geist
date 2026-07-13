import React from 'react';
import SettingsSlider from './SettingsSlider';

interface GenerationParamsSectionProps {
  temperature: number;
  maxTokens: number;
  topP: number;
  frequencyPenalty: number;
  presencePenalty: number;
  onTemperatureChange: (value: number) => void;
  onMaxTokensChange: (value: number) => void;
  onTopPChange: (value: number) => void;
  onFrequencyPenaltyChange: (value: number) => void;
  onPresencePenaltyChange: (value: number) => void;
}

const GenerationParamsSection: React.FC<GenerationParamsSectionProps> = ({
  temperature,
  maxTokens,
  topP,
  frequencyPenalty,
  presencePenalty,
  onTemperatureChange,
  onMaxTokensChange,
  onTopPChange,
  onFrequencyPenaltyChange,
  onPresencePenaltyChange
}) => {
  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>Generation Parameters</h3>
        <p>Set the default sampling behavior for model responses.</p>
      </header>

      <SettingsSlider
        label="Temperature"
        value={temperature}
        min={0}
        max={2}
        step={0.1}
        onChange={onTemperatureChange}
        description="Controls randomness: lower is more focused, higher is more creative."
      />

      <SettingsSlider
        label="Max Tokens"
        value={maxTokens}
        min={1}
        max={4096}
        step={1}
        onChange={onMaxTokensChange}
        description="Maximum number of tokens to generate in the response."
      />

      <SettingsSlider
        label="Top P"
        value={topP}
        min={0}
        max={1}
        step={0.01}
        onChange={onTopPChange}
        description="Nucleus sampling: consider tokens with top_p probability mass."
      />

      <SettingsSlider
        label="Frequency Penalty"
        value={frequencyPenalty}
        min={0}
        max={2}
        step={0.1}
        onChange={onFrequencyPenaltyChange}
        description="Penalizes repeated tokens based on frequency."
      />

      <SettingsSlider
        label="Presence Penalty"
        value={presencePenalty}
        min={0}
        max={2}
        step={0.1}
        onChange={onPresencePenaltyChange}
        description="Penalizes repeated tokens regardless of frequency."
      />
    </section>
  );
};

export default GenerationParamsSection;
