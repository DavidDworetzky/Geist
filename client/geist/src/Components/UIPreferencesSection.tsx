import React from 'react';
import SettingsToggle from './SettingsToggle';
import SettingsSelect from './SettingsSelect';

interface UIPreferencesSectionProps {
  uiPreferences: Record<string, any>;
  onUiPreferencesChange: (preferences: Record<string, any>) => void;
}

const UIPreferencesSection: React.FC<UIPreferencesSectionProps> = ({
  uiPreferences,
  onUiPreferencesChange
}) => {
  const updatePreference = (key: string, value: any) => {
    onUiPreferencesChange({
      ...uiPreferences,
      [key]: value
    });
  };

  const fontSizeOptions = [
    { value: 'small', label: 'Small' },
    { value: 'medium', label: 'Medium' },
    { value: 'large', label: 'Large' }
  ];

  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>UI Preferences</h3>
        <p>Adjust display density and supporting interface behaviors.</p>
      </header>

      <SettingsSelect
        label="Font Size"
        value={uiPreferences.fontSize || 'medium'}
        options={fontSizeOptions}
        onChange={(value) => updatePreference('fontSize', value)}
        description="Adjust the default font size for text."
      />

      <SettingsToggle
        label="Show File Previews"
        checked={uiPreferences.showFilePreviews !== false}
        onChange={(value) => updatePreference('showFilePreviews', value)}
        description="Display file content previews in the Files page."
      />

      <SettingsToggle
        label="Enable Notifications"
        checked={uiPreferences.enableNotifications !== false}
        onChange={(value) => updatePreference('enableNotifications', value)}
        description="Show desktop notifications for important events."
      />

      <SettingsToggle
        label="Compact Mode"
        checked={uiPreferences.compactMode || false}
        onChange={(value) => updatePreference('compactMode', value)}
        description="Use a more compact layout to show more content."
      />

      <SettingsToggle
        label="Show Workflow Thumbnails"
        checked={uiPreferences.showWorkflowThumbnails !== false}
        onChange={(value) => updatePreference('showWorkflowThumbnails', value)}
        description="Display thumbnail previews for workflows."
      />
    </section>
  );
};

export default UIPreferencesSection;
