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

  const themeOptions = [
    { value: 'light', label: 'Light' },
    { value: 'dark', label: 'Dark' },
    { value: 'auto', label: 'Auto (System)' }
  ];

  const fontSizeOptions = [
    { value: 'small', label: 'Small' },
    { value: 'medium', label: 'Medium' },
    { value: 'large', label: 'Large' }
  ];

  return (
    <div style={{
      backgroundColor: 'white',
      padding: '25px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      marginBottom: '20px'
    }}>
      <h3 style={{ 
        margin: '0 0 20px 0', 
        color: '#333', 
        fontSize: '18px',
        borderBottom: '2px solid #007bff',
        paddingBottom: '10px'
      }}>
        UI Preferences
      </h3>

      <SettingsSelect
        label="Font Size"
        value={uiPreferences.fontSize || 'medium'}
        options={fontSizeOptions}
        onChange={(value) => updatePreference('fontSize', value)}
        description="Adjust the default font size for text"
      />

      <SettingsToggle
        label="Show File Previews"
        checked={uiPreferences.showFilePreviews !== false}
        onChange={(value) => updatePreference('showFilePreviews', value)}
        description="Display file content previews in the Files page"
      />

      <SettingsToggle
        label="Enable Notifications"
        checked={uiPreferences.enableNotifications !== false}
        onChange={(value) => updatePreference('enableNotifications', value)}
        description="Show desktop notifications for important events"
      />

      <SettingsToggle
        label="Compact Mode"
        checked={uiPreferences.compactMode || false}
        onChange={(value) => updatePreference('compactMode', value)}
        description="Use a more compact layout to show more content"
      />

      <SettingsToggle
        label="Show Workflow Thumbnails"
        checked={uiPreferences.showWorkflowThumbnails !== false}
        onChange={(value) => updatePreference('showWorkflowThumbnails', value)}
        description="Display thumbnail previews for workflows"
      />
    </div>
  );
};

export default UIPreferencesSection;

