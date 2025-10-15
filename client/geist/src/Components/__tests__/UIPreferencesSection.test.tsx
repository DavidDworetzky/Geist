import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import UIPreferencesSection from '../UIPreferencesSection';

describe('UIPreferencesSection', () => {
  it('updates theme and toggles via callback', () => {
    const onChange = jest.fn();
    const prefs = { theme: 'light', enableNotifications: true };

    render(
      <UIPreferencesSection
        uiPreferences={prefs}
        onUiPreferencesChange={onChange}
      />
    );

    // change theme select
    const select = screen.getByLabelText(/Theme/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'dark' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ theme: 'dark' }));

    // toggle notifications
    fireEvent.click(screen.getByText(/Enable Notifications/i).parentElement!.nextSibling!);
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ enableNotifications: false }));
  });
});
