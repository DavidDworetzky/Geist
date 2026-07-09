import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import UIPreferencesSection from '../UIPreferencesSection';

describe('UIPreferencesSection', () => {
  it('updates font size and toggles via callback', () => {
    const onChange = jest.fn();
    const prefs = { fontSize: 'medium', enableNotifications: true };

    render(
      <UIPreferencesSection
        uiPreferences={prefs}
        onUiPreferencesChange={onChange}
      />
    );

    // change font size select
    const select = screen.getByLabelText(/Font Size/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'large' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ fontSize: 'large' }));

    // toggle notifications
    fireEvent.click(screen.getByText(/Enable Notifications/i).parentElement!.nextSibling!);
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ enableNotifications: false }));
  });
});
