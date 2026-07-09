import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SettingsToggle from '../SettingsToggle';
import SettingsSlider from '../SettingsSlider';
import SettingsSelect from '../SettingsSelect';

describe('Settings controls', () => {
  it('SettingsToggle calls onChange with inverted value', () => {
    const onChange = jest.fn();
    render(<SettingsToggle label="Toggle" checked={true} onChange={onChange} />);
    fireEvent.click(screen.getByText('Toggle').parentElement!.nextSibling!);
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it('SettingsSlider calls onChange with numeric value', () => {
    const onChange = jest.fn();
    render(
      <SettingsSlider label="Temp" value={0.7} min={0} max={2} step={0.1} onChange={onChange} />
    );
    const input = screen.getByRole('slider') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '1.2' } });
    expect(onChange).toHaveBeenCalledWith(1.2);
  });

  it('SettingsSelect calls onChange with selected value', () => {
    const onChange = jest.fn();
    render(
      <SettingsSelect
        label="Agent"
        value={'local'}
        options={[{ value: 'local', label: 'Local' }, { value: 'online', label: 'Online' }]}
        onChange={onChange}
      />
    );
    const select = screen.getByLabelText('Agent') as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'online' } });
    expect(onChange).toHaveBeenCalledWith('online');
  });
});
