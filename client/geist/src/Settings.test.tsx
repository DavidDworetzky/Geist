import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Settings from './Settings';

const baseSettings = {
  user_settings_id: 1,
  user_id: 1,
  default_agent_type: 'local',
  default_local_model: 'Meta-Llama-3.1-8B-Instruct',
  default_online_model: 'gpt-4',
  default_online_provider: 'openai',
  default_file_archives: [],
  enable_rag_by_default: false,
  default_max_tokens: 256,
  default_temperature: 0.7,
  default_top_p: 0.9,
  default_frequency_penalty: 0,
  default_presence_penalty: 0,
  backup_providers: [],
  ui_preferences: {},
  create_date: '2025-01-01T00:00:00Z',
  update_date: '2025-01-01T00:00:00Z'
};

describe('Settings page', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('shows loading, then renders tabs', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => baseSettings });

    render(<Settings />);
    expect(screen.getByText(/Loading settings/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Settings')).toBeInTheDocument();
      expect(screen.getByText('Agent Config')).toBeInTheDocument();
      expect(screen.getByText('Generation')).toBeInTheDocument();
      expect(screen.getByText('RAG & Files')).toBeInTheDocument();
      expect(screen.getByText('UI Preferences')).toBeInTheDocument();
    });
  });

  it('marks unsaved changes when local values change and saves', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: true, json: async () => baseSettings }) // initial GET
      .mockResolvedValueOnce({ ok: true, json: async () => ({ ...baseSettings, default_temperature: 0.8 }) }); // PUT

    render(<Settings />);

    // Wait for the Generation tab to be visible (indicating loading is complete)
    await waitFor(() => {
      expect(screen.getByText('Generation')).toBeInTheDocument();
    });

    // go to Generation tab
    fireEvent.click(screen.getByText('Generation'));

    // change slider value
    const slider = screen.getByRole('slider', { name: /Temperature/i }) as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '0.8' } });

    expect(screen.getByText(/Unsaved Changes/i)).toBeInTheDocument();

    // save
    fireEvent.click(screen.getByText(/Save Changes/i));

    await waitFor(() => {
      expect(screen.getByText(/Settings saved successfully/i)).toBeInTheDocument();
    });
  });

  it('cancel reverts local changes', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => baseSettings });

    render(<Settings />);

    // Wait for the Generation tab to be visible (indicating loading is complete)
    await waitFor(() => {
      expect(screen.getByText('Generation')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Generation'));
    const slider = screen.getByRole('slider', { name: /Temperature/i }) as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '1.1' } });
    expect(screen.getByText(/Unsaved Changes/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText('Cancel'));
    expect(screen.queryByText(/Unsaved Changes/i)).not.toBeInTheDocument();
  });

  it('reset triggers API and shows success', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: true, json: async () => baseSettings }) // initial GET
      .mockResolvedValueOnce({ ok: true, json: async () => baseSettings }); // POST reset

    // confirm window
    jest.spyOn(window, 'confirm').mockReturnValue(true);

    render(<Settings />);

    // Wait for the Reset button to be visible (indicating loading is complete)
    await waitFor(() => {
      expect(screen.getByText(/Reset to Defaults/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/Reset to Defaults/i));
    await waitFor(() => {
      expect(screen.getByText(/Settings reset to defaults successfully/i)).toBeInTheDocument();
    });
  });

  it('shows error with Retry on initial fetch failure', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: false, statusText: 'Server Error' })
      .mockResolvedValueOnce({ ok: true, json: async () => baseSettings });

    render(<Settings />);
    await waitFor(() => screen.getByText(/Error loading settings/i));

    fireEvent.click(screen.getByText('Retry'));
    await waitFor(() => screen.getByText('Settings'));
  });
});
