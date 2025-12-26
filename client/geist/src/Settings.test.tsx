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

const mockModelsResponse = {
  providers: {
    openai: [
      { id: 'gpt-4', name: 'GPT-4', provider: 'openai', recommended: false },
      { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'openai', recommended: true },
    ],
    anthropic: [
      { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', provider: 'anthropic', recommended: true },
    ],
    offline: [
      { id: 'Meta-Llama-3.1-8B-Instruct', name: 'Meta Llama 3.1 8B Instruct', provider: 'offline', recommended: true },
    ],
  },
  last_updated: '2025-01-01T00:00:00Z',
};

// Helper to create fetch mock that handles both settings and models endpoints
const createFetchMock = (settingsResponses: any[]) => {
  let settingsCallIndex = 0;
  return jest.fn((url: string) => {
    if (url === '/api/v1/models/') {
      return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
    }
    // Settings endpoints
    const response = settingsResponses[settingsCallIndex++];
    return Promise.resolve(response);
  });
};

describe('Settings page', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('shows loading, then renders tabs', async () => {
    // @ts-ignore
    global.fetch = createFetchMock([{ ok: true, json: async () => baseSettings }]);

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
    // @ts-ignore
    global.fetch = createFetchMock([
      { ok: true, json: async () => baseSettings }, // initial GET
      { ok: true, json: async () => ({ ...baseSettings, default_temperature: 0.8 }) }, // PUT
    ]);

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
    // @ts-ignore
    global.fetch = createFetchMock([{ ok: true, json: async () => baseSettings }]);

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
    // @ts-ignore
    global.fetch = createFetchMock([
      { ok: true, json: async () => baseSettings }, // initial GET
      { ok: true, json: async () => baseSettings }, // POST reset
    ]);

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
    let callCount = 0;
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      // Settings endpoint - first call fails, second succeeds
      callCount++;
      if (callCount === 1) {
        return Promise.resolve({ ok: false, statusText: 'Server Error' });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    render(<Settings />);
    await waitFor(() => screen.getByText(/Error loading settings/i));

    fireEvent.click(screen.getByText('Retry'));
    await waitFor(() => screen.getByText('Settings'));
  });

  describe('Agent type auto-sync', () => {
    it('auto-syncs agent_type to online when selecting an online provider', async () => {
      let savedUpdates: any = null;
      // @ts-ignore
      global.fetch = jest.fn((url: string, options?: any) => {
        if (url === '/api/v1/models/') {
          return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
        }
        if (options?.method === 'PUT') {
          savedUpdates = JSON.parse(options.body);
          return Promise.resolve({ ok: true, json: async () => ({ ...baseSettings, ...savedUpdates }) });
        }
        return Promise.resolve({ ok: true, json: async () => baseSettings });
      });

      render(<Settings />);

      await waitFor(() => {
        expect(screen.getByText('Agent Config')).toBeInTheDocument();
      });

      // Switch to online agent type first to see the provider dropdown
      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'online' } });

      // Change the online provider
      const providerSelect = screen.getByLabelText('Online Provider');
      fireEvent.change(providerSelect, { target: { value: 'anthropic' } });

      // Save and verify agent_type is 'online'
      fireEvent.click(screen.getByText(/Save Changes/i));

      await waitFor(() => {
        expect(savedUpdates).not.toBeNull();
        expect(savedUpdates.default_agent_type).toBe('online');
        expect(savedUpdates.default_online_provider).toBe('anthropic');
      });
    });

    it('auto-syncs agent_type to online when selecting an online model', async () => {
      let savedUpdates: any = null;
      // @ts-ignore
      global.fetch = jest.fn((url: string, options?: any) => {
        if (url === '/api/v1/models/') {
          return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
        }
        if (options?.method === 'PUT') {
          savedUpdates = JSON.parse(options.body);
          return Promise.resolve({ ok: true, json: async () => ({ ...baseSettings, ...savedUpdates }) });
        }
        return Promise.resolve({ ok: true, json: async () => baseSettings });
      });

      render(<Settings />);

      await waitFor(() => {
        expect(screen.getByText('Agent Config')).toBeInTheDocument();
      });

      // Switch to online agent type to see the model dropdown
      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'online' } });

      // Change the online model
      const modelSelect = screen.getByLabelText('Online Model');
      fireEvent.change(modelSelect, { target: { value: 'gpt-4-turbo' } });

      // Save and verify agent_type is 'online'
      fireEvent.click(screen.getByText(/Save Changes/i));

      await waitFor(() => {
        expect(savedUpdates).not.toBeNull();
        expect(savedUpdates.default_agent_type).toBe('online');
        expect(savedUpdates.default_online_model).toBe('gpt-4-turbo');
      });
    });

    it('auto-syncs agent_type to local when selecting a local model', async () => {
      const onlineSettings = { ...baseSettings, default_agent_type: 'online' };
      let savedUpdates: any = null;
      // @ts-ignore
      global.fetch = jest.fn((url: string, options?: any) => {
        if (url === '/api/v1/models/') {
          return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
        }
        if (options?.method === 'PUT') {
          savedUpdates = JSON.parse(options.body);
          return Promise.resolve({ ok: true, json: async () => ({ ...onlineSettings, ...savedUpdates }) });
        }
        return Promise.resolve({ ok: true, json: async () => onlineSettings });
      });

      render(<Settings />);

      await waitFor(() => {
        expect(screen.getByText('Agent Config')).toBeInTheDocument();
      });

      // Switch to local agent type
      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'local' } });

      // Change the local model
      const modelSelect = screen.getByLabelText('Local Model');
      fireEvent.change(modelSelect, { target: { value: 'Meta-Llama-3.1-8B-Instruct' } });

      // Save and verify agent_type is 'local'
      fireEvent.click(screen.getByText(/Save Changes/i));

      await waitFor(() => {
        expect(savedUpdates).not.toBeNull();
        expect(savedUpdates.default_agent_type).toBe('local');
        expect(savedUpdates.default_local_model).toBe('Meta-Llama-3.1-8B-Instruct');
      });
    });
  });
});
