import React from 'react';
import { act, render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import Settings from './Settings';
import { BrandingProvider } from './branding';
import { UserSettingsProvider } from './Hooks/useUserSettings';
import { installResizeObserverMock } from './testUtils/mockResizeObserver';
import {
  GEIST_HOST_DEVELOPMENT_UPDATED_EVENT,
  GEIST_PLUGIN_API_VERSION,
  geistPluginRuntime,
} from './plugins/runtime';

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

const renderSettings = () => render(
  <UserSettingsProvider>
    <Settings />
  </UserSettingsProvider>,
);

const renderBrandedSettings = () => render(
  <BrandingProvider>
    <UserSettingsProvider>
      <Settings />
    </UserSettingsProvider>
  </BrandingProvider>,
);

describe('Settings page', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    delete window.__GEIST_BRANDING__;
    delete window.__GEIST_HOST_DEVELOPMENT__;
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('shows loading, then renders tabs', async () => {
    // @ts-ignore
    global.fetch = createFetchMock([{ ok: true, json: async () => baseSettings }]);

    renderSettings();
    expect(screen.getByText(/Loading settings/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Settings')).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'General' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Models and Providers' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Generation' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Files and RAG' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Appearance' })).toBeInTheDocument();
      expect(screen.queryByRole('tab', { name: 'Developer' })).not.toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'About' })).toBeInTheDocument();
    });
  });

  it('shows active plugins only after an explicit host-development update', async () => {
    // @ts-ignore
    global.fetch = createFetchMock([{ ok: true, json: async () => baseSettings }]);
    const unregister = geistPluginRuntime.register({
      apiVersion: GEIST_PLUGIN_API_VERSION,
      id: 'example.host-plugin',
      name: 'Example Host Plugin',
      provider: 'Example Host',
      version: '1.2.3',
      activate: () => undefined,
    });

    try {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'About' })).toBeInTheDocument();
      });
      expect(screen.queryByRole('tab', { name: 'Developer' })).not.toBeInTheDocument();

      act(() => {
        window.__GEIST_HOST_DEVELOPMENT__ = true;
        window.dispatchEvent(new Event(GEIST_HOST_DEVELOPMENT_UPDATED_EVENT));
      });

      fireEvent.click(await screen.findByRole('tab', { name: 'Developer' }));
      expect(screen.getByRole('heading', { name: 'Active Plugins' })).toBeInTheDocument();
      expect(screen.getByText('Example Host Plugin')).toBeInTheDocument();
      expect(
        screen.getByText('Example Host - example.host-plugin v1.2.3 - API 1')
      ).toBeInTheDocument();
      expect(screen.getByText('0/0 mounted')).toBeInTheDocument();
      expect(screen.getByText('Last error: None')).toBeInTheDocument();
    } finally {
      act(() => unregister());
    }
  });

  it('shows host, Geist, machine, and inference details on the About tab', async () => {
    window.__GEIST_BRANDING__ = {
      productName: 'Pitchblend',
      productVersion: '2.3.4',
      logoUrl: 'data:image/png;base64,cGl0Y2hibGVuZA==',
    };
    const systemInfo = {
      version: '0.4.2',
      spa: true,
      platform: {
        system: 'Windows',
        release: '11',
        machine: 'AMD64',
      },
      python: {
        version: '3.11.9',
      },
      inference: {
        mode: 'local',
        engine: 'llama_server',
        model: 'Qwen/Qwen3-4B',
        provider: null,
        acceleration: 'vulkan',
      },
    };

    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/system') {
        return Promise.resolve({ ok: true, json: async () => systemInfo });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderBrandedSettings();
    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'About' })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('tab', { name: 'About' }));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Pitchblend.*v2\.3\.4/i })).toBeInTheDocument();
      expect(screen.getByRole('img', { name: 'Pitchblend logo' })).toBeInTheDocument();
      expect(screen.getByText(/Powered by Geist/i)).toHaveTextContent('v0.4.2');
      expect(screen.getByText('Windows 11')).toBeInTheDocument();
      expect(screen.getByText('AMD64')).toBeInTheDocument();
      expect(screen.getByText('llama.cpp')).toBeInTheDocument();
      expect(screen.getByText('vulkan')).toBeInTheDocument();
      expect(screen.queryByText('Geist API')).not.toBeInTheDocument();
      expect(screen.queryByText('Inference Mode')).not.toBeInTheDocument();
      expect(screen.queryByText('Model')).not.toBeInTheDocument();
      expect(screen.queryByText('Provider')).not.toBeInTheDocument();
    });
    expect(screen.queryByRole('contentinfo', { name: 'Settings actions' })).not.toBeInTheDocument();
  });

  it('keeps About usable when connected to an older Geist system endpoint', async () => {
    window.__GEIST_BRANDING__ = {
      productName: 'Pitchblend',
      productVersion: '2.3.4',
    };

    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/system') {
        return Promise.resolve({
          ok: true,
          json: async () => ({ version: '0.3.0', spa: true }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderBrandedSettings();
    await waitFor(() => screen.getByRole('tab', { name: 'About' }));
    fireEvent.click(screen.getByRole('tab', { name: 'About' }));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Pitchblend.*v2\.3\.4/i })).toBeInTheDocument();
      expect(screen.getByText(/Powered by Geist/i)).toHaveTextContent('v0.3.0');
      expect(screen.getByText(/Restart Geist to load detailed machine/i)).toBeInTheDocument();
    });
  });

  it('keeps the settings actions outside the scrollable generation content', async () => {
    // @ts-ignore
    global.fetch = createFetchMock([{ ok: true, json: async () => baseSettings }]);

    const { container } = renderSettings();
    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'Generation' })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('tab', { name: 'Generation' }));

    const page = container.querySelector('.settings-page-interactive');
    const scrollRegion = container.querySelector('.settings-scroll-region');
    const actions = screen.getByRole('contentinfo', { name: 'Settings actions' });
    expect(page).not.toBeNull();
    expect(scrollRegion).not.toBeNull();
    expect(scrollRegion).toContainElement(screen.getByText('Generation Parameters'));
    expect(scrollRegion).not.toContainElement(actions);
    expect(actions.parentElement).toBe(page);
    expect(within(actions).getByRole('button', { name: 'Reset to Defaults' })).toBeInTheDocument();
    expect(within(actions).getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(within(actions).getByRole('button', { name: 'Save Changes' })).toBeInTheDocument();
  });

  it('aligns the settings actions with content when the scroll region overflows', async () => {
    const resizeObserver = installResizeObserverMock();
    // @ts-ignore
    global.fetch = createFetchMock([{ ok: true, json: async () => baseSettings }]);

    const renderResult = renderSettings();

    try {
      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'Generation' })).toBeInTheDocument();
      });

      const { container } = renderResult;
      const page = container.querySelector<HTMLDivElement>('.settings-page-interactive');
      const scrollRegion = container.querySelector<HTMLDivElement>('.settings-scroll-region');
      expect(page).not.toBeNull();
      expect(scrollRegion).not.toBeNull();
      expect(page).not.toHaveClass('settings-scrollbar-visible');

      let scrollHeight = 700;
      Object.defineProperty(scrollRegion, 'clientHeight', {
        configurable: true,
        get: () => 500,
      });
      Object.defineProperty(scrollRegion, 'scrollHeight', {
        configurable: true,
        get: () => scrollHeight,
      });

      act(() => resizeObserver.trigger(scrollRegion as HTMLDivElement));
      expect(page).toHaveClass('settings-scrollbar-visible');

      scrollHeight = 500;
      act(() => resizeObserver.trigger(scrollRegion as HTMLDivElement));
      expect(page).not.toHaveClass('settings-scrollbar-visible');
    } finally {
      renderResult.unmount();
      resizeObserver.restore();
    }
  });

  it('marks unsaved changes when local values change and saves', async () => {
    // @ts-ignore
    global.fetch = createFetchMock([
      { ok: true, json: async () => baseSettings }, // initial GET
      { ok: true, json: async () => ({ ...baseSettings, default_temperature: 0.8 }) }, // PUT
    ]);

    renderSettings();

    // Wait for the Generation tab to be visible (indicating loading is complete)
    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'Generation' })).toBeInTheDocument();
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

    renderSettings();

    // Wait for the Generation tab to be visible (indicating loading is complete)
    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'Generation' })).toBeInTheDocument();
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

    renderSettings();

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

    renderSettings();
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

      renderSettings();

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'General' })).toBeInTheDocument();
      });

      // Switch to online agent type first to see the provider dropdown
      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'online' } });

      fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

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

      renderSettings();

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'General' })).toBeInTheDocument();
      });

      // Switch to online agent type to see the model dropdown
      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'online' } });

      fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

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

      renderSettings();

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'General' })).toBeInTheDocument();
      });

      // Switch to local agent type
      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'local' } });

      fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

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
