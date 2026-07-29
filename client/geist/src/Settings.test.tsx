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
  default_local_model: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
  default_local_artifact_id: null,
  llama_backend: null,
  llama_gpu_device_ids: [],
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
  agent_permissions: { mode: 'default', always_allow: [] },
  create_date: '2025-01-01T00:00:00Z',
  update_date: '2025-01-01T00:00:00Z'
};

const mockDeviceInventory = {
  available: true,
  managed_by_environment: false,
  forced_backend: null,
  devices: [
    {
      id: 'gpu-nvidia',
      compatibility_ids: [],
      name: 'NVIDIA GeForce RTX 3080',
      total_memory_mib: 16384,
      free_memory_mib: 12000,
      kind: 'discrete',
      recommended: true,
    },
    {
      id: 'gpu-intel',
      compatibility_ids: [],
      name: 'Intel(R) UHD Graphics',
      total_memory_mib: 2048,
      free_memory_mib: 1024,
      kind: 'integrated',
      recommended: false,
    },
  ],
  recommended_backend: 'gpu',
  recommended_device_ids: ['gpu-nvidia'],
  reason: 'NVIDIA GeForce RTX 3080 is the recommended discrete GPU.',
  error: null,
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
      { id: 'meta-llama/Meta-Llama-3.1-8B-Instruct', name: 'Meta Llama 3.1 8B Instruct', provider: 'offline', recommended: true },
      { id: 'Qwen/Qwen3-4B', name: 'Qwen 3 4B', provider: 'offline', recommended: true },
    ],
  },
  last_updated: '2025-01-01T00:00:00Z',
};

// Helper to create fetch mock that handles both settings and models endpoints
const createFetchMock = (settingsResponses: any[], inventory = mockDeviceInventory) => {
  let mutationCallIndex = 1;
  return jest.fn((url: string, options?: RequestInit) => {
    if (url === '/api/v1/models/') {
      return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
    }
    if (url === '/api/v1/models/local/runtime/devices') {
      return Promise.resolve({ ok: true, json: async () => inventory });
    }
    const response = !options?.method || options.method === 'GET'
      ? settingsResponses[0]
      : settingsResponses[mutationCallIndex++];
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

const waitForSettingsRefresh = async () => {
  const controls = await screen.findByRole('group', { name: 'Settings controls' });
  await waitFor(() => expect(controls).toHaveAttribute('aria-busy', 'false'));
};

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
      expect(screen.getByRole('tab', { name: 'Permissions' })).toBeInTheDocument();
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
    await waitForSettingsRefresh();

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

  it('selects the stored canonical local model', async () => {
    // @ts-ignore
    global.fetch = createFetchMock([{ ok: true, json: async () => baseSettings }]);

    renderSettings();

    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'Models and Providers' })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

    await waitFor(() => {
      expect(screen.getByLabelText('Local Model')).toHaveValue(
        'meta-llama/Meta-Llama-3.1-8B-Instruct'
      );
    });
  });

  it('shows the FastAPI detail when a settings update is rejected', async () => {
    const detail = 'The selected llama.cpp GPU is no longer available';
    // @ts-ignore
    global.fetch = jest.fn((url: string, options?: RequestInit) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (options?.method === 'PUT') {
        return Promise.resolve({
          ok: false,
          statusText: 'Unprocessable Entity',
          json: async () => ({ detail }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Generation' }));
    await waitForSettingsRefresh();
    fireEvent.change(screen.getByRole('slider', { name: /Temperature/i }), {
      target: { value: '0.8' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(await screen.findByText(detail)).toBeInTheDocument();
    expect(screen.queryByText(/Failed to update settings: Unprocessable Entity/i))
      .not.toBeInTheDocument();
  });

  it('selects GPU explicitly from automatic and saves one or more devices', async () => {
    let savedUpdates: any = null;
    // @ts-ignore
    global.fetch = jest.fn((url: string, options?: any) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        return Promise.resolve({ ok: true, json: async () => mockDeviceInventory });
      }
      if (options?.method === 'PUT') {
        savedUpdates = JSON.parse(options.body);
        return Promise.resolve({ ok: true, json: async () => ({ ...baseSettings, ...savedUpdates }) });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();

    const computeBackend = await screen.findByLabelText('Compute Backend');
    expect(computeBackend).toHaveValue('automatic');
    fireEvent.change(computeBackend, { target: { value: 'gpu' } });
    expect(computeBackend).toHaveValue('gpu');

    const nvidia = screen.getByRole('checkbox', { name: /NVIDIA GeForce RTX 3080/i });
    const intel = screen.getByRole('checkbox', { name: /Intel\(R\) UHD Graphics/i });
    expect(nvidia).toBeChecked();
    expect(nvidia).toBeDisabled();
    expect(intel).not.toBeDisabled();
    expect(screen.getByText(/integrated.*not recommended/i)).toBeInTheDocument();
    fireEvent.click(intel);
    expect(nvidia).not.toBeDisabled();
    expect(intel).not.toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(savedUpdates.llama_backend).toBe('gpu');
    });
    expect(savedUpdates.llama_gpu_device_ids).toEqual(['gpu-nvidia', 'gpu-intel']);
  });

  it('blocks an empty explicit GPU selection until an available device is chosen', async () => {
    const manualDeviceInventory = {
      ...mockDeviceInventory,
      devices: [{
        id: 'gpu-integrated',
        compatibility_ids: [],
        name: 'Integrated GPU',
        total_memory_mib: 4096,
        free_memory_mib: 2048,
        kind: 'integrated',
        recommended: false,
      }],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'Only integrated Vulkan devices were detected, so CPU is recommended.',
    };
    let savedUpdates: any = null;
    let putCalls = 0;
    // @ts-ignore
    global.fetch = jest.fn((url: string, options?: any) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        return Promise.resolve({ ok: true, json: async () => manualDeviceInventory });
      }
      if (options?.method === 'PUT') {
        putCalls += 1;
        savedUpdates = JSON.parse(options.body);
        return Promise.resolve({ ok: true, json: async () => ({ ...baseSettings, ...savedUpdates }) });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    fireEvent.change(await screen.findByLabelText('Compute Backend'), {
      target: { value: 'gpu' },
    });

    await screen.findByText(
      /choose at least one available GPU device before saving/i,
    );
    const validation = screen.getByRole('alert');
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(save).toBeDisabled();
    expect(save).toHaveAttribute('aria-describedby', validation.id);
    fireEvent.click(save);
    expect(putCalls).toBe(0);

    const integratedGpu = screen.getByRole('checkbox', { name: /Integrated GPU/i });
    expect(integratedGpu).toBeEnabled();
    fireEvent.click(integratedGpu);

    await waitFor(() => expect(save).toBeEnabled());
    expect(save).not.toHaveAttribute('aria-describedby');
    fireEvent.click(save);
    await waitFor(() => expect(putCalls).toBe(1));
    expect(savedUpdates.llama_backend).toBe('gpu');
    expect(savedUpdates.llama_gpu_device_ids).toEqual(['gpu-integrated']);
  });

  it('allows an unrelated save after observing an invalid persisted compute selection', async () => {
    const invalidPersistedSettings = {
      ...baseSettings,
      llama_backend: 'gpu' as const,
      llama_gpu_device_ids: [],
    };
    const manualDeviceInventory = {
      ...mockDeviceInventory,
      devices: [{
        id: 'gpu-integrated',
        compatibility_ids: [],
        name: 'Integrated GPU',
        total_memory_mib: 4096,
        free_memory_mib: 2048,
        kind: 'integrated',
        recommended: false,
      }],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'Only integrated Vulkan devices were detected, so CPU is recommended.',
    };
    let savedUpdates: any = null;
    // @ts-ignore
    global.fetch = jest.fn((url: string, options?: RequestInit) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        return Promise.resolve({ ok: true, json: async () => manualDeviceInventory });
      }
      if (options?.method === 'PUT') {
        savedUpdates = JSON.parse(options.body as string);
        return Promise.resolve({
          ok: true,
          json: async () => ({ ...invalidPersistedSettings, ...savedUpdates }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => invalidPersistedSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    expect(await screen.findByText(
      /choose at least one available GPU device before saving/i,
    )).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'Generation' }));
    fireEvent.change(screen.getByRole('slider', { name: /Temperature/i }), {
      target: { value: '0.8' },
    });
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(save).toBeEnabled();
    expect(save).not.toHaveAttribute('aria-describedby');
    fireEvent.click(save);

    await waitFor(() => expect(savedUpdates).not.toBeNull());
    expect(savedUpdates).not.toHaveProperty('llama_backend');
    expect(savedUpdates).not.toHaveProperty('llama_gpu_device_ids');
  });

  it('describes a dirty invalid compute edit when Models is showing the online agent', async () => {
    const manualDeviceInventory = {
      ...mockDeviceInventory,
      devices: [{
        id: 'gpu-integrated',
        compatibility_ids: [],
        name: 'Integrated GPU',
        total_memory_mib: 4096,
        free_memory_mib: 2048,
        kind: 'integrated',
        recommended: false,
      }],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'Only integrated Vulkan devices were detected, so CPU is recommended.',
    };
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        return Promise.resolve({ ok: true, json: async () => manualDeviceInventory });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    fireEvent.change(await screen.findByLabelText('Compute Backend'), {
      target: { value: 'gpu' },
    });
    expect(await screen.findByText(
      /choose at least one available GPU device before saving/i,
    )).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'General' }));
    fireEvent.change(screen.getByLabelText('Default Agent Type'), {
      target: { value: 'online' },
    });
    fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

    expect(await screen.findByLabelText('Online Provider')).toBeInTheDocument();
    const validation = screen.getByRole('alert');
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(validation).toHaveTextContent(/resolve the GPU device selection before saving/i);
    expect(save).toBeDisabled();
    expect(save).toHaveAttribute('aria-describedby', validation.id);
    expect(document.getElementById(save.getAttribute('aria-describedby') ?? ''))
      .toBe(validation);
  });

  it('restores cached valid compute state when cancelling an invalid GPU edit off-tab', async () => {
    const manualDeviceInventory = {
      ...mockDeviceInventory,
      devices: [{
        id: 'gpu-integrated',
        compatibility_ids: [],
        name: 'Integrated GPU',
        total_memory_mib: 4096,
        free_memory_mib: 2048,
        kind: 'integrated',
        recommended: false,
      }],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'Only integrated Vulkan devices were detected, so CPU is recommended.',
    };
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        return Promise.resolve({ ok: true, json: async () => manualDeviceInventory });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    fireEvent.change(await screen.findByLabelText('Compute Backend'), {
      target: { value: 'gpu' },
    });
    expect(await screen.findByText(
      /choose at least one available GPU device before saving/i,
    )).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'Generation' }));
    expect(screen.getByRole('alert')).toHaveTextContent(
      /resolve the GPU device selection before saving/i,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(screen.queryByText(
      /resolve the GPU device selection before saving/i,
    )).not.toBeInTheDocument();
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(save).not.toHaveAttribute('aria-describedby');
    fireEvent.change(screen.getByRole('slider', { name: /Temperature/i }), {
      target: { value: '0.8' },
    });
    expect(save).toBeEnabled();
  });

  it('does not gate an unrelated edit on an unresolved inventory request', async () => {
    const persistedGpuSettings = {
      ...baseSettings,
      llama_backend: 'gpu' as const,
      llama_gpu_device_ids: ['gpu-nvidia'],
    };
    let resolveInventory: ((response: any) => void) | null = null;
    const inventoryResponse = new Promise((resolve) => {
      resolveInventory = resolve;
    });
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        return inventoryResponse;
      }
      return Promise.resolve({ ok: true, json: async () => persistedGpuSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    expect(await screen.findByText(/Detecting llama\.cpp compute devices/i)).toBeInTheDocument();
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(save).not.toHaveAttribute('aria-describedby');

    fireEvent.click(screen.getByRole('tab', { name: 'Generation' }));
    fireEvent.change(screen.getByRole('slider', { name: /Temperature/i }), {
      target: { value: '0.8' },
    });
    expect(save).toBeEnabled();

    await act(async () => {
      resolveInventory?.({ ok: true, json: async () => mockDeviceInventory });
      await inventoryResponse;
    });
  });

  it('does not let refreshed compute validity block an unrelated dirty save', async () => {
    const persistedGpuSettings = {
      ...baseSettings,
      llama_backend: 'gpu' as const,
      llama_gpu_device_ids: ['gpu-nvidia'],
    };
    const inventoryWithoutSelectedGpu = {
      ...mockDeviceInventory,
      devices: [mockDeviceInventory.devices[1]],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'The previously selected GPU is no longer available.',
    };
    let inventoryCalls = 0;
    let resolveRemountInventory: ((response: any) => void) | null = null;
    const remountInventoryResponse = new Promise((resolve) => {
      resolveRemountInventory = resolve;
    });
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        inventoryCalls += 1;
        if (inventoryCalls === 1) {
          return Promise.resolve({ ok: true, json: async () => mockDeviceInventory });
        }
        return remountInventoryResponse;
      }
      return Promise.resolve({ ok: true, json: async () => persistedGpuSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    expect(await screen.findByRole('checkbox', {
      name: /NVIDIA GeForce RTX 3080/i,
    })).toBeChecked();

    fireEvent.click(screen.getByRole('tab', { name: 'Generation' }));
    fireEvent.change(screen.getByRole('slider', { name: /Temperature/i }), {
      target: { value: '0.8' },
    });
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(save).toBeEnabled();

    fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));
    expect(await screen.findByText(/Detecting llama\.cpp compute devices/i)).toBeInTheDocument();
    expect(save).toBeEnabled();
    expect(save).not.toHaveAttribute('aria-describedby');

    await act(async () => {
      resolveRemountInventory?.({
        ok: true,
        json: async () => inventoryWithoutSelectedGpu,
      });
      await remountInventoryResponse;
    });

    const validation = await screen.findByText(
      /^Resolve the GPU device selection before saving\.$/i,
    );
    expect(validation).toHaveTextContent(/resolve the GPU device selection before saving/i);
    expect(save).toBeEnabled();
    expect(save).not.toHaveAttribute('aria-describedby');
  });

  it('uses a failed inventory request to describe a blocked dirty compute edit', async () => {
    const manualDeviceInventory = {
      ...mockDeviceInventory,
      devices: [{
        id: 'gpu-integrated',
        compatibility_ids: [],
        name: 'Integrated GPU',
        total_memory_mib: 4096,
        free_memory_mib: 2048,
        kind: 'integrated',
        recommended: false,
      }],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'Only integrated Vulkan devices were detected, so CPU is recommended.',
    };
    let inventoryCalls = 0;
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        inventoryCalls += 1;
        if (inventoryCalls === 1) {
          return Promise.resolve({ ok: true, json: async () => manualDeviceInventory });
        }
        return Promise.resolve({
          ok: false,
          statusText: 'Device service unavailable',
        });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    fireEvent.change(await screen.findByLabelText('Compute Backend'), {
      target: { value: 'gpu' },
    });
    expect(await screen.findByText(
      /choose at least one available GPU device before saving/i,
    )).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'Generation' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

    const requestAlert = await screen.findByRole('alert');
    expect(requestAlert).toHaveTextContent(/device service unavailable/i);
    expect(requestAlert).toHaveAttribute('id', 'llama-compute-selection-validation');
    expect(screen.queryByText(/previously selected GPU is unavailable/i))
      .not.toBeInTheDocument();
    expect(screen.queryByText(/^Resolve the GPU device selection before saving\.$/i))
      .not.toBeInTheDocument();
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(save).toBeDisabled();
    expect(save).toHaveAttribute('aria-describedby', requestAlert.id);
    expect(document.getElementById(save.getAttribute('aria-describedby') ?? ''))
      .toBe(requestAlert);
  });

  it('keeps a dirty CPU selection valid when a later inventory request fails', async () => {
    let inventoryCalls = 0;
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        inventoryCalls += 1;
        if (inventoryCalls === 1) {
          return Promise.resolve({ ok: true, json: async () => mockDeviceInventory });
        }
        return Promise.resolve({
          ok: false,
          statusText: 'Device service unavailable',
        });
      }
      return Promise.resolve({ ok: true, json: async () => baseSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Models and Providers' }));
    await waitForSettingsRefresh();
    fireEvent.change(await screen.findByLabelText('Compute Backend'), {
      target: { value: 'cpu' },
    });

    fireEvent.click(screen.getByRole('tab', { name: 'Generation' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

    const requestAlert = await screen.findByRole('alert');
    expect(requestAlert).toHaveTextContent(/device service unavailable/i);
    expect(requestAlert).not.toHaveAttribute('id');
    const save = screen.getByRole('button', { name: 'Save Changes' });
    expect(save).toBeEnabled();
    expect(save).not.toHaveAttribute('aria-describedby');
  });

  it('omits stale persisted compute settings from an unrelated save', async () => {
    const stalePersistedSettings = {
      ...baseSettings,
      llama_backend: 'gpu' as const,
      llama_gpu_device_ids: ['gpu-from-an-older-runtime'],
    };
    let savedUpdates: any = null;
    // @ts-ignore
    global.fetch = jest.fn((url: string, options?: any) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (options?.method === 'PUT') {
        savedUpdates = JSON.parse(options.body);
        return Promise.resolve({
          ok: true,
          json: async () => ({ ...stalePersistedSettings, ...savedUpdates }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => stalePersistedSettings });
    });

    renderSettings();
    fireEvent.click(await screen.findByRole('tab', { name: 'Generation' }));
    await waitForSettingsRefresh();
    fireEvent.change(screen.getByRole('slider', { name: /Temperature/i }), {
      target: { value: '0.8' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(savedUpdates).not.toBeNull());
    expect(savedUpdates).not.toHaveProperty('llama_backend');
    expect(savedUpdates).not.toHaveProperty('llama_gpu_device_ids');
    expect((global.fetch as jest.Mock).mock.calls.map(([url]) => url)).not.toContain(
      '/api/v1/models/local/runtime/devices',
    );
  });

  it('keeps settings usable while refreshing a backend selection on mount', async () => {
    let settingsGets = 0;
    let resolveRefresh: ((response: any) => void) | null = null;
    const resolvedSettings = {
      ...baseSettings,
      llama_backend: 'gpu' as const,
      llama_gpu_device_ids: ['gpu-nvidia'],
    };
    // @ts-ignore
    global.fetch = jest.fn((url: string) => {
      if (url === '/api/v1/models/') {
        return Promise.resolve({ ok: true, json: async () => mockModelsResponse });
      }
      if (url === '/api/v1/models/local/runtime/devices') {
        return Promise.resolve({ ok: true, json: async () => mockDeviceInventory });
      }
      settingsGets += 1;
      if (settingsGets === 1) {
        return Promise.resolve({ ok: true, json: async () => baseSettings });
      }
      return new Promise((resolve) => {
        resolveRefresh = resolve;
      });
    });

    renderSettings();
    await waitFor(() => expect(settingsGets).toBe(2));
    expect(screen.queryByText(/Loading settings/i)).not.toBeInTheDocument();
    const modelsTab = screen.getByRole('tab', { name: 'Models and Providers' });
    const generationTab = screen.getByRole('tab', { name: 'Generation' });
    expect(modelsTab).not.toBeDisabled();
    expect(generationTab).not.toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent(
      /keep editing.*Save and Reset.*refresh finishes/i,
    );
    expect(screen.getByRole('group', { name: 'Settings controls' })).toHaveAttribute(
      'aria-busy',
      'true',
    );
    expect(screen.getByLabelText('Default Agent Type')).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reset to Defaults' })).toBeDisabled();

    fireEvent.click(generationTab);
    const temperature = screen.getByRole('slider', { name: /Temperature/i });
    expect(temperature).not.toBeDisabled();
    fireEvent.change(temperature, { target: { value: '0.8' } });
    expect(screen.getByText(/Unsaved Changes/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Save Changes' })).toBeDisabled();

    await act(async () => {
      resolveRefresh?.({ ok: true, json: async () => resolvedSettings });
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Save Changes' })).not.toBeDisabled();
    });
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Settings controls' })).toHaveAttribute(
      'aria-busy',
      'false',
    );
    expect(temperature).toHaveValue('0.8');
    expect(screen.getByRole('button', { name: 'Reset to Defaults' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).not.toBeDisabled();

    fireEvent.click(modelsTab);
    expect(await screen.findByLabelText('Compute Backend')).toHaveValue('gpu');
    expect(screen.queryByText(/pending|verified|verification/i)).not.toBeInTheDocument();
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
    await waitForSettingsRefresh();
    const slider = screen.getByRole('slider', { name: /Temperature/i }) as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '1.1' } });
    expect(screen.getByText(/Unsaved Changes/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText('Cancel'));
    expect(screen.queryByText(/Unsaved Changes/i)).not.toBeInTheDocument();
  });

  it('reset triggers API and shows success', async () => {
    const resolvedSettings = {
      ...baseSettings,
      llama_backend: 'gpu' as const,
      llama_gpu_device_ids: ['gpu-nvidia'],
    };
    // @ts-ignore
    global.fetch = createFetchMock([
      { ok: true, json: async () => resolvedSettings }, // initial GET
      { ok: true, json: async () => baseSettings }, // POST reset
    ]);

    // confirm window
    jest.spyOn(window, 'confirm').mockReturnValue(true);

    renderSettings();

    // Wait for the Reset button to be visible (indicating loading is complete)
    await waitFor(() => {
      expect(screen.getByText(/Reset to Defaults/i)).toBeInTheDocument();
    });
    await waitForSettingsRefresh();

    fireEvent.click(screen.getByText(/Reset to Defaults/i));
    await waitFor(() => {
      expect(screen.getByText(/Settings reset to defaults successfully/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));
    expect(await screen.findByLabelText('Compute Backend')).toHaveValue('automatic');
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
      await waitForSettingsRefresh();

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
      await waitForSettingsRefresh();

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
      await waitForSettingsRefresh();

      // Switch to local agent type
      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'local' } });

      fireEvent.click(screen.getByRole('tab', { name: 'Models and Providers' }));

      // Change the local model
      const modelSelect = screen.getByLabelText('Local Model');
      await waitFor(() => {
        expect(screen.getByRole('option', { name: 'Qwen 3 4B' })).toBeInTheDocument();
      });
      fireEvent.change(modelSelect, { target: { value: 'Qwen/Qwen3-4B' } });

      // Save and verify agent_type is 'local'
      fireEvent.click(screen.getByText(/Save Changes/i));

      await waitFor(() => {
        expect(savedUpdates).not.toBeNull();
        expect(savedUpdates.default_agent_type).toBe('local');
        expect(savedUpdates.default_local_model).toBe('Qwen/Qwen3-4B');
      });
    });
  });
});
