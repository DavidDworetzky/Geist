import React, { ReactNode } from 'react';
import {
  renderHook,
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import {
  useUserSettings,
  UserSettings,
  UserSettingsProvider,
  UserSettingsUpdate,
} from '../useUserSettings';

const mockSettings: UserSettings = {
  user_settings_id: 1,
  user_id: 1,
  default_agent_type: 'local',
  default_local_model: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
  default_local_artifact_id: 'meta-llama-3.1-8b-instruct',
  llama_backend: 'gpu',
  llama_gpu_device_ids: ['gpu-nvidia'],
  default_online_model: 'gpt-4',
  default_online_provider: 'openai',
  default_file_archives: [101, 102],
  enable_rag_by_default: true,
  default_max_tokens: 512,
  default_temperature: 0.7,
  default_top_p: 0.9,
  default_frequency_penalty: 0,
  default_presence_penalty: 0,
  backup_providers: [],
  ui_preferences: { theme: 'light' },
  agent_permissions: { mode: 'default', always_allow: [] },
  create_date: '2025-01-01T00:00:00Z',
  update_date: '2025-01-01T00:00:00Z'
};

const wrapper = ({ children }: { children: ReactNode }) => (
  <UserSettingsProvider>{children}</UserSettingsProvider>
);

describe('useUserSettings', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('fetches settings on mount (success)', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockSettings,
    });

    const { result } = renderHook(() => useUserSettings(), { wrapper });

    expect(result.current.loading).toBe(true);

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.settings).toEqual(mockSettings);
  });

  it('handles fetch error on mount', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      statusText: 'Server Error',
    });

    const { result } = renderHook(() => useUserSettings(), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.loading).toBe(false);
    expect(result.current.settings).toBeNull();
    expect(result.current.error).toMatch(/Failed to fetch settings/i);

    consoleErrorSpy.mockRestore();
  });

  it('updates settings (PUT success)', async () => {
    // initial fetch
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockSettings });
    const { result } = renderHook(() => useUserSettings(), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    // update
    const updated: UserSettings = { ...mockSettings, default_temperature: 0.8 };
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => updated });

    await act(async () => {
      const updates: UserSettingsUpdate = { default_temperature: 0.8 };
      await result.current.updateSettings(updates);
    });

    expect(result.current.error).toBeNull();
    expect(result.current.settings).toEqual(updated);
  });

  it('handles update error (PUT)', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockSettings });
    const { result } = renderHook(() => useUserSettings(), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, statusText: 'Bad Request' });

    await act(async () => {
      await expect(result.current.updateSettings({ default_temperature: 0.9 })).rejects.toBeDefined();
    });

    expect(result.current.error).toMatch(/Failed to update settings/i);

    consoleErrorSpy.mockRestore();
  });

  it('surfaces FastAPI detail from an update error', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
    const detail = 'Select at least one llama.cpp GPU device';

    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockSettings });
    const { result } = renderHook(() => useUserSettings(), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      statusText: 'Unprocessable Entity',
      json: async () => ({ detail }),
    });

    await act(async () => {
      await expect(result.current.updateSettings({
        llama_backend: 'gpu',
        llama_gpu_device_ids: [],
      })).rejects.toThrow(detail);
    });

    expect(result.current.error).toBe(detail);
    consoleErrorSpy.mockRestore();
  });

  it('resets settings (POST success)', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockSettings });
    const { result } = renderHook(() => useUserSettings(), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    const resetResponse: UserSettings = { ...mockSettings, default_temperature: 0.5 };
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => resetResponse });

    await act(async () => {
      await result.current.resetSettings();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.settings).toEqual(resetResponse);
  });

  it('refetch updates settings', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: true, json: async () => mockSettings })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ ...mockSettings, default_top_p: 0.95 }) });

    const { result } = renderHook(() => useUserSettings(), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.refetch();
    });

    expect(result.current.settings?.default_top_p).toBe(0.95);
  });

  it('shares model updates with every settings consumer', async () => {
    const updatedSettings: UserSettings = {
      ...mockSettings,
      default_local_model: 'Qwen/Qwen3-4B',
      default_local_artifact_id: 'qwen3-4b-q4-k-m',
    };
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: true, json: async () => mockSettings })
      .mockResolvedValueOnce({ ok: true, json: async () => updatedSettings });

    const ModelSelector = () => {
      const { updateSettings } = useUserSettings();
      return (
        <button
          type="button"
          onClick={() => void updateSettings({
            default_local_model: updatedSettings.default_local_model,
            default_local_artifact_id: updatedSettings.default_local_artifact_id,
          })}
        >
          Select Qwen
        </button>
      );
    };
    const RuntimeBadge = () => {
      const { settings } = useUserSettings();
      return <span aria-label="Runtime model">{settings?.default_local_model}</span>;
    };

    render(
      <UserSettingsProvider>
        <ModelSelector />
        <RuntimeBadge />
      </UserSettingsProvider>,
    );

    expect(await screen.findByText(mockSettings.default_local_model)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Select Qwen' }));
    expect(await screen.findByText(updatedSettings.default_local_model)).toBeInTheDocument();
  });
});
