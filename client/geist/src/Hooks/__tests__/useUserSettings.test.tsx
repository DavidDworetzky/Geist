import { renderHook, act } from '@testing-library/react';
import { useUserSettings, UserSettings, UserSettingsUpdate } from '../useUserSettings';

const mockSettings: UserSettings = {
  user_settings_id: 1,
  user_id: 1,
  default_agent_type: 'local',
  default_local_model: 'Meta-Llama-3.1-8B-Instruct',
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
  create_date: '2025-01-01T00:00:00Z',
  update_date: '2025-01-01T00:00:00Z'
};

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

    const { result } = renderHook(() => useUserSettings());

    expect(result.current.loading).toBe(true);

    // wait for effect
    await act(async () => {});

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

    const { result } = renderHook(() => useUserSettings());
    await act(async () => {});

    expect(result.current.loading).toBe(false);
    expect(result.current.settings).toBeNull();
    expect(result.current.error).toMatch(/Failed to fetch settings/i);

    consoleErrorSpy.mockRestore();
  });

  it('updates settings (PUT success)', async () => {
    // initial fetch
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockSettings });
    const { result } = renderHook(() => useUserSettings());
    await act(async () => {});

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
    const { result } = renderHook(() => useUserSettings());
    await act(async () => {});

    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, statusText: 'Bad Request' });

    await act(async () => {
      await expect(result.current.updateSettings({ default_temperature: 0.9 })).rejects.toBeDefined();
    });

    expect(result.current.error).toMatch(/Failed to update settings/i);

    consoleErrorSpy.mockRestore();
  });

  it('resets settings (POST success)', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockSettings });
    const { result } = renderHook(() => useUserSettings());
    await act(async () => {});

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

    const { result } = renderHook(() => useUserSettings());
    await act(async () => {});

    await act(async () => {
      await result.current.refetch();
    });

    expect(result.current.settings?.default_top_p).toBe(0.95);
  });
});
