import { renderHook, waitFor } from '@testing-library/react';
import useLocalRuntimeReadiness from '../useLocalRuntimeReadiness';
import { UserSettings } from '../useUserSettings';

const settings = {
  default_agent_type: 'local',
  default_local_model: 'Qwen/Qwen3.8-27B',
  default_local_artifact_id: 'qwen3.8-27b-4bit-mlx',
} as UserSettings;

const status = (state: 'loading' | 'ready' | 'failed', detail: string) => ({
  model_id: settings.default_local_model,
  state,
  detail,
  started_at: '2026-09-03T00:00:00Z',
  updated_at: '2026-09-03T00:00:01Z',
});

describe('useLocalRuntimeReadiness', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('surfaces a fatal preflight result before chat submission', async () => {
    const fetchMock = jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => status('failed', 'Installed model files are missing.'),
    } as Response);

    const { result } = renderHook(() => useLocalRuntimeReadiness(settings));

    await waitFor(() => expect(result.current.status?.state).toBe('failed'));
    expect(result.current.status?.detail).toContain('files are missing');
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/models/local/runtime/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
  });

  it('polls until a background local runtime load is ready', async () => {
    const fetchMock = jest.spyOn(global, 'fetch')
      .mockResolvedValueOnce({
        ok: true,
        json: async () => status('loading', 'Starting local runtime.'),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => status('ready', 'Model is loaded and ready.'),
      } as Response);

    const { result } = renderHook(() => useLocalRuntimeReadiness(settings));

    await waitFor(() => expect(result.current.status?.state).toBe('ready'));
    expect(fetchMock).toHaveBeenLastCalledWith(
      `/api/v1/models/status/${encodeURIComponent(settings.default_local_model)}`,
      { headers: { 'Content-Type': 'application/json' } },
    );
  });

  it('does not start the runtime before the selected artifact is installed', async () => {
    const fetchMock = jest.spyOn(global, 'fetch');

    const { result } = renderHook(() => useLocalRuntimeReadiness(settings, false));

    await waitFor(() => expect(result.current.status).toBeNull());
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
