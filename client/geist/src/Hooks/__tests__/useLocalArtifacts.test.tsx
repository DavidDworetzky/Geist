import { act, renderHook } from '@testing-library/react';
import useLocalArtifacts, { LocalArtifact } from '../useLocalArtifacts';

jest.mock('../useUserSettings', () => ({
  __esModule: true,
  default: () => ({ updateSettings: jest.fn() }),
}));

const artifact = {
  id: 'qwen3.8-27b-4bit-mlx',
  model_id: 'Qwen/Qwen3.8-27B',
  display_name: 'Qwen 3.8 27B 4-bit (MLX)',
  format: 'snapshot',
  backend: 'mlx_llama',
  status: 'not_installed',
  bytes_downloaded: 0,
  total_bytes: 16_074_530_674,
  source: 'curated',
} as LocalArtifact;

describe('useLocalArtifacts', () => {
  afterEach(() => jest.restoreAllMocks());

  it('surfaces the backend storage detail when an install cannot start', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: false,
      status: 507,
      statusText: 'Insufficient Storage',
      json: async () => ({
        detail: 'Not enough space to install Qwen. 15.2 GB needed; 512.0 MB available.',
      }),
    } as Response);
    const { result } = renderHook(() => useLocalArtifacts({ enabled: false }));

    await expect(act(async () => result.current.downloadArtifact(artifact)))
      .rejects.toThrow('15.2 GB needed; 512.0 MB available');
  });
});
