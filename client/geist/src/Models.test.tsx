import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import Models from './Models';


const mockUpdateSettings = jest.fn().mockResolvedValue(undefined);

jest.mock('./Hooks/useAvailableModels', () => ({
  __esModule: true,
  default: () => ({
    models: { providers: { offline: [] }, last_updated: null },
    loading: false,
    error: null,
    refetch: jest.fn(),
    providers: ['offline'],
  }),
}));

jest.mock('./Hooks/useUserSettings', () => ({
  __esModule: true,
  default: () => ({
    settings: {
      default_agent_type: 'local',
      default_local_model: 'legacy-model',
      default_local_artifact_id: null,
      default_online_model: 'gpt-4',
      default_online_provider: 'openai',
    },
    updateSettings: mockUpdateSettings,
  }),
}));

const artifact = {
  id: 'qwen3-4b-q4-k-m',
  model_id: 'Qwen/Qwen3-4B',
  display_name: 'Qwen3 4B Q4_K_M (GGUF)',
  format: 'gguf',
  backend: 'llama_server',
  quantization: 'Q4_K_M',
  status: 'not_installed',
  bytes_downloaded: 0,
  total_bytes: 2497280256,
  source: 'curated',
  supported: true,
};

const mlxArtifact = {
  id: 'meta-llama-3.1-8b-instruct-mlx',
  model_id: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
  display_name: 'Meta Llama 3.1 8B Instruct (MLX)',
  format: 'snapshot',
  backend: 'mlx_llama',
  quantization: 'MLX',
  status: 'installed',
  bytes_downloaded: 16000000000,
  total_bytes: 16000000000,
  progress_unit: 'files',
  progress_completed: 8,
  progress_total: 8,
  source: 'curated',
  supported: true,
  requires_auth: true,
};

let availableArtifacts = [artifact];

beforeEach(() => {
  mockUpdateSettings.mockClear();
  availableArtifacts = [artifact];
  global.fetch = jest.fn().mockImplementation((url: string, options?: RequestInit) => {
    if (url === '/api/v1/models/local/artifacts' && !options?.method) {
      return Promise.resolve({ ok: true, json: async () => ({ artifacts: availableArtifacts }) });
    }
    if (url.endsWith('/download')) {
      return Promise.resolve({ ok: true, json: async () => ({ ...artifact, status: 'queued' }) });
    }
    return Promise.resolve({ ok: true, json: async () => ({ artifacts: [artifact] }) });
  }) as jest.Mock;
});

it('shows, selects, and keeps GGUF controls out of the Apple silicon MLX view', async () => {
  availableArtifacts = [
    { ...artifact, supported: false },
    mlxArtifact,
  ];

  render(<Models />);

  await screen.findByText(mlxArtifact.display_name);
  expect(screen.queryByText(artifact.display_name)).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Import GGUF model')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Use' }));

  await waitFor(() => expect(mockUpdateSettings).toHaveBeenCalledWith({
    default_agent_type: 'local',
    default_local_model: mlxArtifact.model_id,
    default_local_artifact_id: mlxArtifact.id,
  }));
});

it('starts the built-in GGUF download from the Models page', async () => {
  render(<Models />);

  const download = await screen.findByRole('button', { name: 'Download' });
  fireEvent.click(download);
  expect(download).toBeDisabled();

  await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
    '/api/v1/models/local/artifacts/qwen3-4b-q4-k-m/download',
    { method: 'POST' },
  ));
  await waitFor(() => expect(download).not.toBeDisabled());
});

it('imports a local GGUF through the managed API', async () => {
  render(<Models />);
  await screen.findByText(artifact.display_name);
  const input = await screen.findByLabelText('Import GGUF model');
  const file = new File(['GGUFmodel'], 'local.gguf', { type: 'application/octet-stream' });

  fireEvent.change(input, { target: { files: [file] } });
  expect(input).toBeDisabled();

  await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
    '/api/v1/models/local/import',
    expect.objectContaining({ method: 'POST', body: expect.any(FormData) }),
  ));
  await waitFor(() => expect(input).not.toBeDisabled());
});
