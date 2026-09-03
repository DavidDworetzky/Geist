import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import Models from './Models';


const mockUpdateSettings = jest.fn().mockResolvedValue(undefined);
const mockUseAvailableModels = jest.fn();
const mockUseUserSettings = jest.fn();

jest.mock('./Hooks/useAvailableModels', () => ({
  __esModule: true,
  default: () => mockUseAvailableModels(),
}));

jest.mock('./Hooks/useUserSettings', () => ({
  __esModule: true,
  default: () => mockUseUserSettings(),
}));

const defaultUserSettingsHook = () => ({
  settings: {
    default_agent_type: 'local',
    default_local_model: 'legacy-model',
    default_local_artifact_id: null,
    default_online_model: 'gpt-4',
    default_online_provider: 'openai',
  },
  loading: false,
  updateSettings: mockUpdateSettings,
});

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
  repo_id: 'Qwen/Qwen3-4B-GGUF',
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
  repo_id: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
  supported: true,
  requires_auth: true,
};

let availableArtifacts: any[] = [artifact];

beforeEach(() => {
  mockUpdateSettings.mockClear();
  mockUseUserSettings.mockReturnValue(defaultUserSettingsHook());
  mockUseAvailableModels.mockReturnValue({
    models: { providers: { offline: [] }, last_updated: null },
    loading: false,
    error: null,
    refetch: jest.fn(),
    providers: ['offline'],
  });
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

it('separates local models from online models', async () => {
  render(<Models />);

  const localModelsTab = screen.getByRole('tab', { name: 'Local' });
  const onlineModelsTab = screen.getByRole('tab', { name: 'Online' });

  expect(localModelsTab).toHaveAttribute('aria-selected', 'true');
  expect(await screen.findByRole('region', { name: 'Local models' })).toBeInTheDocument();
  expect(screen.queryByText('Inference mode')).not.toBeInTheDocument();

  fireEvent.click(onlineModelsTab);

  expect(onlineModelsTab).toHaveAttribute('aria-selected', 'true');
  expect(screen.queryByRole('region', { name: 'Local models' })).not.toBeInTheDocument();
  expect(screen.getByText('Inference mode')).toBeInTheDocument();
  expect(screen.getByText('openai · gpt-4')).toBeInTheDocument();
  expect(screen.getByRole('region', { name: 'Online models' })).toBeInTheDocument();
});

it('shows only local artifacts compatible with the current architecture', async () => {
  availableArtifacts = [
    { ...artifact, supported: false },
    mlxArtifact,
  ];

  render(<Models />);

  await screen.findByText(mlxArtifact.display_name);
  expect(screen.queryByText(artifact.display_name)).not.toBeInTheDocument();
  expect(screen.queryByText('unavailable on this platform')).not.toBeInTheDocument();
  expect(screen.getByText('Installed')).toBeInTheDocument();
  expect(screen.queryByLabelText('Import GGUF model')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Use' }));

  await waitFor(() => expect(mockUpdateSettings).toHaveBeenCalledWith({
    default_agent_type: 'local',
    default_local_model: mlxArtifact.model_id,
    default_local_artifact_id: mlxArtifact.id,
  }));
});

it('does not label a stale selected artifact as downloaded', async () => {
  mockUseUserSettings.mockReturnValue({
    ...defaultUserSettingsHook(),
    settings: {
      ...defaultUserSettingsHook().settings,
      default_local_model: artifact.model_id,
      default_local_artifact_id: artifact.id,
    },
  });

  render(<Models />);

  expect(await screen.findByText('Not installed')).toBeInTheDocument();
  expect(screen.queryByText('Installed · Active')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Install' })).toBeInTheDocument();
});

it('shows install progress with a single cancel action', async () => {
  availableArtifacts = [{
    ...artifact,
    status: 'downloading',
    bytes_downloaded: 25,
    total_bytes: 100,
    progress_unit: 'bytes',
    progress_completed: 25,
    progress_total: 100,
  }];

  render(<Models />);

  expect(await screen.findByText('Installing 25%')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
});

it('shows no competing action while cancellation settles', async () => {
  availableArtifacts = [{
    ...artifact,
    status: 'cancelling',
    bytes_downloaded: 25,
    total_bytes: 100,
  }];

  render(<Models />);

  expect(await screen.findByText('Cancelling…')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
});

it('does not start a second model install while one is active', async () => {
  availableArtifacts = [
    { ...artifact, status: 'downloading', bytes_downloaded: 25, total_bytes: 100 },
    {
      ...artifact,
      id: 'second-model',
      model_id: 'Qwen/Second',
      display_name: 'Second model',
      status: 'not_installed',
    },
  ];

  render(<Models />);

  expect(await screen.findByText('Installing 25%')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Install' })).toBeDisabled();
});

it('shows a failed install with its retry action and error', async () => {
  availableArtifacts = [{
    ...artifact,
    status: 'failed',
    error: 'Not enough space to finish installing this model.',
  }];

  render(<Models />);

  expect(await screen.findByText('Install failed')).toBeInTheDocument();
  expect(screen.getByText('Not enough space to finish installing this model.'))
    .toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
});

it('collapses provider sections and selects a provider model', async () => {
  const openAiModel = {
    id: 'gpt-4o',
    name: 'GPT-4o',
    provider: 'openai',
    context_window: 128000,
    max_output_tokens: 16384,
    supports_vision: true,
    supports_function_calling: true,
    supports_streaming: true,
    recommended: true,
    family: 'gpt-4o',
  };
  mockUseAvailableModels.mockReturnValue({
    models: {
      providers: {
        openai: [openAiModel],
        anthropic: [],
        huggingface: [openAiModel],
        'self-hosted': [openAiModel],
        offline: [],
      },
      last_updated: null,
    },
    loading: false,
    error: null,
    refetch: jest.fn(),
    providers: ['openai', 'anthropic', 'huggingface', 'self-hosted', 'offline'],
  });

  render(<Models />);
  fireEvent.click(screen.getByRole('tab', { name: 'Online' }));

  const openAiToggle = screen.getByRole('button', { name: /openai 1 model/i });
  const anthropicToggle = screen.getByRole('button', { name: /anthropic 0 models/i });
  await waitFor(() => expect(openAiToggle).toHaveAttribute('aria-expanded', 'true'));
  expect(anthropicToggle).toHaveAttribute('aria-expanded', 'false');
  expect(screen.getByText('GPT-4o')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /huggingface/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /self-hosted/i })).not.toBeInTheDocument();

  fireEvent.click(openAiToggle);
  expect(openAiToggle).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByText('GPT-4o')).not.toBeInTheDocument();

  fireEvent.click(openAiToggle);
  fireEvent.click(screen.getByRole('button', { name: 'Use' }));
  await waitFor(() => expect(mockUpdateSettings).toHaveBeenCalledWith({
    default_agent_type: 'online',
    default_online_provider: 'openai',
    default_online_model: 'gpt-4o',
  }));
});

it('waits for settings before expanding the configured provider', async () => {
  mockUseAvailableModels.mockReturnValue({
    models: {
      providers: { openai: [], anthropic: [], offline: [] },
      last_updated: null,
    },
    loading: false,
    error: null,
    refetch: jest.fn(),
    providers: ['openai', 'anthropic', 'offline'],
  });
  mockUseUserSettings.mockReturnValue({
    ...defaultUserSettingsHook(),
    settings: null,
    loading: true,
  });

  const { rerender } = render(<Models />);
  fireEvent.click(screen.getByRole('tab', { name: 'Online' }));
  expect(screen.getByRole('button', { name: /openai 0 models/i }))
    .toHaveAttribute('aria-expanded', 'false');
  expect(screen.getByRole('button', { name: /anthropic 0 models/i }))
    .toHaveAttribute('aria-expanded', 'false');

  mockUseUserSettings.mockReturnValue({
    ...defaultUserSettingsHook(),
    settings: {
      ...defaultUserSettingsHook().settings,
      default_agent_type: 'online',
      default_online_provider: 'anthropic',
      default_online_model: 'claude-opus-4-5-20251101',
    },
    loading: false,
  });
  rerender(<Models />);

  await waitFor(() => expect(screen.getByRole('button', { name: /anthropic 0 models/i }))
    .toHaveAttribute('aria-expanded', 'true'));
  expect(screen.getByRole('button', { name: /openai 0 models/i }))
    .toHaveAttribute('aria-expanded', 'false');
});

it('installs from the Models page without changing the active model', async () => {
  render(<Models />);

  const localModelPanel = await screen.findByRole('region', { name: 'Local models' });
  expect(localModelPanel).toHaveClass('local-model-panel');
  const download = await screen.findByRole('button', { name: 'Install' });
  fireEvent.click(download);
  expect(download).toBeDisabled();

  await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
    '/api/v1/models/local/artifacts/qwen3-4b-q4-k-m/download',
    { method: 'POST' },
  ));
  expect(mockUpdateSettings).not.toHaveBeenCalled();
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
