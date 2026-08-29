import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import AppShell from './AppShell';
import useLocalArtifacts from './Hooks/useLocalArtifacts';
import useUserSettings from './Hooks/useUserSettings';


jest.mock('./Hooks/useUserSettings');

const mockUseUserSettings = useUserSettings as jest.MockedFunction<typeof useUserSettings>;
const baseSettings = {
  user_settings_id: 1,
  user_id: 1,
  default_agent_type: 'local',
  default_local_model: 'Qwen/Qwen3-4B',
  default_local_artifact_id: 'qwen3-4b-q4-k-m',
  default_online_model: 'gpt-4o',
  default_online_provider: 'openai',
  default_file_archives: [],
  enable_rag_by_default: false,
  default_max_tokens: 1024,
  default_temperature: 1,
  default_top_p: 1,
  default_frequency_penalty: 0,
  default_presence_penalty: 0,
  backup_providers: [],
  ui_preferences: {},
  create_date: '2026-07-27T00:00:00Z',
  update_date: '2026-07-27T00:00:00Z',
};

const artifacts = [
  {
    id: 'qwen3-4b-q4-k-m',
    model_id: 'Qwen/Qwen3-4B',
    display_name: 'Qwen3 4B Q4_K_M (GGUF)',
    status: 'installed',
    supported: true,
  },
  {
    id: 'meta-llama-3.1-8b-instruct-mlx',
    model_id: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    display_name: 'Meta Llama 3.1 8B Instruct (MLX)',
    status: 'installed',
    supported: true,
  },
  {
    id: 'qwen3-8b-downloading',
    model_id: 'Qwen/Qwen3-8B',
    display_name: 'Qwen 3 8B',
    status: 'downloading',
    supported: true,
  },
  {
    id: 'unsupported-installed-model',
    model_id: 'example/unsupported',
    display_name: 'Unsupported model',
    status: 'installed',
    supported: false,
  },
];
let availableArtifacts = artifacts;

function ArtifactRefreshHarness(): JSX.Element {
  const { refreshLocalArtifacts } = useLocalArtifacts();
  return <button onClick={() => void refreshLocalArtifacts()}>Refresh artifacts</button>;
}

function renderShell(children: React.ReactNode = <div>Content</div>): void {
  render(
    <MemoryRouter>
      <AppShell>
        {children}
      </AppShell>
    </MemoryRouter>,
  );
}

function mockSettings(overrides = {}, updateSettings = jest.fn().mockResolvedValue(undefined)): void {
  mockUseUserSettings.mockReturnValue({
    settings: { ...baseSettings, ...overrides },
    loading: false,
    error: null,
    updateSettings,
    resetSettings: jest.fn(),
    refetch: jest.fn(),
  });
}

describe('AppShell runtime model selector', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    availableArtifacts = artifacts;
    global.fetch = jest.fn().mockImplementation(() => Promise.resolve({
      ok: true,
      json: async () => ({ artifacts: availableArtifacts }),
    }));
  });

  it('lists installed local models without repeating a provider badge', async () => {
    mockSettings();
    renderShell();

    const summary = screen.getByLabelText('Current runtime');
    const selector = await within(summary).findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());

    expect(within(summary).getAllByText('local')).toHaveLength(1);
    expect(selector).toHaveValue('qwen3-4b-q4-k-m');
    expect(within(selector).getByRole('option', { name: 'Qwen3 4B Q4_K_M (GGUF)' })).toBeInTheDocument();
    expect(within(selector).getByRole('option', { name: 'Meta Llama 3.1 8B Instruct (MLX)' })).toBeInTheDocument();
    expect(within(selector).queryByRole('option', { name: 'Qwen 3 8B' })).not.toBeInTheDocument();
    expect(within(selector).queryByRole('option', { name: 'Unsupported model' })).not.toBeInTheDocument();
  });

  it('immediately saves the selected installed model and artifact', async () => {
    const updateSettings = jest.fn().mockResolvedValue(undefined);
    mockSettings({}, updateSettings);
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    fireEvent.change(selector, { target: { value: 'meta-llama-3.1-8b-instruct-mlx' } });

    await waitFor(() => {
      expect(updateSettings).toHaveBeenCalledWith({
        default_agent_type: 'local',
        default_local_model: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        default_local_artifact_id: 'meta-llama-3.1-8b-instruct-mlx',
      });
      expect(selector).toBeEnabled();
    });
  });

  it('matches the active installed artifact when settings only contain a model ID', async () => {
    mockSettings({ default_local_artifact_id: null });
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    expect(selector).toHaveValue('qwen3-4b-q4-k-m');
  });

  it('updates from a refresh triggered by another local-artifact consumer', async () => {
    mockSettings();
    renderShell(<ArtifactRefreshHarness />);

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    availableArtifacts = [
      ...artifacts,
      {
        id: 'newly-installed-model',
        model_id: 'example/newly-installed',
        display_name: 'Newly installed model',
        status: 'installed',
        supported: true,
      },
    ];

    fireEvent.click(screen.getByRole('button', { name: 'Refresh artifacts' }));

    expect(await within(selector).findByRole('option', { name: 'Newly installed model' }))
      .toBeInTheDocument();
  });

  it('retries after a transient artifact fetch failure', async () => {
    mockSettings();
    (global.fetch as jest.Mock)
      .mockRejectedValueOnce(new Error('Temporary failure'))
      .mockResolvedValue({
        ok: true,
        json: async () => ({ artifacts }),
      });
    renderShell();

    expect(await screen.findByRole('alert')).toHaveTextContent('Installed models unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    const selector = screen.getByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    expect(screen.queryByText('Installed models unavailable')).not.toBeInTheDocument();
  });

  it('restores the active model and reports an immediate-save failure', async () => {
    const updateSettings = jest.fn().mockRejectedValue(new Error('Server Error'));
    mockSettings({}, updateSettings);
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    fireEvent.change(selector, { target: { value: 'meta-llama-3.1-8b-instruct-mlx' } });

    expect(await screen.findByRole('alert')).toHaveTextContent('Model switch failed');
    await waitFor(() => {
      expect(selector).toHaveValue('qwen3-4b-q4-k-m');
      expect(selector).toBeEnabled();
    });
    expect(selector).toHaveAttribute('aria-invalid', 'true');
  });

  it('shows the provider and read-only model for online inference', () => {
    mockSettings({ default_agent_type: 'online' });
    renderShell();

    const summary = screen.getByLabelText('Current runtime');
    expect(within(summary).getByText('online')).toBeInTheDocument();
    expect(within(summary).getByText('openai')).toBeInTheDocument();
    expect(within(summary).getByText('gpt-4o')).toBeInTheDocument();
    expect(within(summary).queryByRole('combobox', { name: 'Local model' })).not.toBeInTheDocument();
  });
});
