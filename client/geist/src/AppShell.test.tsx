import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
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

const artifacts: any[] = [
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
    status: 'not_installed',
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
let availableArtifacts: any[] = artifacts;

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
    global.fetch = jest.fn().mockImplementation((url: string) => {
      if (url.endsWith('/download')) {
        const parts = url.split('/');
        const artifactId = parts[parts.length - 2];
        const artifact = availableArtifacts.find(item => item.id === artifactId);
        return Promise.resolve({
          ok: true,
          json: async () => ({ ...artifact, status: 'queued' }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ artifacts: availableArtifacts }),
      });
    });
  });

  it('shows a loading chip only until an empty model catalogue finishes loading', async () => {
    mockSettings();
    let resolveCatalog: ((response: unknown) => void) | undefined;
    (global.fetch as jest.Mock).mockImplementation(() => new Promise(resolve => {
      resolveCatalog = resolve;
    }));
    renderShell();

    const selector = screen.getByRole('combobox', { name: 'Local model' });
    expect(selector).toBeDisabled();
    expect(selector).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByRole('status')).toHaveTextContent('Loading models');

    await act(async () => {
      resolveCatalog?.({
        ok: true,
        json: async () => ({ artifacts: [] }),
      });
    });

    await waitFor(() => expect(selector).toBeDisabled());
    expect(selector).toBeDisabled();
    expect(selector).not.toHaveAttribute('aria-busy');
    expect(within(selector).getByRole('option')).toHaveTextContent('No local models');
  });

  it('shows a configured model with a simple not-installed state', async () => {
    mockSettings({
      default_local_model: 'Qwen/Qwen3.8-27B',
      default_local_artifact_id: null,
    });
    const missingArtifact = {
      ...artifacts[0],
      id: 'qwen3.8-27b-4bit-mlx',
      model_id: 'Qwen/Qwen3.8-27B',
      display_name: 'Qwen 3.8 27B 4-bit (MLX)',
      status: 'not_installed',
    };
    availableArtifacts = [missingArtifact];
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    expect(selector).toHaveValue('qwen3.8-27b-4bit-mlx');
    expect(screen.getByText('Not installed')).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalledWith(
      '/api/v1/models/local/artifacts/qwen3.8-27b-4bit-mlx/download',
      { method: 'POST' },
    );
  });

  it('lists all compatible local models without repeating a provider badge', async () => {
    mockSettings();
    renderShell();

    const summary = screen.getByLabelText('Current runtime');
    const selector = await within(summary).findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toHaveValue('qwen3-4b-q4-k-m'));

    expect(within(summary).getAllByText('local')).toHaveLength(1);
    expect(selector).toHaveValue('qwen3-4b-q4-k-m');
    expect(within(selector).getByRole('option', { name: 'Qwen3 4B Q4_K_M (GGUF)' })).toBeInTheDocument();
    expect(within(selector).getByRole('option', { name: 'Meta Llama 3.1 8B Instruct (MLX)' })).toBeInTheDocument();
    expect(within(selector).getByRole('option', { name: 'Qwen 3 8B' })).toBeInTheDocument();
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

  it('selects and starts installing a model from the runtime selector', async () => {
    const updateSettings = jest.fn().mockResolvedValue(undefined);
    mockSettings({}, updateSettings);
    availableArtifacts = [
      ...artifacts,
      {
        ...artifacts[2],
        id: 'qwen3-8b-not-installed',
        status: 'not_installed',
      },
    ];
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await within(selector).findByRole('option', { name: 'Qwen 3 8B' });
    await waitFor(() => expect(
      within(selector).getAllByRole('option', { name: 'Qwen 3 8B' }),
    ).toHaveLength(2));
    fireEvent.change(selector, { target: { value: 'qwen3-8b-not-installed' } });

    expect(await screen.findByText('Installing…')).toBeInTheDocument();
    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
      '/api/v1/models/local/artifacts/qwen3-8b-not-installed/download',
      { method: 'POST' },
    ));

    await waitFor(() => expect(updateSettings).toHaveBeenCalledWith({
      default_agent_type: 'local',
      default_local_model: 'Qwen/Qwen3-8B',
      default_local_artifact_id: 'qwen3-8b-not-installed',
    }));
  });

  it('shows the specific capacity error when a selected install cannot start', async () => {
    const updateSettings = jest.fn().mockResolvedValue(undefined);
    mockSettings({}, updateSettings);
    const missingArtifact = {
      ...artifacts[2],
      id: 'capacity-test-model',
      display_name: 'Capacity test model',
      status: 'not_installed',
    };
    availableArtifacts = [...artifacts, missingArtifact];
    (global.fetch as jest.Mock).mockImplementation((url: string) => {
      if (url.endsWith('/download')) {
        return Promise.resolve({
          ok: false,
          status: 507,
          statusText: 'Insufficient Storage',
          json: async () => ({
            detail: 'Not enough space. 15.2 GB needed; 512.0 MB available.',
          }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ artifacts: availableArtifacts }),
      });
    });
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await within(selector).findByRole('option', { name: 'Capacity test model' });
    await waitFor(() => expect(selector).toBeEnabled());
    fireEvent.change(selector, { target: { value: missingArtifact.id } });

    expect(await screen.findByRole('alert')).toHaveTextContent(
      '15.2 GB needed; 512.0 MB available.',
    );
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it('shows one install chip with percentage progress', async () => {
    mockSettings({
      default_local_model: 'Qwen/Qwen3-8B',
      default_local_artifact_id: 'qwen3-8b-downloading',
    });
    availableArtifacts = [{
      ...artifacts[2],
      status: 'downloading',
      bytes_downloaded: 25,
      total_bytes: 100,
      progress_unit: 'bytes',
      progress_completed: 25,
      progress_total: 100,
    }];
    renderShell();

    expect(await screen.findByText('Installing 25%')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Local model' })).toBeDisabled();
    expect(screen.getByRole('progressbar')).toHaveValue(25);
    expect(screen.queryByText(/download a local model/i)).not.toBeInTheDocument();
  });

  it('matches the active installed artifact when settings only contain a model ID', async () => {
    mockSettings({ default_local_artifact_id: null });
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toHaveValue('qwen3-4b-q4-k-m'));
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

    expect(await screen.findByRole('alert')).toHaveTextContent('Models unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    const selector = screen.getByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    expect(screen.queryByText('Models unavailable')).not.toBeInTheDocument();
  });

  it('restores the active model and reports an immediate-save failure', async () => {
    const updateSettings = jest.fn().mockRejectedValue(new Error('Server Error'));
    mockSettings({}, updateSettings);
    renderShell();

    const selector = await screen.findByRole('combobox', { name: 'Local model' });
    await waitFor(() => expect(selector).toBeEnabled());
    fireEvent.change(selector, { target: { value: 'meta-llama-3.1-8b-instruct-mlx' } });

    expect(await screen.findByRole('alert')).toHaveTextContent('Server Error');
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

  it('uses the packaged Geist profile as the default brand mark', () => {
    mockSettings();
    renderShell();

    const homeLink = screen.getByRole('link', { name: 'Geist home' });
    expect(homeLink.querySelector('.brand-mark-image')).toHaveAttribute('src', '/logo192.png');
    expect(homeLink.querySelector('.brand-mark-svg')).not.toBeInTheDocument();
    expect(within(homeLink).getByText('Private Local AI harness')).toBeInTheDocument();
  });
});
