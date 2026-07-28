import React from 'react';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import AppShell from './AppShell';
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

function renderShell(): void {
  render(
    <MemoryRouter>
      <AppShell>
        <div>Content</div>
      </AppShell>
    </MemoryRouter>,
  );
}

it('does not repeat local as a provider badge', () => {
  mockUseUserSettings.mockReturnValue({
    settings: baseSettings,
    loading: false,
    error: null,
    updateSettings: jest.fn(),
    resetSettings: jest.fn(),
    refetch: jest.fn(),
  });

  renderShell();

  const summary = screen.getByLabelText('Current runtime');
  expect(within(summary).getAllByText('local')).toHaveLength(1);
  expect(within(summary).getByText('Qwen/Qwen3-4B')).toBeInTheDocument();
});

it('shows the provider badge for online inference', () => {
  mockUseUserSettings.mockReturnValue({
    settings: {
      ...baseSettings,
      default_agent_type: 'online',
    },
    loading: false,
    error: null,
    updateSettings: jest.fn(),
    resetSettings: jest.fn(),
    refetch: jest.fn(),
  });

  renderShell();

  const summary = screen.getByLabelText('Current runtime');
  expect(within(summary).getByText('online')).toBeInTheDocument();
  expect(within(summary).getByText('openai')).toBeInTheDocument();
  expect(within(summary).getByText('gpt-4o')).toBeInTheDocument();
});
