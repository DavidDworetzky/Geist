import React from 'react';
import { render, screen, waitForElementToBeRemoved } from '@testing-library/react';
import LlamaComputeSection from '../LlamaComputeSection';


const props = {
  backend: null,
  deviceIds: [],
  onBackendChange: jest.fn(),
  onDeviceIdsChange: jest.fn(),
};

describe('LlamaComputeSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('is absent when no managed llama.cpp runtime is available', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: async () => ({
        available: false,
        managed_by_environment: false,
        forced_backend: null,
        devices: [],
        recommended_backend: 'cpu',
        recommended_device_ids: [],
        reason: 'A managed llama.cpp runtime is not installed on this platform.',
        error: null,
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    await waitForElementToBeRemoved(() => (
      screen.queryByText(/detecting llama\.cpp compute devices/i)
    ));
    expect(screen.queryByLabelText('Compute Backend')).not.toBeInTheDocument();
  });

  it('shows a locked state for operator-managed acceleration', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: async () => ({
        available: true,
        managed_by_environment: true,
        forced_backend: 'gpu',
        devices: [],
        recommended_backend: 'cpu',
        recommended_device_ids: [],
        reason: 'Managed by environment.',
        error: null,
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    expect(await screen.findByText(/managed by the environment \(GPU\)/i)).toBeInTheDocument();
    expect(screen.queryByLabelText('Compute Backend')).not.toBeInTheDocument();
  });

  it('keeps CPU selectable while disabling GPU when no devices exist', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: async () => ({
        available: true,
        managed_by_environment: false,
        forced_backend: null,
        devices: [],
        recommended_backend: 'cpu',
        recommended_device_ids: [],
        reason: 'No Vulkan devices were detected, so CPU is recommended.',
        error: null,
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    const select = await screen.findByLabelText('Compute Backend');
    expect(select).toHaveValue('cpu');
    expect(screen.getByRole('option', { name: 'CPU' })).not.toBeDisabled();
    expect(screen.getByRole('option', { name: 'GPU' })).toBeDisabled();
  });

  it('presents the automatic recommendation as the selected backend', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: async () => ({
        available: true,
        managed_by_environment: false,
        forced_backend: null,
        devices: [{
          id: 'gpu-recommended',
          name: 'Recommended GPU',
          total_memory_mib: 8192,
          free_memory_mib: 6144,
          kind: 'discrete',
          recommended: true,
        }],
        recommended_backend: 'gpu',
        recommended_device_ids: ['gpu-recommended'],
        reason: 'Recommended GPU is the recommended discrete GPU.',
        error: null,
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    expect(await screen.findByLabelText('Compute Backend')).toHaveValue('gpu');
    expect(screen.getByRole('checkbox', { name: /Recommended GPU/i })).toBeChecked();
    expect(screen.queryByText(/pending|verified|verification/i)).not.toBeInTheDocument();
  });

  it('warns when a saved GPU is no longer available', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: async () => ({
        available: true,
        managed_by_environment: false,
        forced_backend: null,
        devices: [{
          id: 'gpu-current',
          name: 'Current GPU',
          total_memory_mib: 8192,
          free_memory_mib: 4096,
          kind: 'discrete',
          recommended: true,
        }],
        recommended_backend: 'gpu',
        recommended_device_ids: ['gpu-current'],
        reason: 'Current GPU is recommended.',
        error: null,
      }),
    }));

    render(<LlamaComputeSection {...props} backend="gpu" deviceIds={['gpu-missing']} />);
    expect(await screen.findByText(/previously selected GPU is unavailable/i)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Current GPU/i })).not.toBeChecked();
  });
});
