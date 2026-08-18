import React from 'react';
import { fireEvent, render, screen, waitForElementToBeRemoved } from '@testing-library/react';
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

  it('shows the inventory error when the managed CPU runtime is broken', async () => {
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
        reason: 'The managed CPU llama.cpp runtime is unavailable.',
        error: 'The CPU llama-server executable is required',
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    expect(await screen.findByText(/CPU llama-server executable is required/i)).toBeInTheDocument();
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

  it('keeps an automatic CPU recommendation selectable while disabling GPU', async () => {
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
    expect(select).toHaveValue('automatic');
    expect(screen.getByRole('option', { name: 'Automatic (CPU recommended)' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'CPU' })).not.toBeDisabled();
    expect(screen.getByRole('option', { name: 'GPU' })).toBeDisabled();
  });

  it('distinguishes an automatic GPU recommendation from an explicit GPU choice', async () => {
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
    const select = await screen.findByLabelText('Compute Backend');
    expect(select).toHaveValue('automatic');
    expect(screen.getByRole('option', { name: 'Automatic (GPU recommended)' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Recommended GPU/i })).toBeChecked();

    fireEvent.change(select, { target: { value: 'gpu' } });

    expect(props.onDeviceIdsChange).toHaveBeenCalledWith(['gpu-recommended']);
    expect(props.onBackendChange).toHaveBeenCalledWith('gpu');
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
    expect(screen.queryByRole('option', { name: /Automatic/i })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Current GPU/i })).not.toBeChecked();
  });
});
