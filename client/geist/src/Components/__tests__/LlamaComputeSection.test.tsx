import React, { StrictMode } from 'react';
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  waitForElementToBeRemoved,
} from '@testing-library/react';
import LlamaComputeSection from '../LlamaComputeSection';


const props = {
  backend: null,
  deviceIds: [],
  onBackendChange: jest.fn(),
  onDeviceIdsChange: jest.fn(),
  onValidityChange: jest.fn(),
};

const discoveredGpuInventory = {
  available: true,
  managed_by_environment: false,
  forced_backend: null,
  devices: [{
    id: 'gpu-detected',
    compatibility_ids: [],
    name: 'Detected GPU',
    total_memory_mib: 8192,
    free_memory_mib: 6144,
    kind: 'discrete',
    recommended: true,
  }],
  recommended_backend: 'gpu',
  recommended_device_ids: ['gpu-detected'],
  reason: 'Detected GPU is recommended.',
  error: null,
  discovery_in_progress: false,
};

const discoveryInProgressInventory = {
  ...discoveredGpuInventory,
  devices: [],
  recommended_backend: 'cpu',
  recommended_device_ids: [],
  reason: 'GPU discovery is already in progress, so CPU is temporarily recommended.',
  error: 'llama.cpp device discovery is already in progress',
  discovery_in_progress: true,
};

describe('LlamaComputeSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('hides compute settings when the platform has no managed llama.cpp runtime', async () => {
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
    expect(screen.queryByRole('heading', { name: 'llama.cpp Compute' })).not.toBeInTheDocument();
    expect(screen.queryByText(/managed llama\.cpp runtime is not installed/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Refresh devices' })).not.toBeInTheDocument();
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
    expect(screen.getByRole('heading', { name: 'llama.cpp Compute' })).toBeInTheDocument();
    expect(screen.getByText(/managed CPU llama\.cpp runtime is unavailable/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh devices' })).toBeEnabled();
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
        reason: 'CUDA_VISIBLE_DEVICES restricts llama.cpp to the operator-selected GPU.',
        error: null,
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    expect(await screen.findByText(/managed by the environment \(GPU\)/i)).toBeInTheDocument();
    expect(screen.getByText(/CUDA_VISIBLE_DEVICES restricts llama\.cpp/i)).toBeInTheDocument();
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
        error: 'The Vulkan llama-server executable is unavailable',
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    const select = await screen.findByLabelText('Compute Backend');
    expect(select).toHaveValue('automatic');
    expect(screen.getByRole('option', { name: 'Automatic (CPU recommended)' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'CPU' })).not.toBeDisabled();
    expect(screen.getByRole('option', { name: 'GPU' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent(/Vulkan llama-server executable/i);
  });

  it('refreshes device discovery explicitly and reports refresh failures', async () => {
    const inventory = {
      available: true,
      managed_by_environment: false,
      forced_backend: null,
      devices: [],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'No Vulkan devices were detected, so CPU is recommended.',
      error: null,
    };
    type MockResponse = {
      ok: boolean;
      statusText: string;
      json: () => Promise<typeof inventory>;
    };
    let resolveRefresh: ((response: MockResponse) => void) | undefined;
    const refreshResponse = new Promise<MockResponse>(resolve => {
      resolveRefresh = resolve;
    });
    const fetchMock = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        statusText: 'OK',
        json: async () => inventory,
      })
      .mockReturnValueOnce(refreshResponse);
    // @ts-ignore
    global.fetch = fetchMock;

    render(<LlamaComputeSection {...props} />);
    const refreshButton = await screen.findByRole('button', { name: 'Refresh devices' });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/v1/models/local/runtime/devices',
      expect.objectContaining({ signal: expect.any(Object) }),
    );

    fireEvent.click(refreshButton);

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/v1/models/local/runtime/devices?refresh=true',
      expect.objectContaining({ signal: expect.any(Object) }),
    );
    expect(screen.getByRole('button', { name: 'Refreshing devices…' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent(/refreshing device discovery/i);

    await act(async () => {
      resolveRefresh?.({
        ok: false,
        statusText: 'Device probe timed out',
        json: async () => inventory,
      });
      await refreshResponse;
    });

    expect(await screen.findByRole('alert')).toHaveTextContent(/Device probe timed out/i);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh devices' })).toBeEnabled();
    expect(screen.getByLabelText('Compute Backend')).toBeInTheDocument();
  });

  it('acknowledges a coalesced refresh and reports when the device list is current', async () => {
    const fetchMock = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        statusText: 'OK',
        json: async () => discoveredGpuInventory,
      })
      .mockResolvedValueOnce({
        ok: true,
        statusText: 'OK',
        json: async () => discoveryInProgressInventory,
      })
      .mockResolvedValueOnce({
        ok: true,
        statusText: 'OK',
        json: async () => discoveredGpuInventory,
      });
    // @ts-ignore
    global.fetch = fetchMock;

    render(<LlamaComputeSection {...props} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Refresh devices' }));

    await waitFor(() => {
      expect(screen.getByRole('status')).toHaveTextContent(
        /device discovery is in progress.*waiting for the current results/i,
      );
    });
    expect(await screen.findByText(/device list is current/i)).toHaveAttribute(
      'role',
      'status',
    );
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      '/api/v1/models/local/runtime/devices',
      expect.objectContaining({ signal: expect.any(Object) }),
    );
    const refreshSignal = (fetchMock.mock.calls[1][1] as RequestInit).signal;
    expect((fetchMock.mock.calls[2][1] as RequestInit).signal).toBe(refreshSignal);
  });

  it('attributes an unresolved saved GPU to the failed inventory request', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: false,
      statusText: 'Device service unavailable',
    }));

    render(
      <LlamaComputeSection
        {...props}
        backend="gpu"
        deviceIds={['gpu-saved']}
      />,
    );

    const requestAlert = await screen.findByRole('alert');
    expect(requestAlert).toHaveTextContent(/device service unavailable/i);
    expect(requestAlert).toHaveAttribute('id', 'llama-compute-selection-validation');
    expect(screen.queryByText(/previously selected GPU is unavailable/i))
      .not.toBeInTheDocument();
    expect(screen.queryByText(/resolve the GPU device selection/i)).not.toBeInTheDocument();
    await waitFor(() => {
      expect(props.onValidityChange).toHaveBeenLastCalledWith(
        false,
        false,
        'Device inventory failed: Device service unavailable',
      );
    });
  });

  it('aborts a pending manual refresh when the section unmounts', async () => {
    const inventory = {
      available: true,
      managed_by_environment: false,
      forced_backend: null,
      devices: [],
      recommended_backend: 'cpu',
      recommended_device_ids: [],
      reason: 'No Vulkan devices were detected, so CPU is recommended.',
      error: null,
    };
    type MockResponse = {
      ok: boolean;
      statusText: string;
      json: () => Promise<typeof inventory>;
    };
    let resolveRefresh: ((response: MockResponse) => void) | undefined;
    const refreshResponse = new Promise<MockResponse>(resolve => {
      resolveRefresh = resolve;
    });
    const fetchMock = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        statusText: 'OK',
        json: async () => inventory,
      })
      .mockReturnValueOnce(refreshResponse);
    // @ts-ignore
    global.fetch = fetchMock;

    const { unmount } = render(<LlamaComputeSection {...props} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Refresh devices' }));
    const refreshOptions = fetchMock.mock.calls[1][1] as RequestInit;
    const refreshSignal = refreshOptions.signal as AbortSignal;
    expect(refreshSignal.aborted).toBe(false);

    unmount();

    expect(refreshSignal.aborted).toBe(true);
    await act(async () => {
      resolveRefresh?.({
        ok: true,
        statusText: 'OK',
        json: async () => inventory,
      });
      await refreshResponse;
    });
  });

  it('recovers from a StrictMode cold-probe race without settling the transient result', async () => {
    let requestCount = 0;
    const fetchMock = jest.fn((_url: string, options?: RequestInit) => {
      requestCount += 1;
      if (requestCount === 1) {
        return new Promise((_resolve, reject) => {
          const signal = options?.signal as AbortSignal;
          const rejectAsAborted = () => {
            const error = new Error('Aborted');
            error.name = 'AbortError';
            reject(error);
          };
          if (signal.aborted) {
            rejectAsAborted();
          } else {
            signal.addEventListener('abort', rejectAsAborted, { once: true });
          }
        });
      }
      const inventory = requestCount === 2
        ? discoveryInProgressInventory
        : discoveredGpuInventory;
      return Promise.resolve({
        ok: true,
        statusText: 'OK',
        json: async () => inventory,
      });
    });
    // @ts-ignore
    global.fetch = fetchMock;

    render(
      <StrictMode>
        <LlamaComputeSection
          {...props}
          backend="gpu"
          deviceIds={['gpu-detected']}
        />
      </StrictMode>,
    );

    expect(await screen.findByRole('checkbox', { name: /Detected GPU/i })).toBeChecked();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      '/api/v1/models/local/runtime/devices',
      expect.objectContaining({ signal: expect.any(Object) }),
    );
    expect(props.onValidityChange.mock.calls.some(
      ([valid, settled]) => valid === false && settled === true,
    )).toBe(false);
    await waitFor(() => {
      expect(props.onValidityChange).toHaveBeenLastCalledWith(true, true, null);
    });
    expect(screen.queryByText(/device discovery is already in progress/i))
      .not.toBeInTheDocument();
  });

  it('stops retrying a persistent discovery marker and keeps it unsettled', async () => {
    jest.useFakeTimers();
    const retryStartedAt = Date.now();
    const requestTimes: number[] = [];
    const fetchMock = jest.fn((_url: string, _options?: RequestInit) => {
      requestTimes.push(Date.now());
      return Promise.resolve({
        ok: true,
        statusText: 'OK',
        json: async () => discoveryInProgressInventory,
      });
    });
    // @ts-ignore
    global.fetch = fetchMock;

    render(
      <LlamaComputeSection
        {...props}
        backend="gpu"
        deviceIds={['gpu-detected']}
      />,
    );
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    for (let retry = 0; retry < 9; retry += 1) {
      await act(async () => {
        jest.runOnlyPendingTimers();
        await Promise.resolve();
        await Promise.resolve();
      });
    }

    expect(fetchMock).toHaveBeenCalledTimes(10);
    expect(Date.now() - retryStartedAt).toBeGreaterThan(10000);
    const requestGaps = requestTimes.slice(1).map(
      (requestTime, index) => requestTime - requestTimes[index],
    );
    expect(Math.max(...requestGaps)).toBeLessThan(3000);
    const requestSignals = fetchMock.mock.calls.map(
      call => (call[1] as RequestInit).signal,
    );
    expect(new Set(requestSignals).size).toBe(1);
    expect(screen.getByText(/device discovery is already in progress/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh devices' })).toBeEnabled();
    expect(props.onValidityChange.mock.calls.some(
      ([valid, settled]) => valid === false && settled === true,
    )).toBe(false);
  });

  it('cancels a pending discovery-marker retry when unmounted', async () => {
    jest.useFakeTimers();
    const fetchMock = jest.fn((_url: string, _options?: RequestInit) => Promise.resolve({
      ok: true,
      statusText: 'OK',
      json: async () => discoveryInProgressInventory,
    }));
    // @ts-ignore
    global.fetch = fetchMock;

    const { unmount } = render(<LlamaComputeSection {...props} />);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    const requestOptions = fetchMock.mock.calls[0][1] as RequestInit;
    const signal = requestOptions.signal as AbortSignal;
    expect(signal.aborted).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    unmount();
    expect(signal.aborted).toBe(true);
    await act(async () => {
      jest.runOnlyPendingTimers();
      await Promise.resolve();
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('uses an unknown-kind GPU recommendation for an explicit GPU choice', async () => {
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
          kind: 'unknown',
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
    const recommendedGpu = screen.getByRole('checkbox', { name: /Recommended GPU/i });
    expect(recommendedGpu).toBeChecked();
    expect(recommendedGpu).toBeDisabled();

    fireEvent.change(select, { target: { value: 'gpu' } });

    expect(props.onDeviceIdsChange).toHaveBeenCalledWith(['gpu-recommended']);
    expect(props.onBackendChange).toHaveBeenCalledWith('gpu');
  });

  it('requires a deliberate choice when only non-recommended GPUs are available', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: async () => ({
        available: true,
        managed_by_environment: false,
        forced_backend: null,
        devices: [
          {
            id: 'gpu-integrated',
            name: 'Integrated GPU',
            total_memory_mib: 4096,
            free_memory_mib: 2048,
            kind: 'integrated',
            recommended: false,
          },
          {
            id: 'gpu-software',
            name: 'Software Vulkan Device',
            total_memory_mib: null,
            free_memory_mib: null,
            kind: 'software',
            recommended: false,
          },
          {
            id: 'gpu-discrete-fallback',
            name: 'Non-recommended Discrete GPU',
            total_memory_mib: 2048,
            free_memory_mib: 1024,
            kind: 'discrete',
            recommended: false,
          },
        ],
        recommended_backend: 'cpu',
        recommended_device_ids: [],
        reason: 'Only non-recommended Vulkan devices were detected.',
        error: null,
      }),
    }));

    const { rerender } = render(<LlamaComputeSection {...props} />);
    fireEvent.change(await screen.findByLabelText('Compute Backend'), {
      target: { value: 'gpu' },
    });

    expect(props.onDeviceIdsChange).toHaveBeenCalledWith([]);
    expect(props.onBackendChange).toHaveBeenCalledWith('gpu');

    rerender(<LlamaComputeSection {...props} backend="gpu" deviceIds={[]} />);
    expect(screen.getByRole('alert')).toHaveTextContent(
      /choose at least one available GPU device/i,
    );
    await waitFor(() => {
      expect(props.onValidityChange).toHaveBeenLastCalledWith(false, true, null);
    });
    const integratedGpu = screen.getByRole('checkbox', { name: /Integrated GPU/i });
    const softwareGpu = screen.getByRole('checkbox', { name: /Software Vulkan Device/i });
    const nonRecommendedDiscrete = screen.getByRole('checkbox', {
      name: /Non-recommended Discrete GPU/i,
    });
    expect(integratedGpu).not.toBeChecked();
    expect(integratedGpu).toBeEnabled();
    expect(softwareGpu).not.toBeChecked();
    expect(nonRecommendedDiscrete).not.toBeChecked();

    fireEvent.click(integratedGpu);
    expect(props.onDeviceIdsChange).toHaveBeenLastCalledWith(['gpu-integrated']);
    expect(props.onBackendChange).toHaveBeenLastCalledWith('gpu');

    rerender(
      <LlamaComputeSection {...props} backend="gpu" deviceIds={['gpu-integrated']} />,
    );
    await waitFor(() => {
      expect(props.onValidityChange).toHaveBeenLastCalledWith(true, true, null);
    });
  });

  it('maps legacy IDs to one canonical checked device and normalizes on interaction', async () => {
    // @ts-ignore
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true,
      json: async () => ({
        available: true,
        managed_by_environment: false,
        forced_backend: null,
        devices: [{
          id: 'gpu-stable',
          compatibility_ids: ['gpu-legacy'],
          name: 'Stable GPU',
          total_memory_mib: 12288,
          free_memory_mib: 10000,
          kind: 'discrete',
          recommended: true,
        }],
        recommended_backend: 'gpu',
        recommended_device_ids: ['gpu-stable'],
        reason: 'Stable GPU is recommended.',
        error: null,
      }),
    }));

    const { rerender } = render(
      <LlamaComputeSection
        {...props}
        backend="gpu"
        deviceIds={['gpu-legacy']}
      />,
    );
    const stableGpu = await screen.findByRole('checkbox', { name: /Stable GPU/i });
    expect(stableGpu).toBeChecked();
    expect(stableGpu).toBeDisabled();
    expect(screen.queryByText(/previously selected GPU is unavailable/i)).not.toBeInTheDocument();
    await waitFor(() => {
      expect(props.onValidityChange).toHaveBeenLastCalledWith(true, true, null);
    });

    rerender(
      <LlamaComputeSection
        {...props}
        backend="gpu"
        deviceIds={['gpu-legacy', 'gpu-stable']}
      />,
    );
    expect(await screen.findByText(/previously selected GPU is unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/resolve the GPU device selection before saving/i)).toHaveAttribute(
      'role',
      'alert',
    );
    await waitFor(() => {
      expect(props.onValidityChange).toHaveBeenLastCalledWith(false, true, null);
    });
    fireEvent.click(screen.getByRole('button', { name: 'Use available devices' }));
    expect(props.onDeviceIdsChange).toHaveBeenLastCalledWith(['gpu-stable']);
    expect(props.onBackendChange).toHaveBeenLastCalledWith('gpu');
  });

  it('falls back to an available GPU when the automatic recommendation is stale', async () => {
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
        recommended_device_ids: ['gpu-missing'],
        reason: 'Current GPU is recommended.',
        error: null,
      }),
    }));

    render(<LlamaComputeSection {...props} />);
    expect(await screen.findByText(/previously selected GPU is unavailable/i)).toHaveAttribute(
      'role',
      'alert',
    );
    expect(screen.getByRole('option', { name: /Automatic/i })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Current GPU/i })).not.toBeChecked();

    fireEvent.change(screen.getByLabelText('Compute Backend'), { target: { value: 'gpu' } });
    expect(props.onDeviceIdsChange).toHaveBeenCalledWith(['gpu-current']);
    expect(props.onBackendChange).toHaveBeenCalledWith('gpu');
  });

  it('can normalize a mixed available and stale explicit selection', async () => {
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

    render(
      <LlamaComputeSection
        {...props}
        backend="gpu"
        deviceIds={['gpu-current', 'gpu-missing']}
      />,
    );
    expect(await screen.findByText(/previously selected GPU is unavailable/i)).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Automatic/i })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Current GPU/i })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Use available devices' }));
    expect(props.onDeviceIdsChange).toHaveBeenCalledWith(['gpu-current']);
    expect(props.onBackendChange).toHaveBeenCalledWith('gpu');
  });
});
