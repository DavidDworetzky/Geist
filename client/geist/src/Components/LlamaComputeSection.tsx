import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type LlamaBackend = 'cpu' | 'gpu' | null;

interface LlamaDevice {
  id: string;
  compatibility_ids: string[];
  name: string;
  total_memory_mib: number | null;
  free_memory_mib: number | null;
  kind: 'discrete' | 'integrated' | 'software' | 'unknown';
  recommended: boolean;
}

interface LlamaDeviceInventory {
  available: boolean;
  managed_by_environment: boolean;
  forced_backend: 'cpu' | 'gpu' | null;
  devices: LlamaDevice[];
  recommended_backend: 'cpu' | 'gpu';
  recommended_device_ids: string[];
  reason: string;
  error: string | null;
  discovery_in_progress?: boolean;
}

interface LlamaComputeSectionProps {
  backend: LlamaBackend;
  deviceIds: string[];
  onBackendChange: (backend: LlamaBackend) => void;
  onDeviceIdsChange: (deviceIds: string[]) => void;
  onValidityChange: (valid: boolean, settled: boolean) => void;
}

export const LLAMA_COMPUTE_VALIDATION_MESSAGE_ID = 'llama-compute-selection-validation';

const DISCOVERY_RETRY_DELAYS_MS = [250, 500, 1000, 2000, 4000, 4000] as const;

function waitForDiscoveryRetry(delayMs: number, signal: AbortSignal): Promise<boolean> {
  return new Promise(resolve => {
    if (signal.aborted) {
      resolve(false);
      return;
    }

    const timeoutId = window.setTimeout(() => {
      signal.removeEventListener('abort', handleAbort);
      resolve(true);
    }, delayMs);
    const handleAbort = () => {
      window.clearTimeout(timeoutId);
      resolve(false);
    };
    signal.addEventListener('abort', handleAbort, { once: true });
  });
}

function memoryLabel(device: LlamaDevice): string {
  if (device.free_memory_mib !== null) {
    return `${device.free_memory_mib.toLocaleString()} MiB free`;
  }
  if (device.total_memory_mib !== null) {
    return `${device.total_memory_mib.toLocaleString()} MiB`;
  }
  return 'Memory unavailable';
}

export default function LlamaComputeSection({
  backend,
  deviceIds,
  onBackendChange,
  onDeviceIdsChange,
  onValidityChange,
}: LlamaComputeSectionProps): JSX.Element | null {
  const [inventory, setInventory] = useState<LlamaDeviceInventory | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const mounted = useRef(true);
  const activeRequest = useRef<AbortController | null>(null);

  const loadInventory = useCallback(async (refresh: boolean) => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    if (refresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setRequestError(null);
    try {
      const url = refresh
        ? '/api/v1/models/local/runtime/devices?refresh=true'
        : '/api/v1/models/local/runtime/devices';
      const fetchInventory = async (requestUrl: string): Promise<LlamaDeviceInventory | null> => {
        const response = await fetch(requestUrl, { signal: controller.signal });
        if (controller.signal.aborted) {
          return null;
        }
        if (!response.ok) {
          throw new Error(`Device inventory failed: ${response.statusText}`);
        }
        return response.json();
      };
      let payload = await fetchInventory(url);
      if (payload === null) {
        return;
      }
      for (const delayMs of DISCOVERY_RETRY_DELAYS_MS) {
        if (!payload.discovery_in_progress) {
          break;
        }
        if (
          !await waitForDiscoveryRetry(delayMs, controller.signal)
          || controller.signal.aborted
        ) {
          return;
        }
        payload = await fetchInventory('/api/v1/models/local/runtime/devices');
        if (payload === null) {
          return;
        }
      }
      if (payload.discovery_in_progress) {
        throw new Error(payload.error || 'llama.cpp device discovery is still in progress');
      }
      if (mounted.current && !controller.signal.aborted) {
        setInventory(payload);
      }
    } catch (error) {
      if (mounted.current && !controller.signal.aborted) {
        setRequestError(error instanceof Error ? error.message : 'Device inventory failed');
      }
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
      }
      if (mounted.current && !controller.signal.aborted) {
        if (refresh) {
          setRefreshing(false);
        } else {
          setLoading(false);
        }
      }
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void loadInventory(false);
    return () => {
      mounted.current = false;
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [loadInventory]);

  const devices = useMemo(() => inventory?.devices ?? [], [inventory]);
  const canonicalDeviceIdsBySelectionId = useMemo(() => {
    const canonicalIds = new Set(devices.map(device => device.id));
    const aliases = new Map<string, Set<string>>();
    const result = new Map<string, string>();

    devices.forEach(device => {
      result.set(device.id, device.id);
      (device.compatibility_ids ?? []).forEach(alias => {
        if (!alias || canonicalIds.has(alias)) {
          return;
        }
        const owners = aliases.get(alias) ?? new Set<string>();
        owners.add(device.id);
        aliases.set(alias, owners);
      });
    });
    aliases.forEach((owners, alias) => {
      if (owners.size === 1) {
        result.set(alias, Array.from(owners)[0]);
      }
    });
    return result;
  }, [devices]);
  const normalizeAvailableDeviceIds = (selectionIds: string[]) => {
    const normalized: string[] = [];
    const seen = new Set<string>();
    selectionIds.forEach(selectionId => {
      const canonicalId = canonicalDeviceIdsBySelectionId.get(selectionId);
      if (canonicalId && !seen.has(canonicalId)) {
        seen.add(canonicalId);
        normalized.push(canonicalId);
      }
    });
    return normalized;
  };
  const selectedDeviceIds = backend === null
    ? inventory?.recommended_device_ids ?? []
    : deviceIds;
  const selectedAvailableDeviceIds = normalizeAvailableDeviceIds(selectedDeviceIds);
  const selectedAvailableDeviceIdSet = new Set(selectedAvailableDeviceIds);
  const unavailableSelections = Array.from(new Set(selectedDeviceIds.filter(
    deviceId => !canonicalDeviceIdsBySelectionId.has(deviceId),
  )));
  const mappedSelectionCount = selectedDeviceIds.filter(
    deviceId => canonicalDeviceIdsBySelectionId.has(deviceId),
  ).length;
  const hasRedundantSelections = mappedSelectionCount > selectedAvailableDeviceIds.length;
  const unsupportedPlatform = Boolean(
    inventory
    && !inventory.available
    && !inventory.managed_by_environment
    && !inventory.error,
  );
  const computeSelectionValid = backend !== 'gpu'
    || Boolean(inventory?.managed_by_environment)
    || unsupportedPlatform
    || Boolean(
      inventory?.available
      && selectedAvailableDeviceIds.length > 0
      && unavailableSelections.length === 0
      && !hasRedundantSelections
    );
  const hasInvalidSelectedDeviceIds = unavailableSelections.length > 0 || hasRedundantSelections;
  const inventorySettled = !loading && inventory !== null;

  useEffect(() => {
    onValidityChange(computeSelectionValid, inventorySettled);
  }, [computeSelectionValid, inventorySettled, onValidityChange]);

  const gpuSelectionValidation = backend === 'gpu' && !computeSelectionValid && !loading ? (
    <div
      id={LLAMA_COMPUTE_VALIDATION_MESSAGE_ID}
      className="notice notice-warning"
      role="alert"
    >
      {hasInvalidSelectedDeviceIds
        ? 'Resolve the GPU device selection before saving.'
        : 'Choose at least one available GPU device before saving.'}
    </div>
  ) : null;

  const sectionHeader = (
    <div className="settings-subsection-header">
      <div>
        <h4>llama.cpp Compute</h4>
        <p className="settings-description">
          Choose CPU or Vulkan GPU acceleration for local GGUF models.
        </p>
      </div>
      <button
        type="button"
        className="button button-secondary button-small"
        disabled={loading || refreshing}
        aria-busy={loading || refreshing}
        onClick={() => void loadInventory(true)}
      >
        {loading
          ? 'Detecting devices…'
          : refreshing
            ? 'Refreshing devices…'
            : 'Refresh devices'}
      </button>
    </div>
  );

  if (loading) {
    return (
      <div className="llama-compute-section" aria-label="llama.cpp compute backend">
        {sectionHeader}
        <p
          id={backend === 'gpu' ? LLAMA_COMPUTE_VALIDATION_MESSAGE_ID : undefined}
          className="settings-description"
          aria-live="polite"
        >
          Detecting llama.cpp compute devices…
        </p>
        {gpuSelectionValidation}
      </div>
    );
  }
  if (!inventory) {
    return (
      <div className="llama-compute-section" aria-label="llama.cpp compute backend">
        {sectionHeader}
        {requestError && (
          <div className="notice notice-warning" role="alert">{requestError}</div>
        )}
        {gpuSelectionValidation}
      </div>
    );
  }
  if (unsupportedPlatform) {
    return null;
  }
  if (!inventory.available && !inventory.managed_by_environment) {
    return (
      <div className="llama-compute-section" aria-label="llama.cpp compute backend">
        {sectionHeader}
        <p className="settings-description">{inventory.reason}</p>
        {inventory.error && (
          <div className="notice notice-warning" role="alert">
            {inventory.error}
          </div>
        )}
        {requestError && (
          <div className="notice notice-warning" role="alert">{requestError}</div>
        )}
        {gpuSelectionValidation}
      </div>
    );
  }

  const locked = inventory.managed_by_environment;
  const gpuAvailable = devices.length > 0;
  const selectedBackend = backend ?? 'automatic';
  const effectiveBackend = backend ?? inventory.recommended_backend;
  const onlySelectedDeviceId = selectedAvailableDeviceIds.length === 1
    ? selectedAvailableDeviceIds[0]
    : null;
  const selectBackend = (value: string) => {
    if (value === 'cpu') {
      onDeviceIdsChange([]);
      onBackendChange('cpu');
      return;
    }
    if (value === 'gpu') {
      const validCurrent = normalizeAvailableDeviceIds(deviceIds);
      const validRecommended = normalizeAvailableDeviceIds(
        inventory.recommended_device_ids,
      ).filter(
        deviceId => devices.some(device => (
          device.id === deviceId
          && device.recommended
          && device.kind !== 'integrated'
          && device.kind !== 'software'
        )),
      );
      const discoveredRecommended = devices
        .filter(device => (
          device.recommended
          && device.kind !== 'integrated'
          && device.kind !== 'software'
          && !validRecommended.includes(device.id)
        ))
        .map(device => device.id);
      const initial = validCurrent.length > 0
        ? validCurrent
        : [...validRecommended, ...discoveredRecommended];
      onDeviceIdsChange(initial);
      onBackendChange('gpu');
    }
  };
  const toggleDevice = (deviceId: string) => {
    const next = selectedAvailableDeviceIdSet.has(deviceId)
      ? selectedAvailableDeviceIds.filter(value => value !== deviceId)
      : [...selectedAvailableDeviceIds, deviceId];
    if (next.length > 0) {
      onDeviceIdsChange(next);
      onBackendChange('gpu');
    }
  };
  const useAvailableDevices = () => {
    onDeviceIdsChange(selectedAvailableDeviceIds);
    onBackendChange('gpu');
  };

  return (
    <div className="llama-compute-section" aria-label="llama.cpp compute backend">
      {sectionHeader}

      {locked ? (
        <div className="notice">
          <div>
            Compute selection is managed by the environment
            {inventory.forced_backend ? ` (${inventory.forced_backend.toUpperCase()})` : ''}.
          </div>
          <p className="settings-description">{inventory.reason}</p>
        </div>
      ) : (
        <div className="settings-field">
          <label className="settings-label" htmlFor="settings-select-llama-backend">
            Compute Backend
          </label>
          <p className="settings-description">
            {inventory.reason}
          </p>
          <select
            id="settings-select-llama-backend"
            className="form-control settings-select-control"
            value={selectedBackend}
            onChange={event => selectBackend(event.target.value)}
          >
            {backend === null && (
              <option value="automatic">
                Automatic ({inventory.recommended_backend.toUpperCase()} recommended)
              </option>
            )}
            <option value="cpu">CPU</option>
            <option value="gpu" disabled={!gpuAvailable}>GPU</option>
          </select>
        </div>
      )}

      {inventory.error && (
        <div className="notice notice-warning" role="alert">{inventory.error}</div>
      )}
      {requestError && (
        <div className="notice notice-warning" role="alert">{requestError}</div>
      )}

      {!locked && effectiveBackend === 'gpu' && (
        <fieldset className="llama-device-picker">
          <legend className="settings-label">GPU Devices</legend>
          <p className="settings-description">
            Select one or more devices. llama.cpp splits model layers across multiple GPUs.
          </p>
          {gpuSelectionValidation}
          {devices.map(device => (
            <label
              className={`llama-device-option${device.id === onlySelectedDeviceId ? ' llama-device-option-disabled' : ''}`}
              key={device.id}
            >
              <input
                type="checkbox"
                checked={selectedAvailableDeviceIdSet.has(device.id)}
                disabled={device.id === onlySelectedDeviceId}
                onChange={() => toggleDevice(device.id)}
              />
              <span>
                <strong>{device.name}</strong>
                <small>
                  {memoryLabel(device)} · {device.kind}
                  {device.recommended ? ' · recommended' : ''}
                  {device.kind === 'integrated' ? ' · not recommended' : ''}
                </small>
              </span>
            </label>
          ))}
          {(unavailableSelections.length > 0 || hasRedundantSelections) && (
            <div className="notice notice-error" role="alert">
              A previously selected GPU is unavailable, duplicated, or no longer uniquely
              identifiable. Choose the desired devices before saving.
              {selectedAvailableDeviceIds.length > 0 && (
                <div>
                  <button
                    type="button"
                    className="button button-secondary llama-device-normalize"
                    onClick={useAvailableDevices}
                  >
                    Use available devices
                  </button>
                </div>
              )}
            </div>
          )}
        </fieldset>
      )}
    </div>
  );
}
