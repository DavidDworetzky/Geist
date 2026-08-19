import React, { useEffect, useMemo, useState } from 'react';

type LlamaBackend = 'cpu' | 'gpu' | null;

interface LlamaDevice {
  id: string;
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
}

interface LlamaComputeSectionProps {
  backend: LlamaBackend;
  deviceIds: string[];
  onBackendChange: (backend: LlamaBackend) => void;
  onDeviceIdsChange: (deviceIds: string[]) => void;
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
}: LlamaComputeSectionProps): JSX.Element | null {
  const [inventory, setInventory] = useState<LlamaDeviceInventory | null>(null);
  const [loading, setLoading] = useState(true);
  const [requestError, setRequestError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/api/v1/models/local/runtime/devices');
        if (!response.ok) {
          throw new Error(`Device inventory failed: ${response.statusText}`);
        }
        const payload = await response.json();
        if (!cancelled) {
          setInventory(payload);
          setRequestError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setRequestError(error instanceof Error ? error.message : 'Device inventory failed');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const availableDeviceIds = useMemo(
    () => new Set(inventory?.devices?.map(device => device.id) ?? []),
    [inventory],
  );

  if (loading) {
    return <p className="settings-description">Detecting llama.cpp compute devices…</p>;
  }
  if (!inventory) {
    return requestError ? <div className="notice notice-warning">{requestError}</div> : null;
  }
  if (!inventory.available && !inventory.managed_by_environment) {
    return inventory.error
      ? <div className="notice notice-warning">{inventory.error}</div>
      : null;
  }

  const locked = inventory.managed_by_environment;
  const devices = inventory.devices ?? [];
  const gpuAvailable = devices.length > 0;
  const selectedBackend = backend ?? 'automatic';
  const effectiveBackend = backend ?? inventory.recommended_backend;
  const selectedDeviceIds = backend === null
    ? inventory.recommended_device_ids
    : deviceIds;
  const selectedAvailableDeviceIds = selectedDeviceIds.filter(
    deviceId => availableDeviceIds.has(deviceId),
  );
  const unavailableSelections = selectedDeviceIds.filter(
    deviceId => !availableDeviceIds.has(deviceId),
  );
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
      const validCurrent = deviceIds.filter(deviceId => availableDeviceIds.has(deviceId));
      const validRecommended = inventory.recommended_device_ids.filter(
        deviceId => availableDeviceIds.has(deviceId),
      );
      const initial = validCurrent.length > 0
        ? validCurrent
        : validRecommended.length > 0
          ? validRecommended
          : devices.slice(0, 1).map(device => device.id);
      onDeviceIdsChange(initial);
      onBackendChange('gpu');
    }
  };
  const toggleDevice = (deviceId: string) => {
    const next = selectedAvailableDeviceIds.includes(deviceId)
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
      <div className="settings-subsection-header">
        <div>
          <h4>llama.cpp Compute</h4>
          <p className="settings-description">
            Choose CPU or Vulkan GPU acceleration for local GGUF models.
          </p>
        </div>
      </div>

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

      {(inventory.error || requestError) && (
        <div className="notice notice-warning">{inventory.error || requestError}</div>
      )}

      {!locked && effectiveBackend === 'gpu' && (
        <fieldset className="llama-device-picker">
          <legend className="settings-label">GPU Devices</legend>
          <p className="settings-description">
            Select one or more devices. llama.cpp splits model layers across multiple GPUs.
          </p>
          {devices.map(device => (
            <label
              className={`llama-device-option${device.id === onlySelectedDeviceId ? ' llama-device-option-disabled' : ''}`}
              key={device.id}
            >
              <input
                type="checkbox"
                checked={selectedDeviceIds.includes(device.id)}
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
          {unavailableSelections.length > 0 && (
            <div className="notice notice-error">
              A previously selected GPU is unavailable. Choose the desired devices before saving.
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
