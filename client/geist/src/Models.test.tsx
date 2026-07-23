import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Models from './Models';

const mockStartDownload = jest.fn();
const mockLocalModelsState: any = {
  localModels: [],
  detectedDirectories: [],
  weightsRoot: '/srv/geist/app/model_weights',
  loading: false,
  error: null,
  startDownload: mockStartDownload,
  refetch: jest.fn(),
};

jest.mock('./Hooks/useLocalModels', () => ({
  __esModule: true,
  useLocalModels: () => mockLocalModelsState,
  default: () => mockLocalModelsState,
}));

jest.mock('./Hooks/useAvailableModels', () => ({
  __esModule: true,
  useAvailableModels: () => ({
    models: { providers: {} },
    providers: [],
    loading: false,
    error: null,
    refetch: jest.fn(),
    getModelById: () => undefined,
    getModelsForProvider: () => [],
  }),
  default: () => ({
    models: { providers: {} },
    providers: [],
    loading: false,
    error: null,
    refetch: jest.fn(),
    getModelById: () => undefined,
    getModelsForProvider: () => [],
  }),
}));

jest.mock('./Hooks/useUserSettings', () => ({
  __esModule: true,
  useUserSettings: () => ({ settings: null, loading: false, error: null }),
  default: () => ({ settings: null, loading: false, error: null }),
}));

const baseModel = {
  family: 'qwen',
  backend: 'transformers',
  gated: false,
  parameter_count: '4B',
  weights_path: '/srv/geist/app/model_weights/Qwen_Qwen3-4B',
  size_bytes: 0,
  download_status: null,
  download_job_id: null,
  download_error: null,
};

describe('Models local weights panel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalModelsState.localModels = [];
    mockLocalModelsState.detectedDirectories = [];
  });

  it('shows download state and weights root', () => {
    mockLocalModelsState.localModels = [
      {
        ...baseModel,
        id: 'Qwen/Qwen3-4B',
        name: 'Qwen 3 4B (Local)',
        downloaded: true,
        size_bytes: 8_000_000_000,
      },
      {
        ...baseModel,
        id: 'Qwen/Qwen3-8B',
        name: 'Qwen 3 8B (Local)',
        downloaded: false,
      },
    ];
    mockLocalModelsState.detectedDirectories = [
      {
        directory: 'my_custom_finetune',
        weights_path: '/srv/geist/app/model_weights/my_custom_finetune',
        size_bytes: 123456789,
      },
    ];

    render(<Models />);

    expect(
      screen.getByText(/Download models from Hugging Face into \/srv\/geist\/app\/model_weights/)
    ).toBeInTheDocument();
    expect(screen.getByTestId('local-status-Qwen/Qwen3-4B')).toHaveTextContent('Downloaded');
    expect(screen.getByTestId('local-status-Qwen/Qwen3-8B')).toHaveTextContent('Not downloaded');
    expect(screen.getByText('my_custom_finetune')).toBeInTheDocument();
    expect(screen.getByText('Unmanaged')).toBeInTheDocument();
    // Downloaded models have no download button.
    expect(screen.getAllByRole('button', { name: 'Download' })).toHaveLength(1);
  });

  it('starts a download for a missing model', async () => {
    mockStartDownload.mockResolvedValue(undefined);
    mockLocalModelsState.localModels = [
      { ...baseModel, id: 'Qwen/Qwen3-8B', name: 'Qwen 3 8B (Local)', downloaded: false },
    ];

    render(<Models />);
    fireEvent.click(screen.getByRole('button', { name: 'Download' }));

    await waitFor(() => {
      expect(mockStartDownload).toHaveBeenCalledWith('Qwen/Qwen3-8B');
    });
  });

  it('shows failed downloads with a retry action and surfaces errors', async () => {
    mockStartDownload.mockRejectedValue(new Error('gated model: add a Hugging Face token'));
    mockLocalModelsState.localModels = [
      {
        ...baseModel,
        id: 'google/gemma-3-1b-it',
        name: 'Gemma 3 1B IT (Local)',
        gated: true,
        downloaded: false,
        download_status: 'failed',
        download_error: 'boom',
      },
    ];

    render(<Models />);

    expect(screen.getByTestId('local-status-google/gemma-3-1b-it')).toHaveTextContent(
      'Download failed'
    );
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    expect(
      await screen.findByText('gated model: add a Hugging Face token')
    ).toBeInTheDocument();
  });
});
