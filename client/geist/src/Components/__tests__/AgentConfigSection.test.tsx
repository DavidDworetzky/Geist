import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import AgentConfigSection from '../AgentConfigSection';

describe('AgentConfigSection', () => {
  const originalFetch = global.fetch;
  const defaultProps = {
    agentType: 'online',
    localModel: 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    onlineProvider: 'openai',
    onlineModel: 'gpt-4',
    onAgentTypeChange: jest.fn(),
    onOnlineProviderChange: jest.fn(),
    onOnlineModelChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  describe('Provider filtering', () => {
    it('shows only OpenAI models when OpenAI provider is selected', () => {
      render(<AgentConfigSection {...defaultProps} onlineProvider="openai" />);

      const modelSelect = screen.getByLabelText('Online Model');
      const options = modelSelect.querySelectorAll('option');

      const optionValues = Array.from(options).map((opt) => opt.getAttribute('value'));

      expect(optionValues).toContain('gpt-4');
      expect(optionValues).toContain('gpt-4-turbo');
      expect(optionValues).toContain('gpt-3.5-turbo');
      expect(optionValues).not.toContain('claude-3-opus-20240229');
      expect(optionValues).not.toContain('claude-3-sonnet-20240229');
    });

    it('shows only Anthropic models when Anthropic provider is selected', () => {
      render(<AgentConfigSection {...defaultProps} onlineProvider="anthropic" onlineModel="claude-3-opus-20240229" />);

      const modelSelect = screen.getByLabelText('Online Model');
      const options = modelSelect.querySelectorAll('option');

      const optionValues = Array.from(options).map((opt) => opt.getAttribute('value'));

      expect(optionValues).toContain('claude-fable-5-1');
      expect(optionValues).toContain('claude-mythos-5-1');
      expect(optionValues).toContain('claude-3-opus-20240229');
      expect(optionValues).toContain('claude-3-sonnet-20240229');
      expect(optionValues).not.toContain('gpt-4');
      expect(optionValues).not.toContain('gpt-4-turbo');
      expect(optionValues).not.toContain('gpt-3.5-turbo');
    });

    it('shows only xAI models when xAI provider is selected', () => {
      render(<AgentConfigSection {...defaultProps} onlineProvider="xai" onlineModel="grok-2" />);

      const modelSelect = screen.getByLabelText('Online Model');
      const options = modelSelect.querySelectorAll('option');

      const optionValues = Array.from(options).map((opt) => opt.getAttribute('value'));

      expect(optionValues).toContain('grok-2');
      expect(optionValues).toContain('grok-3');
      expect(optionValues).not.toContain('gpt-4');
      expect(optionValues).not.toContain('claude-3-opus-20240229');
    });
  });

  describe('Provider change behavior', () => {
    it('resets model to first available when switching from OpenAI to Anthropic', () => {
      const onOnlineProviderChange = jest.fn();
      const onOnlineModelChange = jest.fn();

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="openai"
          onlineModel="gpt-4"
          onOnlineProviderChange={onOnlineProviderChange}
          onOnlineModelChange={onOnlineModelChange}
        />
      );

      const providerSelect = screen.getByLabelText('Online Provider');
      fireEvent.change(providerSelect, { target: { value: 'anthropic' } });

      expect(onOnlineProviderChange).toHaveBeenCalledWith('anthropic');
      expect(onOnlineModelChange).toHaveBeenCalledWith('claude-fable-5-1');
    });

    it('resets model to first available when switching from Anthropic to OpenAI', () => {
      const onOnlineProviderChange = jest.fn();
      const onOnlineModelChange = jest.fn();

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="anthropic"
          onlineModel="claude-3-opus-20240229"
          onOnlineProviderChange={onOnlineProviderChange}
          onOnlineModelChange={onOnlineModelChange}
        />
      );

      const providerSelect = screen.getByLabelText('Online Provider');
      fireEvent.change(providerSelect, { target: { value: 'openai' } });

      expect(onOnlineProviderChange).toHaveBeenCalledWith('openai');
      // First recommended model for OpenAI is gpt-4-turbo
      expect(onOnlineModelChange).toHaveBeenCalledWith('gpt-4-turbo');
    });

    it('does not reset model when switching providers if model somehow matches', () => {
      // This is an edge case - the model already exists in the new provider's list
      // In our current implementation, this shouldn't happen as providers have distinct models
      // But the logic should handle it gracefully
      const onOnlineProviderChange = jest.fn();
      const onOnlineModelChange = jest.fn();

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="openai"
          onlineModel="gpt-4"
          onOnlineProviderChange={onOnlineProviderChange}
          onOnlineModelChange={onOnlineModelChange}
        />
      );

      // Changing to the same provider shouldn't trigger model change
      const providerSelect = screen.getByLabelText('Online Provider');
      fireEvent.change(providerSelect, { target: { value: 'openai' } });

      expect(onOnlineProviderChange).toHaveBeenCalledWith('openai');
      // Model should NOT be changed since gpt-4 is available for openai
      expect(onOnlineModelChange).not.toHaveBeenCalled();
    });
  });

  describe('Agent type switching', () => {
    it('directs local model management to the compatible artifact inventory', () => {
      render(<AgentConfigSection {...defaultProps} agentType="local" />);

      expect(screen.getByText('Local model')).toBeInTheDocument();
      expect(screen.queryByLabelText('Online Provider')).not.toBeInTheDocument();
      expect(screen.queryByLabelText('Online Model')).not.toBeInTheDocument();
      expect(screen.queryByLabelText('Local Model')).not.toBeInTheDocument();
      expect(screen.getByRole('link', { name: 'Manage local models' }))
        .toHaveAttribute('href', '/models');
    });

    it('shows online provider and model options for the persisted online agent type', () => {
      render(<AgentConfigSection {...defaultProps} agentType="online" />);

      expect(screen.queryByLabelText('Local Model')).not.toBeInTheDocument();
      const providerSelect = screen.getByLabelText('Online Provider');
      expect(providerSelect).toBeInTheDocument();
      expect(screen.getByLabelText('Online Model')).toBeInTheDocument();

      const providerValues = Array.from(providerSelect.querySelectorAll('option'))
        .map(option => option.getAttribute('value'));
      expect(providerValues).not.toContain('offline');
      expect(providerValues).not.toContain('huggingface');
      expect(providerValues).not.toContain('self-hosted');
    });
  });

  describe('Model selection', () => {
    it('calls onOnlineModelChange when a model is selected', () => {
      const onOnlineModelChange = jest.fn();

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="openai"
          onOnlineModelChange={onOnlineModelChange}
        />
      );

      const modelSelect = screen.getByLabelText('Online Model');
      fireEvent.change(modelSelect, { target: { value: 'gpt-4-turbo' } });

      expect(onOnlineModelChange).toHaveBeenCalledWith('gpt-4-turbo');
    });

  });

  describe('Rendering', () => {
    it('renders the component with correct title', () => {
      render(<AgentConfigSection {...defaultProps} />);

      expect(screen.getByText('Models and Providers')).toBeInTheDocument();
    });

    it('displays all provider options', () => {
      render(<AgentConfigSection {...defaultProps} />);

      const providerSelect = screen.getByLabelText('Online Provider');
      const options = providerSelect.querySelectorAll('option');
      const optionLabels = Array.from(options).map((opt) => opt.textContent);

      expect(optionLabels).toContain('OpenAI');
      expect(optionLabels).toContain('Anthropic');
      expect(optionLabels).toContain('xAI (Grok)');
      expect(optionLabels).toContain('Groq');
    });

    it('uses the OpenRouter display name from live catalog data', async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          providers: {
            openrouter: [{
              id: 'x-ai/grok-4.6',
              name: 'Grok 4.6',
              provider: 'openrouter',
              context_window: 500000,
              max_output_tokens: null,
              supports_vision: true,
              supports_function_calling: true,
              supports_streaming: true,
              recommended: true,
              family: 'grok',
            }],
          },
          last_updated: null,
        }),
      });

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="openrouter"
          onlineModel="x-ai/grok-4.6"
        />
      );

      expect(await screen.findByRole('option', { name: 'OpenRouter' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Grok 4.6' })).toBeInTheDocument();
    });

    it('shows Gemini 3.8 Flash under the Google Gemini provider', async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          providers: {
            google: [{
              id: 'gemini-3.8-flash',
              name: 'Gemini 3.8 Flash',
              provider: 'google',
              context_window: 1048576,
              max_output_tokens: 65536,
              supports_vision: true,
              supports_function_calling: true,
              supports_reasoning: true,
              supports_streaming: true,
              recommended: true,
              family: 'gemini',
            }],
          },
          last_updated: null,
        }),
      });

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="google"
          onlineModel="gemini-3.8-flash"
        />
      );

      expect(await screen.findByRole('option', { name: 'Google Gemini' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Gemini 3.8 Flash' })).toBeInTheDocument();
    });

    it('shows Muse Spark Contributor from live OpenRouter catalog data', async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          providers: {
            openrouter: [{
              id: 'meta/muse-spark-1.2-contributor',
              name: 'Muse Spark 1.2 Contributor',
              provider: 'openrouter',
              context_window: 1048576,
              max_output_tokens: null,
              supports_vision: true,
              supports_function_calling: true,
              supports_reasoning: true,
              supports_streaming: true,
              recommended: false,
              family: 'muse',
            }],
          },
          last_updated: null,
        }),
      });

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="openrouter"
          onlineModel="meta/muse-spark-1.2-contributor"
        />
      );

      expect(await screen.findByRole('option', { name: 'OpenRouter' })).toBeInTheDocument();
      expect(screen.getByRole('option', {
        name: 'Muse Spark 1.2 Contributor',
      })).toBeInTheDocument();
    });

    it('shows direct Muse Spark options from Meta Model API catalog data', async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          providers: {
            meta: [
              {
                id: 'muse-spark-1.1',
                name: 'Muse Spark 1.1',
                provider: 'meta',
                context_window: 1048576,
                max_output_tokens: null,
                supports_vision: true,
                supports_function_calling: true,
                supports_reasoning: true,
                supports_streaming: true,
                recommended: false,
                family: 'muse',
              },
              {
                id: 'muse-spark-1.2',
                name: 'Muse Spark 1.2',
                provider: 'meta',
                context_window: 1048576,
                max_output_tokens: null,
                supports_vision: true,
                supports_function_calling: true,
                supports_reasoning: true,
                supports_streaming: true,
                recommended: false,
                family: 'muse',
              },
              {
                id: 'muse-spark-1.3',
                name: 'Muse Spark 1.3',
                provider: 'meta',
                context_window: 1048576,
                max_output_tokens: null,
                supports_vision: true,
                supports_function_calling: true,
                supports_reasoning: true,
                supports_streaming: true,
                recommended: true,
                family: 'muse',
              },
            ],
          },
          last_updated: null,
        }),
      });

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="meta"
          onlineModel="muse-spark-1.3"
        />
      );

      expect(await screen.findByRole('option', { name: 'Meta Model API' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Muse Spark 1.1' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Muse Spark 1.2' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Muse Spark 1.3' })).toBeInTheDocument();
    });

    it('shows Qwen3.8 Flash privacy guidance from live catalog data', async () => {
      const performanceNote = 'The current endpoint is not OpenRouter ZDR; do not use it for confidential workloads.';
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          providers: {
            openrouter: [{
              id: 'qwen/qwen3.8-flash',
              name: 'Qwen 3.8 Flash',
              provider: 'openrouter',
              context_window: 1000000,
              max_output_tokens: 131072,
              supports_vision: true,
              supports_function_calling: true,
              supports_reasoning: true,
              supports_streaming: true,
              recommended: true,
              family: 'qwen',
              performance_note: performanceNote,
            }],
          },
          last_updated: null,
        }),
      });

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="openrouter"
          onlineModel="qwen/qwen3.8-flash"
        />
      );

      expect(await screen.findByRole('option', { name: 'Qwen 3.8 Flash' })).toBeInTheDocument();
      expect(screen.getByText(performanceNote)).toBeInTheDocument();
    });

    it('shows Tencent Hy4 Preview guidance from live OpenRouter catalog data', async () => {
      const performanceNote = 'Single Tencent FP8 preview route; enforce ZDR routing.';
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          providers: {
            openrouter: [{
              id: 'tencent/hy4-preview',
              name: 'Tencent Hy4 Preview',
              provider: 'openrouter',
              context_window: 1048576,
              max_output_tokens: 64000,
              supports_vision: false,
              supports_function_calling: true,
              supports_reasoning: true,
              supports_streaming: true,
              recommended: true,
              family: 'hy',
              performance_note: performanceNote,
            }],
          },
          last_updated: null,
        }),
      });

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="openrouter"
          onlineModel="tencent/hy4-preview"
        />
      );

      expect(await screen.findByRole('option', { name: 'Tencent Hy4 Preview' })).toBeInTheDocument();
      expect(screen.getByText(performanceNote)).toBeInTheDocument();
    });

    it('displays correct descriptions for settings', () => {
      render(<AgentConfigSection {...defaultProps} />);

      expect(screen.getByText('Select the online API provider Geist should use.')).toBeInTheDocument();
      // Description may show "Loading models..." or the actual text depending on loading state
      expect(screen.getByText(/Choose which model from the provider to use|Loading models.../)).toBeInTheDocument();
    });

    it('displays performance guidance supplied by the model API', async () => {
      const performanceNote = 'Use 4-bit weights for tolerable laptop latency.';
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          providers: {
            offline: [{
              id: 'example/future-model-3b',
              name: 'Future Model 3B',
              provider: 'offline',
              context_window: 32768,
              max_output_tokens: 4096,
              supports_vision: false,
              supports_function_calling: true,
              supports_streaming: false,
              recommended: true,
              family: 'future-family',
              performance_note: performanceNote,
            }],
          },
          last_updated: null,
        }),
      });

      render(
        <AgentConfigSection
          {...defaultProps}
          agentType="local"
          localModel="example/future-model-3b"
        />
      );

      expect(await screen.findByText(performanceNote)).toBeInTheDocument();
      expect(screen.getByText('Future Model 3B')).toBeInTheDocument();
      expect(screen.getByRole('link', { name: 'Manage local models' }))
        .toHaveAttribute('href', '/models');
    });
  });
});
