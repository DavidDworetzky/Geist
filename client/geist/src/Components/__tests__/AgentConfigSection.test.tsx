import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import AgentConfigSection from '../AgentConfigSection';

describe('AgentConfigSection', () => {
  const defaultProps = {
    agentType: 'online',
    localModel: 'Meta-Llama-3.1-8B-Instruct',
    onlineProvider: 'openai',
    onlineModel: 'gpt-4',
    onAgentTypeChange: jest.fn(),
    onLocalModelChange: jest.fn(),
    onOnlineProviderChange: jest.fn(),
    onOnlineModelChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
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
      expect(optionValues).not.toContain('claude-3-opus');
      expect(optionValues).not.toContain('claude-3-sonnet');
    });

    it('shows only Anthropic models when Anthropic provider is selected', () => {
      render(<AgentConfigSection {...defaultProps} onlineProvider="anthropic" onlineModel="claude-3-opus" />);

      const modelSelect = screen.getByLabelText('Online Model');
      const options = modelSelect.querySelectorAll('option');

      const optionValues = Array.from(options).map((opt) => opt.getAttribute('value'));
      
      expect(optionValues).toContain('claude-3-opus');
      expect(optionValues).toContain('claude-3-sonnet');
      expect(optionValues).not.toContain('gpt-4');
      expect(optionValues).not.toContain('gpt-4-turbo');
      expect(optionValues).not.toContain('gpt-3.5-turbo');
    });

    it('shows only custom models when custom provider is selected', () => {
      render(<AgentConfigSection {...defaultProps} onlineProvider="custom" onlineModel="custom-model" />);

      const modelSelect = screen.getByLabelText('Online Model');
      const options = modelSelect.querySelectorAll('option');

      const optionValues = Array.from(options).map((opt) => opt.getAttribute('value'));
      
      expect(optionValues).toContain('custom-model');
      expect(optionValues).not.toContain('gpt-4');
      expect(optionValues).not.toContain('claude-3-opus');
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
      expect(onOnlineModelChange).toHaveBeenCalledWith('claude-3-opus');
    });

    it('resets model to first available when switching from Anthropic to OpenAI', () => {
      const onOnlineProviderChange = jest.fn();
      const onOnlineModelChange = jest.fn();

      render(
        <AgentConfigSection
          {...defaultProps}
          onlineProvider="anthropic"
          onlineModel="claude-3-opus"
          onOnlineProviderChange={onOnlineProviderChange}
          onOnlineModelChange={onOnlineModelChange}
        />
      );

      const providerSelect = screen.getByLabelText('Online Provider');
      fireEvent.change(providerSelect, { target: { value: 'openai' } });

      expect(onOnlineProviderChange).toHaveBeenCalledWith('openai');
      expect(onOnlineModelChange).toHaveBeenCalledWith('gpt-4');
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
    it('shows local model options when agent type is local', () => {
      render(<AgentConfigSection {...defaultProps} agentType="local" />);

      expect(screen.getByLabelText('Local Model')).toBeInTheDocument();
      expect(screen.queryByLabelText('Online Provider')).not.toBeInTheDocument();
      expect(screen.queryByLabelText('Online Model')).not.toBeInTheDocument();

      const modelSelect = screen.getByLabelText('Local Model');
      const options = modelSelect.querySelectorAll('option');
      const optionValues = Array.from(options).map((opt) => opt.getAttribute('value'));

      expect(optionValues).toContain('Meta-Llama-3.1-8B-Instruct');
      expect(optionValues).toContain('Meta-Llama-3.1-8B');
      expect(optionValues).toContain('Meta-Llama-3-8B-Instruct');
    });

    it('shows online provider and model options when agent type is online', () => {
      render(<AgentConfigSection {...defaultProps} agentType="online" />);

      expect(screen.queryByLabelText('Local Model')).not.toBeInTheDocument();
      expect(screen.getByLabelText('Online Provider')).toBeInTheDocument();
      expect(screen.getByLabelText('Online Model')).toBeInTheDocument();
    });

    it('calls onAgentTypeChange when agent type is changed', () => {
      const onAgentTypeChange = jest.fn();

      render(<AgentConfigSection {...defaultProps} onAgentTypeChange={onAgentTypeChange} />);

      const agentTypeSelect = screen.getByLabelText('Default Agent Type');
      fireEvent.change(agentTypeSelect, { target: { value: 'local' } });

      expect(onAgentTypeChange).toHaveBeenCalledWith('local');
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

    it('calls onLocalModelChange when a local model is selected', () => {
      const onLocalModelChange = jest.fn();

      render(
        <AgentConfigSection
          {...defaultProps}
          agentType="local"
          onLocalModelChange={onLocalModelChange}
        />
      );

      const modelSelect = screen.getByLabelText('Local Model');
      fireEvent.change(modelSelect, { target: { value: 'Meta-Llama-3.1-8B' } });

      expect(onLocalModelChange).toHaveBeenCalledWith('Meta-Llama-3.1-8B');
    });
  });

  describe('Rendering', () => {
    it('renders the component with correct title', () => {
      render(<AgentConfigSection {...defaultProps} />);

      expect(screen.getByText('Agent Configuration')).toBeInTheDocument();
    });

    it('displays all provider options', () => {
      render(<AgentConfigSection {...defaultProps} />);

      const providerSelect = screen.getByLabelText('Online Provider');
      const options = providerSelect.querySelectorAll('option');
      const optionLabels = Array.from(options).map((opt) => opt.textContent);

      expect(optionLabels).toContain('OpenAI');
      expect(optionLabels).toContain('Anthropic');
      expect(optionLabels).toContain('Custom Provider');
    });

    it('displays correct descriptions for settings', () => {
      render(<AgentConfigSection {...defaultProps} />);

      expect(screen.getByText('Choose whether to use a local or online language model by default')).toBeInTheDocument();
      expect(screen.getByText('Select your preferred online API provider')).toBeInTheDocument();
      expect(screen.getByText('Choose which model from the provider to use')).toBeInTheDocument();
    });
  });
});

