import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AgentPermissionsSection from '../AgentPermissionsSection';

const toolsResponse = {
  tools: [
    {
      name: 'web.search',
      description: 'Search the web',
      requires_approval: false,
      side_effect: 'read',
    },
    {
      name: 'communication.email.send',
      description: 'Send an email',
      requires_approval: true,
      side_effect: 'external_write',
    },
  ],
};

describe('AgentPermissionsSection', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('loads tools and toggles the always-allow list', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => toolsResponse });

    const onChange = jest.fn();

    render(
      <AgentPermissionsSection
        agentPermissions={{ mode: 'default', always_allow: [] }}
        onChange={onChange}
      />
    );

    expect(screen.getByText(/Loading tools/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('web.search')).toBeInTheDocument();
      expect(screen.getByText('communication.email.send')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('web.search'));
    expect(onChange).toHaveBeenCalledWith({ mode: 'default', always_allow: ['web.search'] });
    expect(screen.getByRole('button', { name: 'web.search' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('changes the approval mode', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => toolsResponse });

    const onChange = jest.fn();

    render(
      <AgentPermissionsSection
        agentPermissions={{ mode: 'default', always_allow: [] }}
        onChange={onChange}
      />
    );

    await waitFor(() => screen.getByText('web.search'));

    fireEvent.change(screen.getByLabelText(/Approval Mode/i), {
      target: { value: 'auto_approve' },
    });
    expect(onChange).toHaveBeenCalledWith({ mode: 'auto_approve', always_allow: [] });
  });

  it('shows the auto-approve hint when that mode is active', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => toolsResponse });

    render(
      <AgentPermissionsSection
        agentPermissions={{ mode: 'auto_approve', always_allow: [] }}
        onChange={() => {}}
      />
    );

    await waitFor(() => screen.getByText('web.search'));
    expect(screen.getByText(/Auto-approve is on/i)).toBeInTheDocument();
    expect(screen.getByText(/MCP\/plugin tools whose definitions can change without notice/)).toBeInTheDocument();
  });

  it('removes a tool from the allowlist and supports clear all', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => toolsResponse });

    const onChange = jest.fn();

    render(
      <AgentPermissionsSection
        agentPermissions={{ mode: 'require_approval', always_allow: ['web.search'] }}
        onChange={onChange}
      />
    );

    await waitFor(() => screen.getByText('web.search'));

    expect(screen.getByRole('button', { name: 'web.search' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByText('web.search'));
    expect(onChange).toHaveBeenCalledWith({ mode: 'require_approval', always_allow: [] });

    fireEvent.click(screen.getByText(/Clear All/i));
    expect(onChange).toHaveBeenCalledWith({ mode: 'require_approval', always_allow: [] });
  });

  it('labels high-impact tools and prevents new grants for disabled tools', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => ({
      tools: toolsResponse.tools.map(tool => ({ ...tool, enabled: false }))
    }) });
    render(<AgentPermissionsSection agentPermissions={{ mode: 'default', always_allow: ['web.search'] }} onChange={() => {}} />);
    const granted = await screen.findByRole('button', { name: /web.search/ });
    expect(granted).not.toBeDisabled();
    expect(screen.getByRole('button', { name: /communication.email.send/ })).toBeDisabled();
    expect(screen.getByText(/can affect external systems/)).toBeInTheDocument();
  });

  it('surfaces unavailable saved grants and lets the user revoke them', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => toolsResponse });
    const onChange = jest.fn();
    render(<AgentPermissionsSection agentPermissions={{ mode: 'default', always_allow: ['old.tool'] }} onChange={onChange} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Remove unavailable grant: old.tool' }));
    expect(onChange).toHaveBeenCalledWith({ mode: 'default', always_allow: [] });
  });

  it.each(['rejected', 'non-ok', 'invalid-json', 'missing-tools', 'invalid-tool', 'missing-name', 'invalid-name'])('preserves grants when the catalog response is %s', async (failure) => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const fetchMock = global.fetch as jest.Mock;
    if (failure === 'rejected') {
      fetchMock.mockRejectedValueOnce(new Error('offline'));
    } else if (failure === 'non-ok') {
      fetchMock.mockResolvedValueOnce({ ok: false });
    } else if (failure === 'invalid-json') {
      fetchMock.mockResolvedValueOnce({ ok: true, json: async () => { throw new Error('invalid JSON'); } });
    } else {
      const tool = failure === 'missing-name' ? {} : failure === 'invalid-name' ? { name: 4 } : null;
      fetchMock.mockResolvedValueOnce({ ok: true, json: async () => failure === 'missing-tools' ? {} : { tools: [tool] } });
    }
    const onChange = jest.fn();
    render(<AgentPermissionsSection agentPermissions={{ mode: 'default', always_allow: ['web.search'] }} onChange={onChange} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Your saved grants are unchanged');
    expect(screen.queryByText(/Unavailable saved grants/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Remove unavailable grant/ })).not.toBeInTheDocument();
    expect(screen.queryByText('No agent tools are available.')).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('distinguishes a successfully loaded empty catalog from a failed request', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => ({ tools: [] }) });
    render(<AgentPermissionsSection agentPermissions={{ mode: 'default', always_allow: ['old.tool'] }} onChange={() => {}} />);
    expect(await screen.findByText('No agent tools are available.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove unavailable grant: old.tool' })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it.each([false, true])('prevents new dynamic grants while allowing revocation (selected=%s)', async (selected) => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => ({
      tools: [{ ...toolsResponse.tools[1], name: 'mcp.weather.forecast', allows_standing_grant: false }]
    }) });
    const onChange = jest.fn();
    render(<AgentPermissionsSection agentPermissions={{ mode: 'default', always_allow: selected ? ['mcp.weather.forecast'] : [] }} onChange={onChange} />);
    const tool = await screen.findByRole('button', { name: /mcp.weather.forecast/ });
    expect(tool).toHaveProperty('disabled', !selected);
    expect(tool.textContent).toContain(selected ? 'stored grant is not honored' : 'standing grants unavailable');
    fireEvent.click(tool);
    expect(onChange.mock.calls).toEqual(selected ? [[{ mode: 'default', always_allow: [] }]] : []);
  });
});
