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
});
