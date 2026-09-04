import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
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
    {
      name: 'terminal.run',
      description: 'Run a protected host command',
      requires_approval: true,
      requires_per_call_approval: true,
      side_effect: 'process',
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

    expect(await screen.findByText('web.search')).toBeInTheDocument();
    expect(screen.getByText('communication.email.send')).toBeInTheDocument();

    fireEvent.click(screen.getByText('web.search'));
    expect(onChange).toHaveBeenCalledWith({ mode: 'default', always_allow: ['web.search'] });
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

    await screen.findByText('web.search');

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

    await screen.findByText('web.search');
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

    await screen.findByText('web.search');

    fireEvent.click(screen.getByText('web.search'));
    expect(onChange).toHaveBeenCalledWith({ mode: 'require_approval', always_allow: [] });

    fireEvent.click(screen.getByText(/Clear All/i));
    expect(onChange).toHaveBeenCalledWith({ mode: 'require_approval', always_allow: [] });
  });

  it('does not allow protected terminal commands onto the standing allowlist', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => toolsResponse });
    const onChange = jest.fn();

    render(
      <AgentPermissionsSection
        agentPermissions={{ mode: 'auto_approve', always_allow: [] }}
        onChange={onChange}
      />
    );

    const terminal = await screen.findByRole('button', { name: /terminal\.run/i });
    expect(terminal).toBeDisabled();
    expect(terminal).toHaveTextContent('always asks');
    fireEvent.click(terminal);
    expect(onChange).not.toHaveBeenCalled();
  });
});
