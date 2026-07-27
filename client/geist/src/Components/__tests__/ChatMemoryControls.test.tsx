import { fireEvent, render, screen, within } from '@testing-library/react';
import ChatMemoryControls from '../ChatMemoryControls';


describe('ChatMemoryControls', () => {
  const onScopeChange = jest.fn();
  const onManageFolders = jest.fn();
  const baseProps = {
    settings: {
      memory_enabled: true,
      memory_mode: 'public' as const,
      folder_id: null,
      effective_scope: 'public' as const,
      status: 'ready',
    },
    folders: [
      {
        folder_id: 5,
        name: 'Private research',
        color: 'violet',
        chat_count: 1,
      },
      {
        folder_id: 9,
        name: 'Product ideas',
        color: 'blue',
        chat_count: 2,
      },
    ],
    loading: false,
    error: null,
    onScopeChange,
    onManageFolders,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  function openMenu() {
    fireEvent.click(screen.getByRole('button', { name: 'Memory settings: Memory Enabled' }));
    return screen.getByRole('menu', { name: 'Memory options' });
  }

  it('offers the three memory choices, folder scopes, and folder management', () => {
    render(<ChatMemoryControls {...baseProps} />);

    const trigger = screen.getByRole('button', { name: 'Memory settings: Memory Enabled' });
    expect(trigger).toHaveAttribute('aria-haspopup', 'menu');
    expect(trigger).toHaveAttribute('aria-expanded', 'false');

    const menu = openMenu();
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(within(menu).getByRole('menuitemradio', { name: 'Memory Disabled' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitemradio', { name: 'Memory Enabled' })).toHaveAttribute('aria-checked', 'true');
    expect(within(menu).getByRole('menuitemradio', { name: 'Memory Enabled (this chat only)' })).toBeInTheDocument();
    expect(within(menu).getByRole('group', { name: 'Folder memory' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitemradio', { name: 'Private research' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitemradio', { name: 'Product ideas' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: 'Manage folders…' })).toBeInTheDocument();
    expect(screen.queryByText(/ready/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Eligible for your global profile/i)).not.toBeInTheDocument();
  });

  it.each([
    ['Memory Disabled', { kind: 'disabled' }],
    ['Memory Enabled (this chat only)', { kind: 'private' }],
  ])('changes scope through %s', (choice, expectedScope) => {
    render(<ChatMemoryControls {...baseProps} />);
    const menu = openMenu();

    fireEvent.click(within(menu).getByRole('menuitemradio', { name: new RegExp(`^${choice.replace(/[()]/g, '\\$&')}`) }));

    expect(onScopeChange).toHaveBeenCalledWith(expectedScope);
    expect(screen.queryByRole('menu', { name: 'Memory options' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Memory settings: Memory Enabled' })).toHaveFocus();
  });

  it('can return a private chat to shared memory', () => {
    render(
      <ChatMemoryControls
        {...baseProps}
        settings={{
          ...baseProps.settings,
          memory_mode: 'private',
          effective_scope: 'private',
        }}
      />,
    );

    fireEvent.click(screen.getByRole('button', {
      name: 'Memory settings: Memory Enabled (this chat only)',
    }));
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Memory Enabled' }));

    expect(onScopeChange).toHaveBeenCalledWith({ kind: 'public' });
  });

  it('selects a folder scope and labels the trigger with the active folder', () => {
    const { rerender } = render(<ChatMemoryControls {...baseProps} />);
    const menu = openMenu();
    fireEvent.click(within(menu).getByRole('menuitemradio', { name: 'Private research' }));
    expect(onScopeChange).toHaveBeenCalledWith({ kind: 'folder', folderId: 5 });

    rerender(
      <ChatMemoryControls
        {...baseProps}
        settings={{
          ...baseProps.settings,
          memory_mode: 'private',
          folder_id: 5,
          effective_scope: 'folder',
        }}
      />,
    );

    expect(screen.getByRole('button', { name: 'Memory settings: Memory: Private research' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Memory settings: Memory: Private research' }));
    expect(screen.getByRole('menuitemradio', { name: 'Private research' })).toHaveAttribute('aria-checked', 'true');
  });

  it('opens folder management and closes the dropdown', () => {
    render(<ChatMemoryControls {...baseProps} />);
    fireEvent.click(within(openMenu()).getByRole('menuitem', { name: 'Manage folders…' }));

    expect(onManageFolders).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('menu', { name: 'Memory options' })).not.toBeInTheDocument();
  });

  it('uses an icon-only but labelled trigger in compact mode', () => {
    render(<ChatMemoryControls {...baseProps} compact />);

    const trigger = screen.getByRole('button', { name: 'Memory settings: Memory Enabled' });
    expect(trigger).toHaveAttribute('title', 'Memory Enabled');
    expect(screen.queryByText('Memory Enabled')).not.toBeInTheDocument();
  });

  it('closes on Escape, restores trigger focus, and closes on outside click', () => {
    render(
      <div>
        <ChatMemoryControls {...baseProps} />
        <button type="button">Outside</button>
      </div>,
    );

    openMenu();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('menu', { name: 'Memory options' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Memory settings: Memory Enabled' })).toHaveFocus();

    openMenu();
    fireEvent.mouseDown(screen.getByRole('button', { name: 'Outside' }));
    expect(screen.queryByRole('menu', { name: 'Memory options' })).not.toBeInTheDocument();
  });

  it('opens and navigates the menu from the keyboard', () => {
    render(<ChatMemoryControls {...baseProps} />);
    const trigger = screen.getByRole('button', { name: 'Memory settings: Memory Enabled' });

    fireEvent.keyDown(trigger, { key: 'ArrowDown' });
    expect(screen.getByRole('menuitemradio', { name: 'Memory Disabled' })).toHaveFocus();

    fireEvent.keyDown(screen.getByRole('menuitemradio', { name: 'Memory Disabled' }), {
      key: 'ArrowDown',
    });
    expect(screen.getByRole('menuitemradio', { name: 'Memory Enabled' })).toHaveFocus();

    fireEvent.keyDown(screen.getByRole('menuitemradio', { name: 'Memory Enabled' }), {
      key: 'End',
    });
    expect(screen.getByRole('menuitem', { name: 'Manage folders…' })).toHaveFocus();
  });

  it('disables interaction while loading or explicitly disabled', () => {
    const { rerender } = render(<ChatMemoryControls {...baseProps} loading />);
    expect(screen.getByRole('button', { name: 'Memory settings: Memory Enabled' })).toBeDisabled();

    rerender(<ChatMemoryControls {...baseProps} disabled />);
    expect(screen.getByRole('button', { name: 'Memory settings: Memory Enabled' })).toBeDisabled();
  });

  it('reports update errors without restoring the old status badge', () => {
    render(<ChatMemoryControls {...baseProps} error="Could not update memory settings" />);

    expect(screen.getByRole('alert')).toHaveTextContent('Could not update memory settings');
    expect(screen.queryByText(/ready/i)).not.toBeInTheDocument();
  });
});
