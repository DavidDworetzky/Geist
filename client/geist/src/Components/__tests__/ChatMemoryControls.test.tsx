import { fireEvent, render, screen } from '@testing-library/react';
import ChatMemoryControls from '../ChatMemoryControls';


describe('ChatMemoryControls', () => {
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
    ],
    loading: false,
    error: null,
    onMemoryEnabledChange: jest.fn(),
    onPrivateChange: jest.fn(),
    onFolderChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('exposes chat-level memory and privacy switches', () => {
    render(<ChatMemoryControls {...baseProps} />);

    const memorySwitch = screen.getByRole('switch', { name: 'Memory enabled' });
    const privateSwitch = screen.getByRole('switch', { name: 'Private' });
    expect(memorySwitch).toHaveAttribute('aria-checked', 'true');
    expect(privateSwitch).toHaveAttribute('aria-checked', 'false');
    expect(screen.getByText('Eligible for your global profile')).toBeInTheDocument();

    fireEvent.click(memorySwitch);
    fireEvent.click(privateSwitch);

    expect(baseProps.onMemoryEnabledChange).toHaveBeenCalledWith(false);
    expect(baseProps.onPrivateChange).toHaveBeenCalledWith(true);
  });

  it('assigns a private folder from the compact control', () => {
    render(<ChatMemoryControls {...baseProps} />);

    fireEvent.change(screen.getByRole('combobox', { name: 'Memory folder' }), {
      target: { value: '5' },
    });

    expect(baseProps.onFolderChange).toHaveBeenCalledWith(5);
  });

  it('clearly communicates when memory is disabled', () => {
    render(
      <ChatMemoryControls
        {...baseProps}
        settings={{
          ...baseProps.settings,
          memory_enabled: false,
          effective_scope: 'disabled',
        }}
      />,
    );

    expect(screen.getByText('Memory is off for this chat')).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: 'Private' })).toBeDisabled();
  });
});
