import { useEffect, useId, useRef, useState } from 'react';
import {
  ChatMemorySettings,
  getMemoryScope,
  MemoryFolder,
  MemoryScope,
} from '../Hooks/useChatMemory';
import StagePanelIcon from './StagePanelIcon';

interface ChatMemoryControlsProps {
  settings: ChatMemorySettings;
  folders: MemoryFolder[];
  loading: boolean;
  disabled?: boolean;
  compact?: boolean;
  error: string | null;
  onScopeChange: (scope: MemoryScope) => void;
  onManageFolders: () => void;
}

interface ScopeOption {
  scope: MemoryScope;
  label: string;
  description: string;
}

const STANDARD_OPTIONS: ScopeOption[] = [
  {
    scope: { kind: 'disabled' },
    label: 'Memory Disabled',
    description: 'Do not save or recall memory.',
  },
  {
    scope: { kind: 'public' },
    label: 'Memory Enabled',
    description: 'Use memory across chats.',
  },
  {
    scope: { kind: 'private' },
    label: 'Memory Enabled (this chat only)',
    description: 'Keep memory within this chat.',
  },
];

function scopesMatch(left: MemoryScope, right: MemoryScope): boolean {
  if (left.kind !== right.kind) return false;
  return left.kind !== 'folder'
    || (right.kind === 'folder' && left.folderId === right.folderId);
}

function triggerLabel(scope: MemoryScope, folders: MemoryFolder[]): string {
  if (scope.kind === 'folder') {
    const folder = folders.find(item => item.folder_id === scope.folderId);
    return `Memory: ${folder?.name ?? 'Folder'}`;
  }
  if (scope.kind === 'disabled') return 'Memory Disabled';
  if (scope.kind === 'private') return 'Memory Enabled (this chat only)';
  return 'Memory Enabled';
}

export default function ChatMemoryControls({
  settings,
  folders,
  loading,
  disabled = false,
  compact = false,
  error,
  onScopeChange,
  onManageFolders,
}: ChatMemoryControlsProps) {
  const [open, setOpen] = useState(false);
  const menuId = useId();
  const folderHeadingId = `${menuId}-folders`;
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const openFocusRef = useRef<'selected' | 'first' | 'last'>('selected');
  const currentScope = getMemoryScope(settings);
  const label = triggerLabel(currentScope, folders);
  const interactionDisabled = disabled || loading;

  const closeMenu = (restoreFocus = false) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  };

  useEffect(() => {
    if (!open) return undefined;

    const handleOutsideClick = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
      }
    };

    document.addEventListener('mousedown', handleOutsideClick);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [open]);

  useEffect(() => {
    if (interactionDisabled) setOpen(false);
  }, [interactionDisabled]);

  useEffect(() => {
    if (!open) return;
    const menu = rootRef.current?.querySelector<HTMLElement>('[role="menu"]');
    if (!menu) return;
    const items = Array.from(
      menu.querySelectorAll<HTMLButtonElement>(
        '[role="menuitemradio"]:not(:disabled), [role="menuitem"]:not(:disabled)',
      ),
    );
    const target = openFocusRef.current === 'last'
      ? items[items.length - 1]
      : openFocusRef.current === 'first'
        ? items[0]
        : menu.querySelector<HTMLButtonElement>('[role="menuitemradio"][aria-checked="true"]')
          ?? items[0];
    target?.focus();
  }, [open]);

  const selectScope = (scope: MemoryScope) => {
    if (!scopesMatch(scope, currentScope)) onScopeChange(scope);
    closeMenu(true);
  };

  const moveMenuFocus = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>(
        '[role="menuitemradio"]:not(:disabled), [role="menuitem"]:not(:disabled)',
      ),
    );
    if (items.length === 0) return;
    const currentIndex = items.indexOf(document.activeElement as HTMLButtonElement);
    let nextIndex = currentIndex;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = items.length - 1;
    if (event.key === 'ArrowDown') nextIndex = (currentIndex + 1 + items.length) % items.length;
    if (event.key === 'ArrowUp') {
      nextIndex = (currentIndex - 1 + items.length) % items.length;
    }
    items[nextIndex].focus();
  };

  const renderScopeOption = (option: ScopeOption) => {
    const selected = scopesMatch(option.scope, currentScope);
    const optionKey = option.scope.kind === 'folder'
      ? `folder-${option.scope.folderId}`
      : option.scope.kind;
    return (
      <button
        key={optionKey}
        type="button"
        className={`chat-memory-option${selected ? ' is-selected' : ''}`}
        role="menuitemradio"
        aria-checked={selected}
        aria-label={option.label}
        onClick={() => selectScope(option.scope)}
      >
        <span className="chat-memory-option-marker" aria-hidden="true">
          {selected ? '✓' : ''}
        </span>
        <span className="chat-memory-option-copy">
          <strong>{option.label}</strong>
          <small>{option.description}</small>
        </span>
      </button>
    );
  };

  return (
    <div
      ref={rootRef}
      className={`chat-memory-scope${compact ? ' is-compact' : ''}`}
      aria-busy={loading || undefined}
    >
      <button
        ref={triggerRef}
        type="button"
        className="chat-memory-trigger"
        aria-label={`Memory settings: ${label}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={menuId}
        title={compact ? label : undefined}
        disabled={interactionDisabled}
        onClick={() => {
          openFocusRef.current = 'selected';
          setOpen(value => !value);
        }}
        onKeyDown={event => {
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            openFocusRef.current = event.key === 'ArrowDown' ? 'first' : 'last';
            setOpen(true);
          }
        }}
      >
        <StagePanelIcon name="memory" />
        {!compact && <span className="chat-memory-trigger-label">{label}</span>}
        <StagePanelIcon name="chevron" />
      </button>

      {open && (
        <div
          className="chat-memory-menu"
          id={menuId}
          role="menu"
          aria-label="Memory options"
          onKeyDown={event => {
            if (event.key === 'Tab') setOpen(false);
            moveMenuFocus(event);
          }}
        >
          <div className="chat-memory-menu-section">
            {STANDARD_OPTIONS.map(renderScopeOption)}
          </div>

          <div
            className="chat-memory-menu-section"
            role="group"
            aria-labelledby={folderHeadingId}
          >
            <span id={folderHeadingId} className="chat-memory-menu-heading">
              Folder memory
            </span>
            {folders.map(folder => renderScopeOption({
              scope: { kind: 'folder', folderId: folder.folder_id },
              label: folder.name,
              description: `Share memory with chats in ${folder.name}.`,
            }))}
          </div>

          <button
            type="button"
            className="chat-memory-manage"
            role="menuitem"
            onClick={() => {
              closeMenu(true);
              onManageFolders();
            }}
          >
            Manage folders…
          </button>
        </div>
      )}

      {error && <span className="chat-memory-error" role="alert">{error}</span>}
    </div>
  );
}
