import { MemoryFolder, ChatMemorySettings } from '../Hooks/useChatMemory';

interface ChatMemoryControlsProps {
  settings: ChatMemorySettings;
  folders: MemoryFolder[];
  loading: boolean;
  error: string | null;
  onMemoryEnabledChange: (enabled: boolean) => void;
  onPrivateChange: (isPrivate: boolean) => void;
  onFolderChange: (folderId: number | null) => void;
}

function MemorySwitch({
  label,
  checked,
  disabled = false,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      className={`chat-memory-switch${checked ? ' is-on' : ''}`}
      role="switch"
      aria-label={label}
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span className="chat-memory-switch-track">
        <span className="chat-memory-switch-thumb" />
      </span>
      <span>{label}</span>
    </button>
  );
}

export default function ChatMemoryControls({
  settings,
  folders,
  loading,
  error,
  onMemoryEnabledChange,
  onPrivateChange,
  onFolderChange,
}: ChatMemoryControlsProps) {
  const isPrivate = settings.memory_mode === 'private' || settings.folder_id !== null;
  const scopeLabel = !settings.memory_enabled
    ? 'Memory is off for this chat'
    : settings.folder_id !== null
      ? 'Stored only in this private folder'
      : isPrivate
        ? 'Stored only in this private chat'
        : 'Eligible for your global profile';

  return (
    <section className="chat-memory-controls" aria-label="Chat memory settings">
      <div className="chat-memory-status">
        <span className={`chat-memory-orb${settings.memory_enabled ? ' is-active' : ''}`} />
        <span>
          <strong>
            Memory
            {settings.status && (
              <em className={`chat-memory-state state-${settings.status}`}>
                {settings.status.replace(/_/g, ' ')}
              </em>
            )}
          </strong>
          <small>{scopeLabel}</small>
        </span>
      </div>
      <div className="chat-memory-actions">
        <MemorySwitch
          label="Memory enabled"
          checked={settings.memory_enabled}
          disabled={loading}
          onChange={onMemoryEnabledChange}
        />
        <MemorySwitch
          label="Private"
          checked={isPrivate}
          disabled={loading || !settings.memory_enabled}
          onChange={onPrivateChange}
        />
        <label className="chat-memory-folder-select">
          <span>Folder</span>
          <select
            aria-label="Memory folder"
            value={settings.folder_id ?? ''}
            disabled={loading || !settings.memory_enabled}
            onChange={event => onFolderChange(event.target.value ? Number(event.target.value) : null)}
          >
            <option value="">Unfiled</option>
            {folders.map(folder => (
              <option key={folder.folder_id} value={folder.folder_id}>{folder.name}</option>
            ))}
          </select>
        </label>
      </div>
      {error && <span className="chat-memory-error" role="alert">{error}</span>}
    </section>
  );
}
