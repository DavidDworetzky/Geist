import React, { ReactNode, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import useLocalArtifacts, {
  installProgress,
  isArtifactInstalling,
  LocalArtifact,
} from './Hooks/useLocalArtifacts';
import useUserSettings from './Hooks/useUserSettings';
import { useBranding } from './branding';

interface ShellProps {
  children: ReactNode;
}

interface NavItem {
  name: string;
  path: string;
  description: string;
  icon: JSX.Element;
}

const navItems: NavItem[] = [
  {
    name: 'Chat',
    path: '/chat',
    description: 'Conversations',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4.5 5.5A3.5 3.5 0 0 1 8 2h8a3.5 3.5 0 0 1 3.5 3.5v6A3.5 3.5 0 0 1 16 15h-3.4l-4.1 4.1A1.2 1.2 0 0 1 6.5 18v-3.1a3.5 3.5 0 0 1-2-3.2v-6.2Z" />
      </svg>
    ),
  },
  {
    name: 'Models',
    path: '/models',
    description: 'Local and online',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 2.5 20.25 7v10L12 21.5 3.75 17V7L12 2.5Zm0 2.28L6.26 7.9 12 11.02l5.74-3.12L12 4.78Zm-6.25 5v6.04L11 18.7v-6.04L5.75 9.78Zm7.25 8.92 5.25-2.88V9.78L13 12.66v6.04Z" />
      </svg>
    ),
  },
  {
    name: 'Schedules',
    path: '/schedules',
    description: 'Recurring prompts',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 2h2v2h6V2h2v2h1.5A2.5 2.5 0 0 1 21 6.5v13a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 19.5v-13A2.5 2.5 0 0 1 5.5 4H7V2Zm12 8H5v9.5c0 .28.22.5.5.5h13a.5.5 0 0 0 .5-.5V10Zm-7 2a3 3 0 1 1 0 6 3 3 0 0 1 0-6Zm0 1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM5.5 6a.5.5 0 0 0-.5.5V8h14V6.5a.5.5 0 0 0-.5-.5h-13Z" />
      </svg>
    ),
  },
  {
    name: 'Settings',
    path: '/settings',
    description: 'Defaults',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M10.6 2h2.8l.55 2.37c.58.18 1.13.41 1.64.69l2.07-1.29 1.98 1.98-1.29 2.07c.28.51.51 1.06.69 1.64L22 10.6v2.8l-2.37.55a7.94 7.94 0 0 1-.69 1.64l1.29 2.07-1.98 1.98-2.07-1.29c-.51.28-1.06.51-1.64.69L13.4 22h-2.8l-.55-2.37a7.94 7.94 0 0 1-1.64-.69l-2.07 1.29-1.98-1.98 1.29-2.07a7.94 7.94 0 0 1-.69-1.64L2 13.4v-2.8l2.37-.55c.18-.58.41-1.13.69-1.64L3.77 6.34l1.98-1.98 2.07 1.29c.51-.28 1.06-.51 1.64-.69L10.6 2Zm1.4 6.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z" />
      </svg>
    ),
  },
  {
    name: 'Tools',
    path: '/tools',
    description: 'Tools and MCPs',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8.5 3a3.5 3.5 0 1 0 0 7h1v4h-1a3.5 3.5 0 1 0 3.35 4.5h2.3A3.5 3.5 0 1 0 17.5 14h-1v-4h1A3.5 3.5 0 1 0 14.15 5.5h-2.3A3.5 3.5 0 0 0 8.5 3Zm0 2a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3Zm7 2.5h-3v9h3v-1h2a1.5 1.5 0 1 1-1.5 1.5v-1h-4v1a1.5 1.5 0 1 1-1.5-1.5h1v-9h-1v1A1.5 1.5 0 1 1 12 6h4v-1a1.5 1.5 0 1 1 1.5 1.5h-2v1Z" />
      </svg>
    ),
  },
  {
    name: 'Files',
    path: '/files',
    description: 'Local context',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 3.75C5 2.78 5.78 2 6.75 2h6.1c.46 0 .91.18 1.24.51l4.4 4.4c.33.33.51.78.51 1.24v12.1c0 .97-.78 1.75-1.75 1.75H6.75C5.78 22 5 21.22 5 20.25V3.75Zm8 0V8h4.25L13 3.75Z" />
      </svg>
    ),
  },
];

function BrandMark(): JSX.Element {
  const branding = useBranding();

  if (branding.logoUrl) {
    return <img src={branding.logoUrl} alt="" className="brand-mark-image" />;
  }

  return (
    <svg className="brand-mark-svg" viewBox="0 0 48 48" aria-hidden="true">
      <path d="M24 3.5 41.75 13.75v20.5L24 44.5 6.25 34.25v-20.5L24 3.5Z" />
      <path d="M24 12.5 34 18.25v11.5L24 35.5l-10-5.75v-11.5L24 12.5Z" />
      <path d="M24 18.5 29 21.35v5.3l-5 2.85-5-2.85v-5.3L24 18.5Z" />
    </svg>
  );
}

function pageTitle(pathname: string): string {
  if (pathname.startsWith('/workflows')) return 'Workflows';
  if (pathname.startsWith('/files')) return 'Files';
  if (pathname.startsWith('/models')) return 'Models';
  if (pathname.startsWith('/schedules')) return 'Schedules';
  if (pathname.startsWith('/tools')) return 'Tools';
  if (pathname.startsWith('/settings')) return 'Settings';
  return 'Chat';
}

function RuntimeSummary(): JSX.Element {
  const { settings, loading } = useUserSettings();
  const mode = settings?.default_agent_type || 'local';
  const {
    artifacts,
    loading: artifactsLoading,
    loaded: artifactsLoaded,
    error: artifactsError,
    refreshLocalArtifacts,
    activateArtifact,
    downloadArtifact,
  } = useLocalArtifacts({
    enabled: mode === 'local',
    pollWhileBusy: true,
  });
  const [pendingArtifactId, setPendingArtifactId] = useState<string | null>(null);
  const [savingModel, setSavingModel] = useState(false);
  const [modelSaveError, setModelSaveError] = useState<string | null>(null);
  const compatibleArtifacts = artifacts.filter(artifact => artifact.supported !== false);

  const model = mode === 'online'
    ? settings?.default_online_model
    : settings?.default_local_model;
  const configuredArtifact = settings
    ? compatibleArtifacts.find(
      artifact => artifact.id === settings.default_local_artifact_id,
    ) ?? compatibleArtifacts.find(
      artifact => artifact.model_id === settings.default_local_model,
    )
    : undefined;
  const pendingArtifact = compatibleArtifacts.find(artifact => artifact.id === pendingArtifactId);
  const selectedArtifact = pendingArtifact ?? configuredArtifact;
  const installingArtifact = isArtifactInstalling(configuredArtifact)
    ? configuredArtifact
    : undefined;
  const anyArtifactInstalling = compatibleArtifacts.some(isArtifactInstalling);
  const installState = installProgress(installingArtifact);
  const selectionStartsInstall = Boolean(
    savingModel && pendingArtifact && pendingArtifact.status !== 'installed',
  );
  const catalogLoading = mode === 'local' && artifactsLoading && !artifactsLoaded;
  const modelControlBusy = catalogLoading || savingModel;
  const selectedArtifactId = pendingArtifactId ?? selectedArtifact?.id ?? '';
  const noCompatibleModels = artifactsLoaded && compatibleArtifacts.length === 0;

  if (loading) {
    return <span className="runtime-chip">Loading…</span>;
  }

  if (!settings) {
    return <span className="runtime-chip runtime-chip-warning">Settings unavailable</span>;
  }

  const handleLocalModelChange = async (event: React.ChangeEvent<HTMLSelectElement>) => {
    const nextArtifact = compatibleArtifacts.find(artifact => artifact.id === event.target.value);
    if (!nextArtifact) return;

    setPendingArtifactId(nextArtifact.id);
    setSavingModel(true);
    setModelSaveError(null);

    try {
      if (nextArtifact.status !== 'installed' && !isArtifactInstalling(nextArtifact)) {
        await downloadArtifact(nextArtifact);
      }
      await activateArtifact(nextArtifact);
    } catch (error) {
      setModelSaveError(error instanceof Error ? error.message : 'Could not select model.');
    } finally {
      setPendingArtifactId(null);
      setSavingModel(false);
    }
  };

  const retryInstall = async (artifact: LocalArtifact) => {
    setModelSaveError(null);
    try {
      await downloadArtifact(artifact);
    } catch (error) {
      setModelSaveError(error instanceof Error ? error.message : 'Could not install model.');
    }
  };

  return (
    <div className="runtime-summary" aria-label="Current runtime">
      <span className="runtime-chip">{mode}</span>
      {mode === 'online' && (
        <span className="runtime-chip">{settings.default_online_provider}</span>
      )}
      {mode === 'local' ? (
        <div className="runtime-model-control">
          <select
            aria-busy={modelControlBusy || anyArtifactInstalling || undefined}
            aria-label="Local model"
            aria-invalid={modelSaveError ? 'true' : undefined}
            className="runtime-model runtime-model-select"
            disabled={modelControlBusy || anyArtifactInstalling || noCompatibleModels}
            onChange={handleLocalModelChange}
            title={selectedArtifact?.display_name ?? model ?? 'No model selected'}
            value={selectedArtifactId}
          >
            {!selectedArtifact && (
              <option value="" disabled>
                {catalogLoading
                  ? 'Loading model catalogue…'
                  : noCompatibleModels ? 'No local models' : 'Select model'}
              </option>
            )}
            {compatibleArtifacts.map((artifact) => (
              <option key={artifact.id} value={artifact.id}>
                {artifact.display_name}
              </option>
            ))}
          </select>
          {modelControlBusy && (
            <span className="runtime-model-loading-chip" role="status">
              <span className="runtime-model-spinner" aria-hidden="true" />
              {catalogLoading ? 'Loading models' : selectionStartsInstall ? 'Installing…' : 'Selecting…'}
            </span>
          )}
          {!modelControlBusy && installingArtifact && (
            <span className="runtime-model-loading-chip" role="status">
              <span className="runtime-model-spinner" aria-hidden="true" />
              {installState.label}
              {installState.percent !== null && (
                <progress
                  aria-label={`${installingArtifact.display_name} installation progress`}
                  max="100"
                  value={installState.percent}
                />
              )}
            </span>
          )}
        </div>
      ) : (
        <span className="runtime-model" title={model}>{model || 'No model selected'}</span>
      )}
      {mode === 'local' && !modelControlBusy && selectedArtifact
        && !installingArtifact && selectedArtifact.status !== 'installed' && (
        <>
          <span
            className={selectedArtifact.status === 'failed'
              ? 'runtime-model-error'
              : 'runtime-model-state'}
            role="status"
          >
            {selectedArtifact.status === 'failed' ? 'Install failed' : 'Not installed'}
          </span>
          {(selectedArtifact.status === 'failed' || modelSaveError) && (
            <button
              className="button button-secondary button-small"
              onClick={() => void retryInstall(selectedArtifact)}
              type="button"
            >
              Retry
            </button>
          )}
        </>
      )}
      {mode === 'local' && (modelSaveError || selectedArtifact?.error) && (
        <span
          className="runtime-model-error"
          role="alert"
          title={modelSaveError ?? selectedArtifact?.error ?? undefined}
        >
          {modelSaveError ?? selectedArtifact?.error}
        </span>
      )}
      {mode === 'local' && artifactsError && (
        <>
          <span className="runtime-model-error" role="alert">Models unavailable</span>
          <button
            className="button button-secondary button-small"
            onClick={() => void refreshLocalArtifacts()}
            type="button"
          >
            Retry
          </button>
        </>
      )}
    </div>
  );
}

export default function AppShell({ children }: ShellProps): JSX.Element {
  const location = useLocation();
  const branding = useBranding();
  const productName = branding.productName ?? 'Geist';

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary">
        <NavLink to="/chat" className="brand-lockup" aria-label={`${productName} home`}>
          <BrandMark />
          <span className="brand-text">
            <span className="brand-name">{productName}</span>
            <span className="brand-subtitle">Private Local AI workbench</span>
          </span>
        </NavLink>

        <nav className="primary-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `primary-nav-link${isActive ? ' active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>
                <span className="nav-label">{item.name}</span>
                <span className="nav-description">{item.description}</span>
              </span>
            </NavLink>
          ))}
        </nav>

      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="topbar-eyebrow">Runtime</p>
            <h1>{pageTitle(location.pathname)}</h1>
          </div>
          <RuntimeSummary />
        </header>

        <main className="workspace-content">
          {children}
        </main>

        <footer className="statusbar">
          <span>Ready</span>
          <span>Local workspace</span>
        </footer>
      </section>
    </div>
  );
}
