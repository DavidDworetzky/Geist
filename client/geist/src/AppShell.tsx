import React, { ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
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
    name: 'Workflows',
    path: '/workflows',
    description: 'Automations',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 7.75A2.75 2.75 0 1 1 7 2.25a2.75 2.75 0 0 1 0 5.5Zm10 14A2.75 2.75 0 1 1 17 16.25a2.75 2.75 0 0 1 0 5.5ZM7 21.75A2.75 2.75 0 1 1 7 16.25a2.75 2.75 0 0 1 0 5.5ZM9.5 5h5.25A3.25 3.25 0 0 1 18 8.25v2.5A3.25 3.25 0 0 1 14.75 14H9.5v-2h5.25c.69 0 1.25-.56 1.25-1.25v-2.5C16 7.56 15.44 7 14.75 7H9.5V5Z" />
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
  {
    name: 'Models',
    path: '/models',
    description: 'Providers',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 2.5 20.25 7v10L12 21.5 3.75 17V7L12 2.5Zm0 2.28L6.26 7.9 12 11.02l5.74-3.12L12 4.78Zm-6.25 5v6.04L11 18.7v-6.04L5.75 9.78Zm7.25 8.92 5.25-2.88V9.78L13 12.66v6.04Z" />
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
  if (pathname.startsWith('/settings')) return 'Settings';
  return 'Chat';
}

function RuntimeSummary(): JSX.Element {
  const { settings, loading, error } = useUserSettings();

  if (loading) {
    return <span className="runtime-chip">Runtime loading</span>;
  }

  if (error || !settings) {
    return <span className="runtime-chip runtime-chip-warning">Settings unavailable</span>;
  }

  const mode = settings.default_agent_type || 'local';
  const model = mode === 'online' ? settings.default_online_model : settings.default_local_model;
  const provider = mode === 'online' ? settings.default_online_provider : 'local';

  return (
    <div className="runtime-summary" aria-label="Current runtime">
      <span className="runtime-chip">{mode}</span>
      <span className="runtime-chip">{provider}</span>
      <span className="runtime-model" title={model}>{model || 'No model selected'}</span>
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
            <span className="brand-subtitle">Open source host for private local LLMs</span>
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
