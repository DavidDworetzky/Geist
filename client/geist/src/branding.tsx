import React, { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';

export type GeistThemeToken =
  | '--geist-brand-name'
  | '--geist-color-bg'
  | '--geist-color-surface'
  | '--geist-color-surface-strong'
  | '--geist-color-surface-muted'
  | '--geist-color-border'
  | '--geist-color-border-strong'
  | '--geist-color-text'
  | '--geist-color-text-muted'
  | '--geist-color-accent'
  | '--geist-color-accent-strong'
  | '--geist-color-success'
  | '--geist-color-warning'
  | '--geist-color-danger'
  | '--geist-radius-sm'
  | '--geist-radius-md'
  | '--geist-radius-lg'
  | '--geist-shadow-panel';

export interface GeistBranding {
  productName?: string;
  logoUrl?: string;
  faviconUrl?: string;
  titleTemplate?: string;
  theme?: Partial<Record<GeistThemeToken, string>>;
}

declare global {
  interface Window {
    __GEIST_BRANDING__?: GeistBranding;
  }
}

const BRANDING_UPDATED_EVENT = 'geist-branding-updated';

const neutralTheme: Record<GeistThemeToken, string> = {
  '--geist-brand-name': '"Geist"',
  '--geist-color-bg': '#101114',
  '--geist-color-surface': '#181a1f',
  '--geist-color-surface-strong': '#20232a',
  '--geist-color-surface-muted': '#15171c',
  '--geist-color-border': '#2b2f38',
  '--geist-color-border-strong': '#3a404c',
  '--geist-color-text': '#f4f5f7',
  '--geist-color-text-muted': '#9aa3af',
  '--geist-color-accent': '#7c8cff',
  '--geist-color-accent-strong': '#9aa6ff',
  '--geist-color-success': '#34d399',
  '--geist-color-warning': '#fbbf24',
  '--geist-color-danger': '#f87171',
  '--geist-radius-sm': '4px',
  '--geist-radius-md': '6px',
  '--geist-radius-lg': '8px',
  '--geist-shadow-panel': '0 12px 30px rgba(0, 0, 0, 0.22)',
};

export const defaultBranding: Required<Pick<GeistBranding, 'productName' | 'titleTemplate'>> &
  GeistBranding = {
  productName: 'Geist',
  titleTemplate: '%s',
  theme: neutralTheme,
};

const BrandingContext = createContext<GeistBranding>(defaultBranding);

function mergeBranding(next?: GeistBranding | null): GeistBranding {
  return {
    ...defaultBranding,
    ...next,
    theme: {
      ...neutralTheme,
      ...(next?.theme ?? {}),
    },
  };
}

function applyFavicon(faviconUrl?: string): void {
  if (!faviconUrl) {
    return;
  }

  let link = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
  if (!link) {
    link = document.createElement('link');
    link.rel = 'icon';
    document.head.appendChild(link);
  }
  link.href = faviconUrl;
}

function applyBranding(branding: GeistBranding): void {
  const mergedBranding = mergeBranding(branding);
  const theme = mergedBranding.theme ?? neutralTheme;
  Object.entries(theme).forEach(([token, value]) => {
    document.documentElement.style.setProperty(token, value);
  });

  const productName = mergedBranding.productName ?? defaultBranding.productName;
  const titleTemplate = mergedBranding.titleTemplate ?? defaultBranding.titleTemplate;
  document.title = titleTemplate.includes('%s')
    ? titleTemplate.replace('%s', productName)
    : titleTemplate;

  applyFavicon(mergedBranding.faviconUrl);
}

async function loadHostBranding(): Promise<GeistBranding | null> {
  if (window.__GEIST_BRANDING__) {
    return window.__GEIST_BRANDING__;
  }

  try {
    const response = await fetch('/branding.json', {
      cache: 'no-store',
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as GeistBranding;
  } catch {
    return null;
  }
}

export function BrandingProvider({ children }: { children: ReactNode }): JSX.Element {
  const [branding, setBranding] = useState<GeistBranding>(() =>
    mergeBranding(typeof window === 'undefined' ? null : window.__GEIST_BRANDING__)
  );

  useEffect(() => {
    let cancelled = false;
    let runtimeBrandingReceived = Boolean(window.__GEIST_BRANDING__);

    const setNextBranding = (next?: GeistBranding | null) => {
      if (!cancelled) {
        setBranding(mergeBranding(next));
      }
    };

    const handleRuntimeBranding = (event: Event) => {
      const runtimeBranding = (event as CustomEvent<GeistBranding>).detail ?? window.__GEIST_BRANDING__;
      if (runtimeBranding) {
        runtimeBrandingReceived = true;
        setNextBranding(runtimeBranding);
      }
    };

    window.addEventListener(BRANDING_UPDATED_EVENT, handleRuntimeBranding);

    loadHostBranding().then((hostBranding) => {
      if (!runtimeBrandingReceived && hostBranding) {
        setNextBranding(hostBranding);
      }
    });

    return () => {
      cancelled = true;
      window.removeEventListener(BRANDING_UPDATED_EVENT, handleRuntimeBranding);
    };
  }, []);

  useEffect(() => {
    applyBranding(branding);
  }, [branding]);

  const value = useMemo(() => branding, [branding]);

  return <BrandingContext.Provider value={value}>{children}</BrandingContext.Provider>;
}

export function useBranding(): GeistBranding {
  return useContext(BrandingContext);
}
