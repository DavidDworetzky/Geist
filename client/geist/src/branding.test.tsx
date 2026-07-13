import React from 'react';
import { act, render, screen } from '@testing-library/react';
import { BrandingProvider, GeistBranding, useBranding } from './branding';

function BrandingProbe(): JSX.Element {
  const branding = useBranding();
  return <span data-testid="product-name">{branding.productName}</span>;
}

describe('BrandingProvider', () => {
  const originalFetch = window.fetch;

  afterEach(() => {
    delete window.__GEIST_BRANDING__;
    if (originalFetch) {
      window.fetch = originalFetch;
    } else {
      Object.defineProperty(window, 'fetch', {
        configurable: true,
        value: undefined,
        writable: true,
      });
    }
    document.documentElement.removeAttribute('style');
  });

  it('keeps runtime branding when a late host lookup returns no branding', async () => {
    let resolveFetch!: (response: Response) => void;
    const fetchPromise = new Promise<Response>((resolve) => {
      resolveFetch = resolve;
    });
    window.fetch = jest.fn(() => fetchPromise);

    render(
      <BrandingProvider>
        <BrandingProbe />
      </BrandingProvider>
    );

    expect(screen.getByTestId('product-name')).toHaveTextContent('Geist');

    const pitchblendBranding: GeistBranding = {
      productName: 'pitchblend AI',
      theme: {
        '--geist-color-bg': '#1a0b2e',
        '--geist-color-accent': '#a78bfa',
      },
    };

    act(() => {
      window.__GEIST_BRANDING__ = pitchblendBranding;
      window.dispatchEvent(new CustomEvent('geist-branding-updated', { detail: pitchblendBranding }));
    });

    expect(screen.getByTestId('product-name')).toHaveTextContent('pitchblend AI');

    await act(async () => {
      resolveFetch({ ok: false } as Response);
      await fetchPromise;
    });

    expect(screen.getByTestId('product-name')).toHaveTextContent('pitchblend AI');
    expect(document.documentElement.style.getPropertyValue('--geist-color-bg')).toBe('#1a0b2e');
  });

  it('uses runtime branding on the first render when Electron injected it before mount', () => {
    window.__GEIST_BRANDING__ = {
      productName: 'pitchblend AI',
      theme: {
        '--geist-color-bg': '#1a0b2e',
      },
    };
    window.fetch = jest.fn();

    render(
      <BrandingProvider>
        <BrandingProbe />
      </BrandingProvider>
    );

    expect(screen.getByTestId('product-name')).toHaveTextContent('pitchblend AI');
    expect(window.fetch).not.toHaveBeenCalled();
  });
});
