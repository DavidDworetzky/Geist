import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import {
  createGeistPluginRuntime,
  GEIST_PLUGIN_API_VERSION,
  GEIST_PLUGIN_RUNTIME_READY_EVENT,
  GeistPluginContextV1,
  GeistPluginV1,
  useHostDevelopmentEnabled,
} from './runtime';

/* eslint-disable testing-library/no-node-access -- DOM ownership and cleanup are the API under test. */

function pluginDefinition(
  overrides: Partial<GeistPluginV1> = {}
): GeistPluginV1 {
  return {
    apiVersion: GEIST_PLUGIN_API_VERSION,
    id: 'example.plugin',
    name: 'Example Plugin',
    provider: 'Example Provider',
    version: '1.0.0',
    activate: () => undefined,
    ...overrides,
  };
}

describe('Geist plugin runtime', () => {
  afterEach(() => {
    delete window.__GEIST_PLUGIN_RUNTIME_V1__;
    delete window.__GEIST_HOST_DEVELOPMENT__;
    delete (window as unknown as { trustedTypes?: unknown }).trustedTypes;
    jest.restoreAllMocks();
  });

  it('installs API v1 globally and announces readiness', () => {
    const installation = createGeistPluginRuntime();
    const listener = jest.fn();
    window.addEventListener(GEIST_PLUGIN_RUNTIME_READY_EVENT, listener);

    try {
      installation.install();

      expect(window.__GEIST_PLUGIN_RUNTIME_V1__).toBe(installation.runtime);
      expect(installation.runtime.apiVersion).toBe(1);
      expect(listener).toHaveBeenCalledTimes(1);
      expect((listener.mock.calls[0][0] as CustomEvent).detail).toEqual({ apiVersion: 1 });
    } finally {
      window.removeEventListener(GEIST_PLUGIN_RUNTIME_READY_EVENT, listener);
    }
  });

  it('exposes the exact approved V1 definition and activation context', () => {
    const installation = createGeistPluginRuntime();
    let activationContext: GeistPluginContextV1 | undefined;
    const definition = pluginDefinition({
      activate: (context) => {
        activationContext = context;
      },
    });

    expect(Object.keys(definition).sort()).toEqual([
      'activate',
      'apiVersion',
      'id',
      'name',
      'provider',
      'version',
    ]);
    installation.runtime.register(definition);

    expect(Object.keys(activationContext ?? {}).sort()).toEqual([
      'React',
      'document',
      'mountDom',
      'mountReact',
      'observeTargets',
    ]);
    expect(activationContext?.React).toBe(React);
    expect(activationContext?.document).toBe(document);
    expect('window' in (activationContext ?? {})).toBe(false);
  });

  it('mounts shared-React content before the Host boots and owns its cleanup', async () => {
    const installation = createGeistPluginRuntime();
    const target = document.createElement('div');
    target.id = 'plugin-target';
    document.body.appendChild(target);
    const deactivate = jest.fn();

    const unregister = installation.runtime.register(pluginDefinition({
      activate: ({ React: HostReact, mountReact }) => {
        mountReact(target, HostReact.createElement('button', null, 'Host control'));
        return deactivate;
      },
    }));

    render(<installation.Host />);

    expect(await screen.findByRole('button', { name: 'Host control' })).toBeInTheDocument();
    expect(target.querySelector(':scope > geist-plugin-surface')).not.toBeNull();
    expect(installation.runtime.getDiagnostics()).toEqual([
      expect.objectContaining({
        id: 'example.plugin',
        name: 'Example Plugin',
        provider: 'Example Provider',
        status: 'active',
        contributions: [
          expect.objectContaining({ kind: 'react', mounted: true }),
        ],
      }),
    ]);

    act(() => unregister());
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Host control' })).not.toBeInTheDocument();
    });
    expect(target.querySelector('geist-plugin-surface')).toBeNull();
    expect(deactivate).toHaveBeenCalledTimes(1);
  });

  it('mounts Nodes, DocumentFragments, and TrustedHTML while rejecting plain strings', () => {
    const installation = createGeistPluginRuntime();
    const target = document.createElement('div');
    document.body.appendChild(target);
    const trustedHtml = {
      toString: () => '<strong>Trusted host markup</strong>',
    } as unknown as TrustedHTML;
    Object.defineProperty(window, 'trustedTypes', {
      configurable: true,
      value: { isHTML: (value: unknown) => value === trustedHtml },
    });

    const unregister = installation.runtime.register(pluginDefinition({
      activate: ({ document: pluginDocument, mountDom }) => {
        const node = pluginDocument.createElement('span');
        node.textContent = 'Node control';
        mountDom(target, node);

        const fragment = pluginDocument.createDocumentFragment();
        const fragmentNode = pluginDocument.createElement('em');
        fragmentNode.textContent = 'Fragment control';
        fragment.appendChild(fragmentNode);
        mountDom(target, fragment);
        mountDom(target, trustedHtml);
      },
    }));

    expect(target).toHaveTextContent('Node control');
    expect(target).toHaveTextContent('Fragment control');
    expect(target).toHaveTextContent('Trusted host markup');

    act(() => unregister());
    expect(target).toBeEmptyDOMElement();

    const unsafe = createGeistPluginRuntime();
    unsafe.runtime.register(pluginDefinition({
      id: 'unsafe.plugin',
      activate: ({ mountDom }) => {
        mountDom(target, '<em>Unsafe string</em>' as unknown as TrustedHTML);
      },
    }));
    expect(unsafe.runtime.getDiagnostics()[0]).toEqual(expect.objectContaining({
      status: 'error',
      error: expect.stringMatching(/Node, DocumentFragment, or TrustedHTML/u),
    }));
    expect(target).not.toHaveTextContent('Unsafe string');
  });

  it('mounts both shared-React and DOM content into ShadowRoot targets', async () => {
    const installation = createGeistPluginRuntime();
    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'open' });
    document.body.appendChild(host);

    const unregister = installation.runtime.register(pluginDefinition({
      activate: ({ React: HostReact, document: pluginDocument, mountDom, mountReact }) => {
        mountReact(shadow, HostReact.createElement('span', null, 'Shadow React'));
        const node = pluginDocument.createElement('span');
        node.textContent = 'Shadow DOM';
        mountDom(shadow, node);
      },
    }));
    render(<installation.Host />);

    await waitFor(() => expect(shadow).toHaveTextContent('Shadow React'));
    expect(shadow).toHaveTextContent('Shadow DOM');

    act(() => unregister());
    expect(shadow.childNodes).toHaveLength(0);
  });

  it('observes late and replacement targets and runs per-target cleanup', async () => {
    const installation = createGeistPluginRuntime();
    const callback = jest.fn();
    const targetCleanup = jest.fn();

    const unregister = installation.runtime.register(pluginDefinition({
      activate: ({ React: HostReact, mountReact, observeTargets }) => {
        let currentTarget: Element | undefined;
        let unmount = (): void => undefined;
        const clearTarget = (): void => {
          if (currentTarget) {
            targetCleanup(currentTarget);
          }
          currentTarget = undefined;
          unmount();
          unmount = () => undefined;
        };
        const stopObserving = observeTargets('#late-target', (targets) => {
          callback(targets);
          clearTarget();
          currentTarget = targets[0];
          if (currentTarget) {
            unmount = mountReact(
              currentTarget,
              HostReact.createElement('span', null, 'Observed control')
            );
          }
        });
        return () => {
          stopObserving();
          clearTarget();
        };
      },
    }));

    render(<installation.Host />);
    expect(installation.runtime.getDiagnostics()[0].contributions[0]).toEqual(
      expect.objectContaining({ kind: 'observe', mounted: false })
    );
    expect(callback).toHaveBeenCalledWith([]);
    expect(Object.isFrozen(callback.mock.calls[0][0])).toBe(true);

    const first = document.createElement('div');
    first.id = 'late-target';
    act(() => document.body.appendChild(first));
    expect(await screen.findByText('Observed control')).toBeInTheDocument();
    expect(callback).toHaveBeenCalledWith([first]);

    const replacement = document.createElement('div');
    replacement.id = 'late-target';
    act(() => first.replaceWith(replacement));
    await waitFor(() => {
      expect(callback).toHaveBeenCalledWith([replacement]);
    });
    expect(targetCleanup).toHaveBeenCalledWith(first);
    expect(replacement.querySelector('geist-plugin-surface')).not.toBeNull();

    act(() => unregister());
    expect(targetCleanup).toHaveBeenCalledWith(replacement);
    expect(replacement.querySelector('geist-plugin-surface')).toBeNull();
  });

  it('isolates activation, observation, and React render failures', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => undefined);
    const installation = createGeistPluginRuntime();
    const target = document.createElement('div');
    target.id = 'error-target';
    document.body.appendChild(target);

    installation.runtime.register(pluginDefinition({
      id: 'broken.activation',
      name: 'Broken activation',
      activate: () => {
        throw new Error('activation failed');
      },
    }));
    installation.runtime.register(pluginDefinition({
      id: 'broken.observe',
      name: 'Broken observer',
      activate: ({ observeTargets }) => {
        observeTargets('#error-target', () => {
          throw new Error('observer failed');
        });
      },
    }));
    installation.runtime.register(pluginDefinition({
      id: 'broken.render',
      name: 'Broken render',
      activate: ({ React: HostReact, mountReact }) => {
        const Broken = () => {
          throw new Error('render failed');
        };
        mountReact(target, HostReact.createElement(Broken));
      },
    }));
    installation.runtime.register(pluginDefinition({
      id: 'healthy.plugin',
      name: 'Healthy plugin',
      activate: ({ React: HostReact, mountReact }) => {
        mountReact(target, HostReact.createElement('span', null, 'Healthy control'));
      },
    }));

    render(<installation.Host />);
    expect(await screen.findByText('Healthy control')).toBeInTheDocument();
    await waitFor(() => {
      expect(
        installation.runtime.getDiagnostics().find((item) => item.id === 'broken.render')
      ).toEqual(expect.objectContaining({ status: 'error' }));
    });
    expect(
      installation.runtime.getDiagnostics().find((item) => item.id === 'broken.activation')
    ).toEqual(expect.objectContaining({ status: 'error', error: 'activation failed' }));
    expect(
      installation.runtime.getDiagnostics().find((item) => item.id === 'broken.observe')
    ).toEqual(expect.objectContaining({ status: 'error' }));
  });

  it('isolates diagnostics subscribers from registration and cleanup', () => {
    const consoleError = jest.spyOn(console, 'error').mockImplementation(() => undefined);
    const installation = createGeistPluginRuntime();
    const healthyListener = jest.fn();
    const unsubscribeBroken = installation.runtime.subscribeDiagnostics(() => {
      throw new Error('subscriber failed');
    });
    const unsubscribeHealthy = installation.runtime.subscribeDiagnostics(healthyListener);

    let unregister: (() => void) | undefined;
    expect(() => {
      unregister = installation.runtime.register(pluginDefinition());
    }).not.toThrow();
    expect(unregister).toEqual(expect.any(Function));
    expect(healthyListener).toHaveBeenCalledTimes(1);
    expect(installation.runtime.getDiagnostics()).toHaveLength(1);

    expect(() => unregister?.()).not.toThrow();
    expect(healthyListener).toHaveBeenCalledTimes(2);
    expect(installation.runtime.getDiagnostics()).toHaveLength(0);
    expect(() => installation.runtime.register(pluginDefinition())).not.toThrow();
    expect(consoleError).toHaveBeenCalledWith(
      'Geist plugin diagnostics listener failed:',
      expect.any(Error)
    );

    unsubscribeBroken();
    unsubscribeHealthy();
  });

  it('detects a host-development change made before subscription is attached', async () => {
    const DevelopmentStatus = () => (
      <span>{useHostDevelopmentEnabled() ? 'Development enabled' : 'Development disabled'}</span>
    );
    const EnableBeforePassiveEffects = () => {
      React.useLayoutEffect(() => {
        window.__GEIST_HOST_DEVELOPMENT__ = true;
      }, []);
      return null;
    };

    render(
      <>
        <DevelopmentStatus />
        <EnableBeforePassiveEffects />
      </>
    );

    expect(await screen.findByText('Development enabled')).toBeInTheDocument();
  });

  it('retains a read-only last error after an observation recovers', async () => {
    const installation = createGeistPluginRuntime();
    const first = document.createElement('div');
    first.id = 'recover-target';
    document.body.appendChild(first);
    let fail = true;

    installation.runtime.register(pluginDefinition({
      activate: ({ observeTargets }) => {
        observeTargets('#recover-target', () => {
          if (fail) {
            throw new Error('temporary target failure');
          }
        });
      },
    }));
    expect(installation.runtime.getDiagnostics()[0]).toEqual(expect.objectContaining({
      status: 'error',
      lastError: 'temporary target failure',
    }));

    fail = false;
    const replacement = document.createElement('div');
    replacement.id = 'recover-target';
    act(() => first.replaceWith(replacement));

    await waitFor(() => {
      expect(installation.runtime.getDiagnostics()[0]).toEqual(expect.objectContaining({
        status: 'active',
        lastError: 'temporary target failure',
      }));
    });
    expect(() => {
      (installation.runtime.getDiagnostics()[0] as { lastError?: string }).lastError = 'changed';
    }).toThrow();
  });

  it('rejects duplicate plugins, the old ABI, and unsupported API versions', () => {
    const installation = createGeistPluginRuntime();
    installation.runtime.register(pluginDefinition());

    expect(() => installation.runtime.register(pluginDefinition())).toThrow(/already registered/u);
    expect(() => installation.runtime.register(pluginDefinition({
      id: 'future.plugin',
      apiVersion: 2 as 1,
    }))).toThrow(/Unsupported Geist plugin API/u);
    expect(() => installation.runtime.register({
      protocolVersion: 1,
      id: 'old.plugin',
      displayName: 'Old Plugin',
      provider: 'Old Provider',
      version: '1.0.0',
      activate: () => ({ contributions: [] }),
    } as unknown as GeistPluginV1)).toThrow(/Unsupported Geist plugin API/u);
  });
});
