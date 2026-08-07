import React, {
  Component,
  ErrorInfo,
  ReactNode,
  useSyncExternalStore,
} from 'react';
import { createPortal } from 'react-dom';

export const GEIST_PLUGIN_API_VERSION = 1 as const;
export const GEIST_PLUGIN_RUNTIME_READY_EVENT = 'geist-plugin-runtime-ready';
export const GEIST_HOST_DEVELOPMENT_UPDATED_EVENT = 'geist-host-development-updated';

export interface GeistPluginContextV1 {
  React: typeof React;
  document: Document;
  mountReact: (target: Element | ShadowRoot, content: ReactNode) => () => void;
  mountDom: (
    target: Element | ShadowRoot,
    content: Node | DocumentFragment | TrustedHTML
  ) => () => void;
  observeTargets: (
    selector: string,
    callback: (targets: readonly Element[]) => void
  ) => () => void;
}

export type GeistPluginActivationContextV1 = GeistPluginContextV1;

export interface GeistPluginV1 {
  apiVersion: typeof GEIST_PLUGIN_API_VERSION;
  id: string;
  name: string;
  provider: string;
  version: string;
  activate: (context: GeistPluginContextV1) => void | (() => void);
}

export type GeistPluginDefinitionV1 = GeistPluginV1;

export type GeistPluginDiagnosticStatusV1 = 'active' | 'error';
export type GeistPluginContributionKindV1 = 'react' | 'dom' | 'observe';

export interface GeistPluginContributionDiagnosticV1 {
  readonly id: string;
  readonly kind: GeistPluginContributionKindV1;
  readonly mounted: boolean;
  readonly error?: string;
}

export interface GeistPluginDiagnosticV1 {
  readonly apiVersion: typeof GEIST_PLUGIN_API_VERSION;
  readonly id: string;
  readonly name: string;
  readonly provider: string;
  readonly version: string;
  readonly status: GeistPluginDiagnosticStatusV1;
  readonly error?: string;
  readonly lastError?: string;
  readonly contributions: readonly GeistPluginContributionDiagnosticV1[];
}

export interface GeistPluginRuntimeV1 {
  readonly apiVersion: typeof GEIST_PLUGIN_API_VERSION;
  register: (definition: GeistPluginDefinitionV1) => () => void;
  getDiagnostics: () => readonly GeistPluginDiagnosticV1[];
  subscribeDiagnostics: (listener: () => void) => () => void;
}

interface PluginRuntimeEnvironment {
  window: Window & typeof globalThis;
  document: Document;
}

interface PluginOperation {
  id: string;
  kind: GeistPluginContributionKindV1;
  isMounted: () => boolean;
  error?: string;
}

interface ReactMount {
  id: string;
  pluginId: string;
  content: ReactNode;
  surface: HTMLElement;
}

interface RegisteredPlugin {
  definition: GeistPluginDefinitionV1;
  activationCleanup?: () => void;
  activationError?: string;
  lastError?: string;
  operationSequence: number;
  operations: Map<string, PluginOperation>;
  reactMounts: Map<string, ReactMount>;
  resources: Set<() => void>;
  active: boolean;
}

export interface GeistPluginRuntimeInstallation {
  runtime: GeistPluginRuntimeV1;
  Host: React.FC;
  install: () => void;
}

declare global {
  interface Window {
    __GEIST_PLUGIN_RUNTIME_V1__?: GeistPluginRuntimeV1;
    __GEIST_HOST_DEVELOPMENT__?: boolean;
  }
}

class PluginRuntimeStore {
  private readonly plugins = new Map<string, RegisteredPlugin>();
  private readonly listeners = new Set<() => void>();
  private revision = 0;
  private diagnosticsSnapshot: readonly GeistPluginDiagnosticV1[] = Object.freeze([]);

  constructor(private readonly environment: PluginRuntimeEnvironment) {}

  register = (definition: GeistPluginDefinitionV1): (() => void) => {
    validateDefinition(definition);
    const stableDefinition: GeistPluginDefinitionV1 = Object.freeze({
      apiVersion: definition.apiVersion,
      id: definition.id,
      name: definition.name,
      provider: definition.provider,
      version: definition.version,
      activate: definition.activate,
    });
    if (this.plugins.has(stableDefinition.id)) {
      throw new Error(`Geist plugin "${stableDefinition.id}" is already registered.`);
    }

    const plugin: RegisteredPlugin = {
      definition: stableDefinition,
      operationSequence: 0,
      operations: new Map(),
      reactMounts: new Map(),
      resources: new Set(),
      active: true,
    };
    this.plugins.set(stableDefinition.id, plugin);

    try {
      const cleanup = stableDefinition.activate(this.createActivationContext(plugin));
      if (cleanup !== undefined && typeof cleanup !== 'function') {
        throw new Error(
          `Geist plugin "${stableDefinition.id}" must return void or a cleanup function from activate.`
        );
      }
      plugin.activationCleanup = typeof cleanup === 'function' ? cleanup : undefined;
    } catch (error) {
      plugin.activationError = getErrorMessage(error);
      plugin.lastError = plugin.activationError;
      plugin.active = false;
      this.disposeResources(plugin);
    }

    this.emit();
    let unregistered = false;
    return () => {
      if (unregistered) {
        return;
      }
      unregistered = true;

      const current = this.plugins.get(stableDefinition.id);
      if (current !== plugin) {
        return;
      }

      plugin.active = false;
      this.plugins.delete(stableDefinition.id);
      try {
        plugin.activationCleanup?.();
      } catch (error) {
        console.error(`Failed to deactivate Geist plugin "${stableDefinition.id}":`, error);
      }
      this.disposeResources(plugin);
      this.emit();
    };
  };

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getRevision = (): number => this.revision;

  getDiagnostics = (): readonly GeistPluginDiagnosticV1[] => this.diagnosticsSnapshot;

  getReactMounts(): readonly ReactMount[] {
    const mounts: ReactMount[] = [];
    this.plugins.forEach((plugin) => {
      plugin.reactMounts.forEach((mount) => mounts.push(mount));
    });
    return mounts;
  }

  reportReactError(pluginId: string, operationId: string, error: unknown): void {
    const plugin = this.plugins.get(pluginId);
    const operation = plugin?.operations.get(operationId);
    if (!plugin || !operation) {
      return;
    }
    this.setOperationError(plugin, operation, error);
    this.emit();
  }

  private createActivationContext(plugin: RegisteredPlugin): GeistPluginContextV1 {
    return Object.freeze({
      React,
      document: this.environment.document,
      mountReact: (target: Element | ShadowRoot, content: ReactNode) =>
        this.mountReact(plugin, target, content),
      mountDom: (
        target: Element | ShadowRoot,
        content: Node | DocumentFragment | TrustedHTML
      ) => this.mountDom(plugin, target, content),
      observeTargets: (
        selector: string,
        callback: (targets: readonly Element[]) => void
      ) => this.observeTargets(plugin, selector, callback),
    });
  }

  private mountReact(
    plugin: RegisteredPlugin,
    target: Element | ShadowRoot,
    content: ReactNode
  ): () => void {
    this.assertPluginActive(plugin);
    this.assertTarget(target);

    const surface = this.environment.document.createElement('geist-plugin-surface');
    const operation = this.createOperation(plugin, 'react', () => surface.isConnected);
    surface.dataset.geistPluginId = plugin.definition.id;
    surface.dataset.geistPluginOperationId = operation.id;

    try {
      target.appendChild(surface);
    } catch (error) {
      plugin.operations.delete(operation.id);
      throw error;
    }

    plugin.reactMounts.set(operation.id, {
      id: operation.id,
      pluginId: plugin.definition.id,
      content,
      surface,
    });
    this.emit();

    return this.trackResource(plugin, () => {
      plugin.reactMounts.delete(operation.id);
      plugin.operations.delete(operation.id);
      surface.remove();
      this.emit();
    });
  }

  private mountDom(
    plugin: RegisteredPlugin,
    target: Element | ShadowRoot,
    content: Node | DocumentFragment | TrustedHTML
  ): () => void {
    this.assertPluginActive(plugin);
    this.assertTarget(target);

    if (typeof content === 'string') {
      throw new Error('mountDom requires a Node, DocumentFragment, or TrustedHTML object.');
    }

    let mountedNodes: Node[] = [];
    const operation = this.createOperation(
      plugin,
      'dom',
      () => mountedNodes.some((node) => node.isConnected)
    );

    try {
      if (content instanceof this.environment.window.Node) {
        mountedNodes = content instanceof this.environment.window.DocumentFragment
          ? Array.from(content.childNodes)
          : [content];
        target.appendChild(content);
      } else if (isTrustedHtml(this.environment, content)) {
        const surface = this.environment.document.createElement('geist-plugin-surface');
        surface.dataset.geistPluginId = plugin.definition.id;
        surface.dataset.geistPluginOperationId = operation.id;
        surface.innerHTML = content as unknown as string;
        target.appendChild(surface);
        mountedNodes = [surface];
      } else {
        throw new Error('mountDom requires a Node, DocumentFragment, or TrustedHTML object.');
      }
    } catch (error) {
      plugin.operations.delete(operation.id);
      mountedNodes.forEach(removeNode);
      throw error;
    }

    this.emit();
    return this.trackResource(plugin, () => {
      plugin.operations.delete(operation.id);
      mountedNodes.forEach(removeNode);
      this.emit();
    });
  }

  private observeTargets(
    plugin: RegisteredPlugin,
    selector: string,
    callback: (targets: readonly Element[]) => void
  ): () => void {
    this.assertPluginActive(plugin);
    validateNonEmptyString(selector, 'Target selector');
    if (typeof callback !== 'function') {
      throw new Error('observeTargets requires a callback function.');
    }

    // Validate the selector synchronously so registration fails deterministically.
    this.environment.document.querySelectorAll(selector);

    let currentTargets: readonly Element[] = Object.freeze([]);
    let initialized = false;
    const operation = this.createOperation(
      plugin,
      'observe',
      () => currentTargets.length > 0
    );

    const scan = (): void => {
      if (!plugin.active) {
        return;
      }

      const nextTargets = Array.from(
        this.environment.document.querySelectorAll(selector)
      );
      if (initialized && sameTargets(currentTargets, nextTargets)) {
        return;
      }
      initialized = true;
      currentTargets = Object.freeze(nextTargets);

      try {
        const result = callback(currentTargets);
        if (result !== undefined) {
          throw new Error('observeTargets callbacks must return void.');
        }
        this.setOperationError(plugin, operation, undefined);
      } catch (error) {
        this.setOperationError(plugin, operation, error);
      }
      this.emit();
    };

    const observer = new this.environment.window.MutationObserver(scan);
    const observationRoot = this.environment.document.documentElement ?? this.environment.document;
    observer.observe(observationRoot, { attributes: true, childList: true, subtree: true });
    scan();

    return this.trackResource(plugin, () => {
      observer.disconnect();
      currentTargets = Object.freeze([]);
      plugin.operations.delete(operation.id);
      this.emit();
    });
  }

  private createOperation(
    plugin: RegisteredPlugin,
    kind: GeistPluginContributionKindV1,
    isMounted: () => boolean
  ): PluginOperation {
    plugin.operationSequence += 1;
    const operation: PluginOperation = {
      id: `${kind}-${plugin.operationSequence}`,
      kind,
      isMounted,
    };
    plugin.operations.set(operation.id, operation);
    return operation;
  }

  private trackResource(plugin: RegisteredPlugin, cleanup: () => void): () => void {
    let active = true;
    const trackedCleanup = (): void => {
      if (!active) {
        return;
      }
      active = false;
      plugin.resources.delete(trackedCleanup);
      cleanup();
    };
    plugin.resources.add(trackedCleanup);
    return trackedCleanup;
  }

  private disposeResources(plugin: RegisteredPlugin): void {
    Array.from(plugin.resources).reverse().forEach((cleanup) => {
      try {
        cleanup();
      } catch (error) {
        console.error(`Failed to clean up Geist plugin "${plugin.definition.id}":`, error);
      }
    });
  }

  private assertPluginActive(plugin: RegisteredPlugin): void {
    if (!plugin.active || this.plugins.get(plugin.definition.id) !== plugin) {
      throw new Error(`Geist plugin "${plugin.definition.id}" is not active.`);
    }
  }

  private assertTarget(target: Element | ShadowRoot): void {
    const isElement = target instanceof this.environment.window.Element;
    const isShadowRoot =
      typeof this.environment.window.ShadowRoot !== 'undefined' &&
      target instanceof this.environment.window.ShadowRoot;
    if (!isElement && !isShadowRoot) {
      throw new Error(
        'Plugin mount targets must be Element or ShadowRoot objects from the host document.'
      );
    }
    if (target.ownerDocument !== this.environment.document) {
      throw new Error('Plugin mount targets must belong to the host document.');
    }
  }

  private setOperationError(
    plugin: RegisteredPlugin,
    operation: PluginOperation,
    error: unknown
  ): void {
    const nextError = error === undefined ? undefined : getErrorMessage(error);
    if (operation.error === nextError) {
      return;
    }
    operation.error = nextError;
    if (nextError) {
      plugin.lastError = nextError;
    }
  }

  private emit(): void {
    this.revision += 1;
    this.diagnosticsSnapshot = Object.freeze(
      Array.from(this.plugins.values(), (plugin): GeistPluginDiagnosticV1 => {
        const contributions = Object.freeze(
          Array.from(plugin.operations.values(), (operation) => Object.freeze({
            id: operation.id,
            kind: operation.kind,
            mounted: safelyReadMounted(operation),
            ...(operation.error ? { error: operation.error } : {}),
          }))
        );
        return Object.freeze({
          apiVersion: GEIST_PLUGIN_API_VERSION,
          id: plugin.definition.id,
          name: plugin.definition.name,
          provider: plugin.definition.provider,
          version: plugin.definition.version,
          status:
            plugin.activationError || contributions.some((item) => item.error)
              ? 'error'
              : 'active',
          ...(plugin.activationError ? { error: plugin.activationError } : {}),
          ...(plugin.lastError ? { lastError: plugin.lastError } : {}),
          contributions,
        });
      })
    );
    Array.from(this.listeners).forEach((listener) => {
      try {
        listener();
      } catch (error) {
        console.error('Geist plugin diagnostics listener failed:', error);
      }
    });
  }
}

class PluginMountErrorBoundary extends Component<
  { children: ReactNode; onError: (error: unknown) => void },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: Error, _info: ErrorInfo): void {
    this.props.onError(error);
  }

  render(): ReactNode {
    return this.state.failed ? null : this.props.children;
  }
}

function validateDefinition(definition: GeistPluginDefinitionV1): void {
  if (!definition || definition.apiVersion !== GEIST_PLUGIN_API_VERSION) {
    throw new Error(
      `Unsupported Geist plugin API. Expected ${GEIST_PLUGIN_API_VERSION}.`
    );
  }
  const expectedKeys = ['activate', 'apiVersion', 'id', 'name', 'provider', 'version'];
  const actualKeys = Object.keys(definition).sort();
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new Error('Geist plugin definitions must contain exactly the V1 fields.');
  }
  validateNonEmptyString(definition.id, 'Plugin id');
  validateNonEmptyString(definition.name, 'Plugin name');
  validateNonEmptyString(definition.provider, 'Plugin provider');
  validateNonEmptyString(definition.version, 'Plugin version');
  if (typeof definition.activate !== 'function') {
    throw new Error(`Geist plugin "${definition.id}" must provide an activate function.`);
  }
}

function validateNonEmptyString(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
}

function isTrustedHtml(
  environment: PluginRuntimeEnvironment,
  content: unknown
): content is TrustedHTML {
  if (!content || typeof content !== 'object') {
    return false;
  }

  const trustedTypes = (environment.window as unknown as {
    trustedTypes?: { isHTML?: (value: unknown) => boolean };
  }).trustedTypes;
  return trustedTypes?.isHTML?.(content) === true;
}

function safelyReadMounted(operation: PluginOperation): boolean {
  try {
    return operation.isMounted();
  } catch {
    return false;
  }
}

function sameTargets(
  current: readonly Element[],
  next: readonly Element[]
): boolean {
  return current.length === next.length && current.every((target, index) => target === next[index]);
}

function removeNode(node: Node): void {
  node.parentNode?.removeChild(node);
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function createGeistPluginRuntime(
  environment: PluginRuntimeEnvironment = { window, document }
): GeistPluginRuntimeInstallation {
  const store = new PluginRuntimeStore(environment);
  const runtime: GeistPluginRuntimeV1 = Object.freeze({
    apiVersion: GEIST_PLUGIN_API_VERSION,
    register: store.register,
    getDiagnostics: store.getDiagnostics,
    subscribeDiagnostics: store.subscribe,
  });

  const Host: React.FC = () => {
    useSyncExternalStore(store.subscribe, store.getRevision, store.getRevision);

    return (
      <>
        {store.getReactMounts().map((mount) =>
          createPortal(
            <PluginMountErrorBoundary
              key={mount.id}
              onError={(error) => store.reportReactError(mount.pluginId, mount.id, error)}
            >
              {mount.content}
            </PluginMountErrorBoundary>,
            mount.surface,
            `${mount.pluginId}:${mount.id}`
          )
        )}
      </>
    );
  };

  const install = () => {
    const existing = environment.window.__GEIST_PLUGIN_RUNTIME_V1__;
    if (existing && existing !== runtime) {
      throw new Error('A different Geist plugin runtime v1 is already installed.');
    }
    if (!existing) {
      Object.defineProperty(environment.window, '__GEIST_PLUGIN_RUNTIME_V1__', {
        configurable: true,
        enumerable: false,
        value: runtime,
        writable: false,
      });
    }
    environment.window.dispatchEvent(
      new environment.window.CustomEvent(GEIST_PLUGIN_RUNTIME_READY_EVENT, {
        detail: { apiVersion: GEIST_PLUGIN_API_VERSION },
      })
    );
  };

  return { runtime, Host, install };
}

const defaultInstallation = createGeistPluginRuntime();

export const geistPluginRuntime = defaultInstallation.runtime;
export const GeistPluginHost = defaultInstallation.Host;
export const installGeistPluginRuntime = defaultInstallation.install;

export function isHostDevelopmentEnabled(): boolean {
  return window.__GEIST_HOST_DEVELOPMENT__ === true;
}

function subscribeHostDevelopment(listener: () => void): () => void {
  window.addEventListener(GEIST_HOST_DEVELOPMENT_UPDATED_EVENT, listener);
  return () => window.removeEventListener(GEIST_HOST_DEVELOPMENT_UPDATED_EVENT, listener);
}

export function useHostDevelopmentEnabled(): boolean {
  return useSyncExternalStore(
    subscribeHostDevelopment,
    isHostDevelopmentEnabled,
    () => false
  );
}

export function useGeistPluginDiagnostics(): readonly GeistPluginDiagnosticV1[] {
  return useSyncExternalStore(
    geistPluginRuntime.subscribeDiagnostics,
    geistPluginRuntime.getDiagnostics,
    geistPluginRuntime.getDiagnostics
  );
}
