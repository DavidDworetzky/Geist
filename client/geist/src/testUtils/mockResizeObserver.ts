export const installResizeObserverMock = () => {
  const originalResizeObserver = window.ResizeObserver;
  const callbacks = new Map<Element, ResizeObserverCallback>();

  class MockResizeObserver {
    private readonly observedTargets = new Set<Element>();

    constructor(private readonly callback: ResizeObserverCallback) {}

    observe(target: Element) {
      this.observedTargets.add(target);
      callbacks.set(target, this.callback);
    }

    unobserve(target: Element) {
      this.observedTargets.delete(target);
      if (callbacks.get(target) === this.callback) {
        callbacks.delete(target);
      }
    }

    disconnect() {
      this.observedTargets.forEach(target => {
        if (callbacks.get(target) === this.callback) {
          callbacks.delete(target);
        }
      });
      this.observedTargets.clear();
    }
  }

  Object.defineProperty(window, 'ResizeObserver', {
    configurable: true,
    writable: true,
    value: MockResizeObserver,
  });

  return {
    trigger(target: Element) {
      const callback = callbacks.get(target);
      if (!callback) {
        throw new Error('No ResizeObserver callback registered for the target.');
      }
      callback([], {} as ResizeObserver);
    },
    restore() {
      Object.defineProperty(window, 'ResizeObserver', {
        configurable: true,
        writable: true,
        value: originalResizeObserver,
      });
    },
  };
};
