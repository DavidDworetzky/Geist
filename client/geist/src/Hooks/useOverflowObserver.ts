import { useCallback, useLayoutEffect, useRef, useState } from 'react';

const OVERFLOW_TOLERANCE_PX = 1;

const useOverflowObserver = <T extends HTMLElement>(enabled = true) => {
  const ref = useRef<T>(null);
  const [hasOverflow, setHasOverflow] = useState(false);

  const updateOverflowState = useCallback(() => {
    const element = ref.current;
    const nextHasOverflow = Boolean(
      enabled
      && element
      && element.scrollHeight > element.clientHeight + OVERFLOW_TOLERANCE_PX
    );

    setHasOverflow(current => current === nextHasOverflow ? current : nextHasOverflow);
  }, [enabled]);

  useLayoutEffect(() => {
    updateOverflowState();
  });

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return undefined;

    updateOverflowState();
    window.addEventListener('resize', updateOverflowState);

    if (typeof ResizeObserver === 'undefined') {
      return () => window.removeEventListener('resize', updateOverflowState);
    }

    const resizeObserver = new ResizeObserver(updateOverflowState);
    resizeObserver.observe(element);

    const content = element.firstElementChild;
    if (content) {
      resizeObserver.observe(content);
    }

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', updateOverflowState);
    };
  }, [updateOverflowState]);

  return {
    ref,
    hasOverflow,
    updateOverflowState,
  };
};

export default useOverflowObserver;
