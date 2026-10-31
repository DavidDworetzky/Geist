import { PointerEvent } from 'react';

const activeHighlights = new WeakMap<HTMLElement, Animation>();

export function acknowledgeModalBackdrop(event: PointerEvent<HTMLElement>): void {
  if (event.button !== 0 || event.target !== event.currentTarget) return;

  const surface = event.currentTarget;
  const isDialog = surface instanceof HTMLDialogElement;
  const modal = isDialog
    ? surface
    : surface.querySelector<HTMLElement>(':scope > [role="dialog"]');
  if (!modal) return;

  // HTML backdrop events target the dialog itself; ignore its interior padding.
  if (isDialog) {
    const bounds = modal.getBoundingClientRect();
    if (event.clientX >= bounds.left && event.clientX <= bounds.right
      && event.clientY >= bounds.top && event.clientY <= bounds.bottom) return;
  }

  event.preventDefault();
  activeHighlights.get(modal)?.cancel();

  const style = getComputedStyle(modal);
  const accent = style.getPropertyValue('--geist-color-accent').trim();
  const shadow = style.boxShadow === 'none' ? '' : `, ${style.boxShadow}`;
  const resting = { borderColor: style.borderColor, boxShadow: style.boxShadow };
  const highlighted = {
    borderColor: accent,
    boxShadow: `0 0 0 3px color-mix(in srgb, ${accent} 45%, transparent), 0 0 24px color-mix(in srgb, ${accent} 30%, transparent)${shadow}`,
  };
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const animation = modal.animate(
    reducedMotion
      ? [highlighted, highlighted]
      : [resting, { ...highlighted, offset: 0.22 }, resting],
    { duration: reducedMotion ? 180 : 420, easing: 'ease-out' },
  );
  animation.id = 'geist-modal-attention';
  activeHighlights.set(modal, animation);
  const clearHighlight = () => {
    if (activeHighlights.get(modal) === animation) activeHighlights.delete(modal);
  };
  animation.onfinish = clearHighlight;
  animation.oncancel = clearHighlight;
}
