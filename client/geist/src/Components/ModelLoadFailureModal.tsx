import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import { ModelLoadStatus } from '../chatTypes';
import { acknowledgeModalBackdrop } from '../Utils/modalFeedback';

interface Props {
  status: ModelLoadStatus;
  modelName: string;
  onClose: () => void;
  onRetry: () => void;
}

export default function ModelLoadFailureModal({ status, modelName, onClose, onRetry }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    const previousFocus = document.activeElement as HTMLElement | null;
    element?.showModal();
    return () => {
      element?.close();
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, []);

  const title = status.error_code === 'gpu_memory'
    ? 'Not enough GPU memory'
    : status.error_code === 'unified_memory' ? 'Not enough memory'
      : status.error_code === 'system_memory' ? 'Not enough system memory' : 'Model failed to load';
  const detail = status.detail.replace(/^Model failed to load:\s*/i, '');
  return createPortal(
    <dialog
      ref={dialog}
      className="modal-panel model-load-dialog"
      aria-labelledby="model-load-failure-title"
      aria-describedby="model-load-failure-detail"
      onPointerDown={acknowledgeModalBackdrop}
      onCancel={event => { event.preventDefault(); onClose(); }}
    >
      <div className="model-load-dialog-heading">
        <h2 id="model-load-failure-title">{title}</h2>
        <button className="button button-ghost" type="button" onClick={onClose} aria-label="Close model error">×</button>
      </div>
      <p className="model-load-dialog-model">{modelName}</p>
      <p id="model-load-failure-detail">{detail || 'The model could not start. Try again or select a different model.'}</p>
      <div className="modal-actions">
        <Link className="button button-secondary" to="/models" onClick={onClose}>Choose model</Link>
        {status.error_code === 'gpu_memory' && status.can_offload_to_system_ram === true && (
          <Link className="button" to="/settings#models" onClick={onClose}>Memory settings</Link>
        )}
        <button className="button button-secondary" type="button" onClick={onRetry}>Retry</button>
      </div>
    </dialog>,
    document.body,
  );
}
