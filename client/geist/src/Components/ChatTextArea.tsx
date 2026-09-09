import React, { forwardRef, useState } from 'react';
import { ChatHistory, ToolApprovalDecision, ToolCallStatus } from '../chatTypes';


const statusLabel = (status: ToolCallStatus | string): string =>
  status.replace(/_/g, ' ');

const statusTone = (status: ToolCallStatus): string => {
  if (status === 'succeeded') return 'success';
  if (status === 'failed' || status === 'cancelled') return 'danger';
  if (status === 'awaiting_approval') return 'warning';
  return '';
};

const approvalChoices: { decision: ToolApprovalDecision; label: string }[] = [
  { decision: 'approve', label: 'Approve once' },
  { decision: 'session', label: 'Allow for this chat' },
  { decision: 'always', label: 'Always allow' },
  { decision: 'deny', label: 'Deny' }
];

interface ChatTextAreaProps extends ChatHistory {
  isLoading?: boolean;
  onToolApproval?: (
    runId: string,
    callId: string,
    decision: ToolApprovalDecision,
  ) => void | Promise<void>;
}

const ChatTextArea = forwardRef<HTMLDivElement, ChatTextAreaProps>((props, ref) => {
  const isEmpty = props.chatHistory.length === 0 && !props.isLoading;
  const [answeredCallIds, setAnsweredCallIds] = useState<Set<string>>(new Set());
  const [standingGrantUnavailable, setStandingGrantUnavailable] = useState<Set<string>>(new Set());
  const [approvalErrors, setApprovalErrors] = useState<Record<string, string>>({});

  const answerApproval = async (runId: string, callId: string, decision: ToolApprovalDecision) => {
    const key = `${runId}:${callId}`;
    setAnsweredCallIds((prev) => new Set(prev).add(key));
    setApprovalErrors((prev) => ({ ...prev, [key]: '' }));
    try {
      await props.onToolApproval?.(runId, callId, decision);
    } catch (error) {
      if (error instanceof Error && error.name === 'ApprovalUnavailable') {
        setApprovalErrors((prev) => ({ ...prev, [key]: 'This approval or decision is no longer available. Wait for an updated tool call.' }));
        return;
      }
      setAnsweredCallIds((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
      if (error instanceof Error && error.name === 'ApprovalDecisionRejected') {
        setStandingGrantUnavailable((prev) => new Set(prev).add(key));
        setApprovalErrors((prev) => ({ ...prev, [key]: 'This tool needs invocation approval. Choose Approve once or Deny.' }));
        return;
      }
      setApprovalErrors((prev) => ({ ...prev, [key]: 'Approval could not be submitted. Please retry.' }));
    }
  };

  return (
    <div ref={ref} className={`chat-history${isEmpty ? ' chat-history-empty' : ''}`}>
      {isEmpty && (
        <div className="chat-empty-state">Start a conversation with Geist.</div>
      )}
      {props.chatHistory.map((element, index) => (
        <div key={element.run_id ?? index} className="chat-turn">
          <div className="chat-message chat-message-user">
            <span className="chat-speaker">User</span>
            {element.user}
          </div>
          <div className="chat-message chat-message-ai">
            <span className="chat-speaker">Geist</span>
            {element.ai}
          </div>

          {element.model_load
            && (element.model_load.state === 'loading' || element.model_load.state === 'failed')
            && (
            <div
              aria-live="polite"
              aria-label="Model loading status"
              role="status"
              className={`model-load-indicator model-load-${element.model_load.state}`}
            >
              <strong>{element.model_load.model_id}</strong>
              <span>{element.model_load.detail}</span>
            </div>
          )}

          {element.status && element.status !== 'completed' && element.status !== 'model_loading' && (
            <div aria-live="polite" className="input-help">
              Turn status: {statusLabel(element.status)}
            </div>
          )}

          {element.tool_calls?.map((toolCall) => {
            const needsApproval =
              toolCall.status === 'awaiting_approval' ||
              (toolCall.status === 'proposed' && toolCall.requires_approval);

            return (
              <div
                key={toolCall.id}
                data-testid={`tool-call-${toolCall.id}`}
                aria-live="polite"
                style={{
                  maxWidth: 'min(820px, 92%)',
                  padding: '10px 12px',
                  color: 'var(--geist-color-text)',
                  background: 'var(--geist-color-surface-strong)',
                  border: '1px solid var(--geist-color-border)',
                  borderRadius: 'var(--geist-radius-lg)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <strong>{toolCall.name}</strong>{' '}
                  <span className={`status-badge ${statusTone(toolCall.status)}`.trim()}>
                    ({statusLabel(toolCall.status)})
                  </span>
                </div>

                {Object.keys(toolCall.arguments ?? {}).length > 0 && (
                  <details aria-label={`${toolCall.name} arguments`} open={needsApproval} style={{ marginTop: 8 }}>
                    <summary>Arguments</summary>
                    <pre style={{ marginBottom: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                      {JSON.stringify(toolCall.arguments ?? {}, null, 2)}
                    </pre>
                  </details>
                )}

                {needsApproval && (
                  <div style={{ marginTop: 8 }}>
                    <div className="input-help">Approval required</div>
                  </div>
                )}
                {toolCall.status === 'awaiting_approval'
                  && element.run_id
                  && props.onToolApproval
                  && (
                  <div
                    role="group"
                    aria-label={`Approve ${toolCall.name}`}
                    style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8 }}
                  >
                    {approvalChoices.filter((choice) => (toolCall.can_grant && !standingGrantUnavailable.has(`${element.run_id}:${toolCall.id}`))
                      || choice.decision === 'approve' || choice.decision === 'deny').map((choice) => (
                      <button
                        key={choice.decision}
                        type="button"
                        className={`button button-small ${choice.decision === 'deny' ? 'button-danger' : 'button-secondary'}`}
                        disabled={answeredCallIds.has(`${element.run_id}:${toolCall.id}`)}
                        onClick={() => { void answerApproval(element.run_id as string, toolCall.id, choice.decision); }}
                      >
                        {choice.label}
                      </button>
                    ))}
                    {approvalErrors[`${element.run_id}:${toolCall.id}`] && (
                      <div role="alert">{approvalErrors[`${element.run_id}:${toolCall.id}`]}</div>
                    )}
                  </div>
                )}
                {toolCall.result_summary && <div style={{ marginTop: 8 }}>{toolCall.result_summary}</div>}
                {toolCall.error && (
                  <div className="ErrorMessage" style={{ marginTop: 8 }}>{toolCall.error}</div>
                )}
              </div>
            );
          })}

          {element.artifacts?.map((artifact) => {
            const src = artifact.data_base64
              ? `data:${artifact.mime_type};base64,${artifact.data_base64}`
              : artifact.url;

            if (artifact.kind === 'image' && src) {
              return (
                <img
                  key={artifact.id}
                  src={src}
                  alt={artifact.filename ?? 'Generated artifact'}
                  style={{
                    maxWidth: 360,
                    width: '100%',
                    height: 'auto',
                    borderRadius: 'var(--geist-radius-lg)',
                    border: '1px solid var(--geist-color-border)',
                  }}
                />
              );
            }

            if (artifact.url) {
              return (
                <div key={artifact.id} className="input-help">
                  Artifact:{' '}
                  <a href={artifact.url} target="_blank" rel="noreferrer">
                    {artifact.filename ?? artifact.id}
                  </a>
                </div>
              );
            }

            return (
              <div key={artifact.id} className="input-help">
                Artifact: {artifact.filename ?? artifact.id}
              </div>
            );
          })}
        </div>
      ))}
      {props.isLoading && (
        <div
          className="chat-loading-indicator"
          role="status"
          aria-label="Geist is responding"
        >
          <span className="chat-loading-dot" aria-hidden="true" />
          <span className="chat-loading-dot" aria-hidden="true" />
          <span className="chat-loading-dot" aria-hidden="true" />
        </div>
      )}
    </div>
  );
});

ChatTextArea.displayName = 'ChatTextArea';

export default ChatTextArea;
