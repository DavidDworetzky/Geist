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
  { decision: 'session', label: 'Allow for session' },
  { decision: 'always', label: 'Always allow' },
  { decision: 'deny', label: 'Deny' }
];

interface ChatTextAreaProps extends ChatHistory {
  isLoading?: boolean;
  onToolApproval?: (runId: string, callId: string, decision: ToolApprovalDecision) => void;
}

const ChatTextArea = forwardRef<HTMLDivElement, ChatTextAreaProps>((props, ref) => {
  const isEmpty = props.chatHistory.length === 0 && !props.isLoading;
  const [answeredCallIds, setAnsweredCallIds] = useState<Set<string>>(new Set());

  const answerApproval = (runId: string, callId: string, decision: ToolApprovalDecision) => {
    setAnsweredCallIds((prev) => new Set(prev).add(callId));
    props.onToolApproval?.(runId, callId, decision);
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

          {element.orchestration && (
            <section className="agentic-progress" aria-label="Agentic progress">
              <div className="agentic-progress-header">
                <strong>Agentic plan</strong>
                {element.orchestration.goal_status && (
                  <span className="status-badge">
                    {statusLabel(element.orchestration.goal_status)}
                  </span>
                )}
                {typeof element.orchestration.turns_used === 'number'
                  && typeof element.orchestration.max_turns === 'number'
                  && (
                    <span className="input-help">
                      Model calls {element.orchestration.turns_used}/{element.orchestration.max_turns}
                    </span>
                  )}
              </div>
              {element.orchestration.decomposition_warning && (
                <div className="input-help">{element.orchestration.decomposition_warning}</div>
              )}
              <ol className="agentic-task-list">
                {element.orchestration.tasks.map((task) => (
                  <li key={task.id} className={`agentic-task agentic-task-${task.status}`}>
                    <span aria-hidden="true">
                      {task.status === 'completed' ? '✓' : task.status === 'blocked' ? '!' : '○'}
                    </span>
                    <span>
                      <strong>{task.title}</strong>
                      {task.evidence && <small>{task.evidence}</small>}
                      {task.skip_reason && <small>Skipped: {task.skip_reason}</small>}
                    </span>
                  </li>
                ))}
              </ol>
              {element.orchestration.instructions?.map((instruction) => (
                <p key={instruction.id}>Your instruction ({instruction.status}): {instruction.text}</p>
              ))}
            </section>
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
                  <details style={{ marginTop: 8 }}>
                    <summary>Arguments</summary>
                    <pre style={{ marginBottom: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                      {JSON.stringify(toolCall.arguments ?? {}, null, 2)}
                    </pre>
                  </details>
                )}

                {needsApproval && (
                  <div role="status" className="status-badge warning" style={{ marginTop: 8 }}>
                    Approval required
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
                    {approvalChoices
                      .filter((choice) => (
                        !toolCall.requires_per_call_approval
                        || choice.decision === 'approve'
                        || choice.decision === 'deny'
                      ))
                      .map((choice) => (
                      <button
                        key={choice.decision}
                        type="button"
                        className={`button button-small ${choice.decision === 'deny' ? 'button-danger' : 'button-secondary'}`}
                        disabled={answeredCallIds.has(toolCall.id)}
                        onClick={() => answerApproval(element.run_id as string, toolCall.id, choice.decision)}
                      >
                        {choice.label}
                      </button>
                      ))}
                    {toolCall.requires_per_call_approval && (
                      <span className="input-help">Approval applies to this command only.</span>
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
