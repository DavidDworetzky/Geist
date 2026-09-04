export type ToolCallStatus =
  | 'proposed'
  | 'awaiting_approval'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled';

export interface ToolCallResult {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: ToolCallStatus;
  requires_approval?: boolean;
  requires_per_call_approval?: boolean;
  result_summary?: string;
  artifact_ids?: string[];
  error?: string;
}

export interface WorkArtifact {
  id: string;
  kind: string;
  mime_type: string;
  filename?: string;
  sha256: string;
  data_base64?: string;
  url?: string;
}

export type PlanTaskStatus =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'blocked'
  | 'skipped';

export type GoalStatus =
  | 'active'
  | 'complete'
  | 'paused'
  | 'budget_limited'
  | 'failed';

export interface PlanTask {
  id: string;
  title: string;
  acceptance_criteria: string[];
  status: PlanTaskStatus;
  evidence?: string | null;
}

export interface OrchestrationState {
  objective?: string;
  agentic_mode: boolean;
  goal_id?: string | null;
  goal_status?: GoalStatus | null;
  turns_used?: number;
  max_turns?: number;
  tasks: PlanTask[];
  completion_summary?: string | null;
  completion_evidence?: string[];
  decomposition_warning?: string | null;
}

export interface CompleteTextResponse {
  message: string | string[];
  chat_id: number | null;
  run_id?: string | null;
  tool_calls?: ToolCallResult[];
  artifacts?: WorkArtifact[];
  orchestration?: OrchestrationState | null;
}

export interface ChatTurnResult {
  run_id: string | null;
  prompt: string;
  message: string;
  chat_id: number | null;
  origin_chat_id: number | null;
  tool_calls: ToolCallResult[];
  artifacts: WorkArtifact[];
  orchestration?: OrchestrationState | null;
}

export type ModelLoadState = 'unloaded' | 'loading' | 'ready' | 'failed';

export interface ModelLoadStatus {
  model_id: string;
  state: ModelLoadState;
  detail: string;
  started_at: string | null;
  updated_at: string;
}

export type ActiveTurnStatus =
  | 'connecting'
  | 'model_loading'
  | 'cancelling'
  | 'streaming'
  | 'awaiting_approval'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface ActiveChatTurn extends ChatTurnResult {
  status: ActiveTurnStatus;
  started_at: string;
  model_load?: ModelLoadStatus;
}

export interface ChatPair {
  run_id?: string | null;
  user: string;
  ai: string;
  status?: ActiveTurnStatus;
  model_load?: ModelLoadStatus;
  tool_calls?: ToolCallResult[];
  artifacts?: WorkArtifact[];
  orchestration?: OrchestrationState | null;
}

export interface ChatHistory {
  chatHistory: ChatPair[];
}

export type ToolApprovalDecision = 'approve' | 'session' | 'always' | 'deny';
