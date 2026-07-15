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

export interface CompleteTextResponse {
  message: string | string[];
  chat_id: number | null;
  run_id?: string | null;
  tool_calls?: ToolCallResult[];
  artifacts?: WorkArtifact[];
}

export interface ChatTurnResult {
  run_id: string | null;
  prompt: string;
  message: string;
  chat_id: number | null;
  origin_chat_id: number | null;
  tool_calls: ToolCallResult[];
  artifacts: WorkArtifact[];
}

export type ActiveTurnStatus =
  | 'connecting'
  | 'cancelling'
  | 'streaming'
  | 'awaiting_approval'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface ActiveChatTurn extends ChatTurnResult {
  status: ActiveTurnStatus;
}

export interface ChatPair {
  run_id?: string | null;
  user: string;
  ai: string;
  status?: ActiveTurnStatus;
  tool_calls?: ToolCallResult[];
  artifacts?: WorkArtifact[];
}

export interface ChatHistory {
  chatHistory: ChatPair[];
}
