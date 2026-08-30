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

export interface GenerationStats {
  backend: string;
  model_id: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  cached_prompt_tokens?: number | null;
  prompt_seconds?: number | null;
  generation_seconds?: number | null;
  total_seconds?: number | null;
  time_to_first_token?: number | null;
  prompt_tps?: number | null;
  generation_tps?: number | null;
  completion_tps?: number | null;
  peak_memory_gb?: number | null;
}

export interface CompleteTextResponse {
  message: string | string[];
  chat_id: number | null;
  run_id?: string | null;
  tool_calls?: ToolCallResult[];
  artifacts?: WorkArtifact[];
  generation_stats?: GenerationStats[];
}

export interface ChatTurnResult {
  run_id: string | null;
  prompt: string;
  message: string;
  chat_id: number | null;
  origin_chat_id: number | null;
  tool_calls: ToolCallResult[];
  artifacts: WorkArtifact[];
  generation_stats: GenerationStats[];
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
  generation_stats?: GenerationStats[];
}

export interface ChatHistory {
  chatHistory: ChatPair[];
}
