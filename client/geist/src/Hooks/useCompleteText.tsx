import { useEffect, useReducer, useRef, useState } from 'react';
import { UserSettings } from './useUserSettings';
import {
  ActiveChatTurn,
  ChatTurnResult,
  CompleteTextResponse,
  ToolCallResult,
  WorkArtifact,
} from '../chatTypes';


const AGENT_TYPE_MAPPING = {
  "online": "HTTPAGENT",
  "local": "LOCALAGENT",
  "default": "LOCALAGENT"
}

const DEFAULT_AGENT_TYPE = "LOCALAGENT";

const getAgentTypeFromSettings = (settings: UserSettings | null): string => {
  if (!settings) {
    return DEFAULT_AGENT_TYPE;
  }
  const agentType = AGENT_TYPE_MAPPING[settings.default_agent_type as keyof typeof AGENT_TYPE_MAPPING] || DEFAULT_AGENT_TYPE;
  return agentType;
};

const getDefaultParams = (settings: UserSettings | null) => ({
  max_tokens: settings?.default_max_tokens ?? 1024,
  n: 1,
  stop: null,
  temperature: settings?.default_temperature ?? 1,
  top_p: settings?.default_top_p ?? 1,
  frequency_penalty: settings?.default_frequency_penalty ?? 0,
  presence_penalty: settings?.default_presence_penalty ?? 0,
  echo: false,
  best_of: 0,
  prompt_tokens: [0],
  response_format: "text",
  agent_type: getAgentTypeFromSettings(settings),
  enable_tools: true,
});

type ToolCallUpdate = Partial<ToolCallResult> & Pick<ToolCallResult, 'id'>;

export interface ChatStreamState {
  activeTurn: ActiveChatTurn | null;
  completedTurn: ChatTurnResult | null;
  loading: boolean;
  error: string | null;
}

export type ChatStreamAction =
  | { type: 'START'; prompt: string; chatId: number | null }
  | { type: 'RUN_STARTED'; runId: string; chatId?: number | null }
  | { type: 'TEXT_DELTA'; text: string }
  | { type: 'TOOL_UPSERT'; toolCall: ToolCallUpdate }
  | { type: 'ARTIFACT_UPSERT'; artifact: WorkArtifact }
  | { type: 'FINAL'; prompt: string; data: CompleteTextResponse }
  | { type: 'DONE'; runId?: string | null; chatId?: number | null }
  | { type: 'ERROR'; message: string }
  | { type: 'CANCELLING' }
  | { type: 'CANCEL_FAILED'; message: string }
  | { type: 'CANCELLED'; chatId?: number | null }
  | { type: 'RESET' }
  | { type: 'STREAM_CLOSED' };

export const initialChatStreamState: ChatStreamState = {
  activeTurn: null,
  completedTurn: null,
  loading: false,
  error: null,
};

const normalizeMessage = (message: string | string[] | undefined): string => {
  if (Array.isArray(message)) {
    return message[0] ?? '';
  }
  return message ?? '';
};

const mergeToolCall = (
  current: ToolCallResult | undefined,
  update: ToolCallUpdate,
): ToolCallResult => ({
  id: update.id,
  name: update.name ?? current?.name ?? 'Unknown tool',
  arguments: update.arguments ?? current?.arguments ?? {},
  status: update.status ?? current?.status ?? 'proposed',
  requires_approval: update.requires_approval ?? current?.requires_approval,
  result_summary: update.result_summary ?? current?.result_summary,
  artifact_ids: update.artifact_ids ?? current?.artifact_ids,
  error: update.error ?? current?.error,
});

const upsertToolCall = (
  toolCalls: ToolCallResult[],
  update: ToolCallUpdate,
): ToolCallResult[] => {
  const existingIndex = toolCalls.findIndex((toolCall) => toolCall.id === update.id);
  if (existingIndex === -1) {
    return [...toolCalls, mergeToolCall(undefined, update)];
  }

  return toolCalls.map((toolCall, index) =>
    index === existingIndex ? mergeToolCall(toolCall, update) : toolCall
  );
};

const upsertArtifact = (
  artifacts: WorkArtifact[],
  artifact: WorkArtifact,
): WorkArtifact[] => {
  const existingIndex = artifacts.findIndex((current) => current.id === artifact.id);
  if (existingIndex === -1) {
    return [...artifacts, artifact];
  }

  return artifacts.map((current, index) =>
    index === existingIndex ? { ...current, ...artifact } : current
  );
};

const dedupeTools = (toolCalls: ToolCallResult[]): ToolCallResult[] =>
  toolCalls.reduce<ToolCallResult[]>(
    (current, toolCall) => upsertToolCall(current, toolCall),
    [],
  );

const dedupeArtifacts = (artifacts: WorkArtifact[]): WorkArtifact[] =>
  artifacts.reduce<WorkArtifact[]>(
    (current, artifact) => upsertArtifact(current, artifact),
    [],
  );

export const chatStreamReducer = (
  state: ChatStreamState,
  action: ChatStreamAction,
): ChatStreamState => {
  switch (action.type) {
    case 'START':
      return {
        activeTurn: {
          run_id: null,
          prompt: action.prompt,
          message: '',
          chat_id: action.chatId,
          origin_chat_id: action.chatId,
          status: 'connecting',
          tool_calls: [],
          artifacts: [],
        },
        completedTurn: null,
        loading: true,
        error: null,
      };
    case 'RUN_STARTED':
      if (!state.activeTurn) return state;
      return {
        ...state,
        activeTurn: {
          ...state.activeTurn,
          run_id: action.runId,
          chat_id: action.chatId ?? state.activeTurn.chat_id,
          status: 'streaming',
        },
      };
    case 'TEXT_DELTA':
      if (!state.activeTurn) return state;
      return {
        ...state,
        activeTurn: {
          ...state.activeTurn,
          message: state.activeTurn.message + action.text,
          status: 'streaming',
        },
      };
    case 'TOOL_UPSERT': {
      if (!state.activeTurn) return state;
      const toolCalls = upsertToolCall(state.activeTurn.tool_calls, action.toolCall);
      const awaitingApproval = toolCalls.some(
        (toolCall) => toolCall.status === 'awaiting_approval'
      );
      return {
        ...state,
        activeTurn: {
          ...state.activeTurn,
          tool_calls: toolCalls,
          status: awaitingApproval ? 'awaiting_approval' : 'streaming',
        },
      };
    }
    case 'ARTIFACT_UPSERT':
      if (!state.activeTurn) return state;
      return {
        ...state,
        activeTurn: {
          ...state.activeTurn,
          artifacts: upsertArtifact(state.activeTurn.artifacts, action.artifact),
        },
      };
    case 'FINAL': {
      const completedTurn: ChatTurnResult = {
        run_id: action.data.run_id ?? state.activeTurn?.run_id ?? null,
        prompt: state.activeTurn?.prompt ?? action.prompt,
        message: normalizeMessage(action.data.message),
        chat_id: action.data.chat_id,
        origin_chat_id: state.activeTurn?.origin_chat_id ?? null,
        tool_calls: dedupeTools(action.data.tool_calls ?? []),
        artifacts: dedupeArtifacts(action.data.artifacts ?? []),
      };
      return {
        ...state,
        activeTurn: null,
        completedTurn,
        error: null,
      };
    }
    case 'DONE':
      return {
        ...state,
        loading: false,
        activeTurn: state.activeTurn
          ? {
              ...state.activeTurn,
              run_id: action.runId ?? state.activeTurn.run_id,
              chat_id: action.chatId ?? state.activeTurn.chat_id,
            }
          : null,
      };
    case 'ERROR':
      return {
        ...state,
        loading: false,
        error: action.message,
        activeTurn: state.activeTurn
          ? { ...state.activeTurn, status: 'failed' }
          : null,
      };
    case 'CANCELLING':
      return {
        ...state,
        error: null,
        activeTurn: state.activeTurn
          ? { ...state.activeTurn, status: 'cancelling' }
          : null,
      };
    case 'CANCEL_FAILED':
      return {
        ...state,
        error: action.message,
        activeTurn: state.activeTurn
          ? { ...state.activeTurn, status: 'streaming' }
          : null,
      };
    case 'CANCELLED':
      return {
        ...state,
        loading: false,
        error: null,
        activeTurn: state.activeTurn
          ? {
              ...state.activeTurn,
              chat_id: action.chatId ?? state.activeTurn.chat_id,
              status: 'cancelled',
              tool_calls: state.activeTurn.tool_calls.map<ToolCallResult>((toolCall) =>
                ['proposed', 'awaiting_approval', 'running'].includes(toolCall.status)
                  ? { ...toolCall, status: 'cancelled' }
                  : toolCall
              ),
            }
          : null,
      };
    case 'RESET':
      return initialChatStreamState;
    case 'STREAM_CLOSED':
      return { ...state, loading: false };
    default:
      return state;
  }
};

interface ParsedSseEvent {
  event: string;
  data: unknown;
}

export const parseSseEventBlock = (eventBlock: string): ParsedSseEvent | null => {
  const lines = eventBlock.replace(/\r\n/g, '\n').split('\n');
  const eventName = lines
    .find((line) => line.startsWith('event:'))
    ?.slice('event:'.length)
    .trim();
  const dataLines = lines
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice('data:'.length).trimStart());

  if (!eventName || dataLines.length === 0) {
    return null;
  }

  const serializedData = dataLines.join('\n');
  try {
    return { event: eventName, data: JSON.parse(serializedData) };
  } catch {
    return { event: eventName, data: serializedData };
  }
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const useCompleteText = (userSettings: UserSettings | null = null) => {
  const [streamState, dispatch] = useReducer(chatStreamReducer, initialChatStreamState);
  const [state_chat_id, setStateChatId] = useState<number | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const activeRunIdRef = useRef<string | null>(null);

  useEffect(() => () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }, []);

  const handleSseEvent = (eventBlock: string, turnPrompt: string): string | null => {
    const parsedEvent = parseSseEventBlock(eventBlock);
    if (!parsedEvent) return null;

    const { event, data } = parsedEvent;
    if (event === 'run_started' && isRecord(data) && typeof data.run_id === 'string') {
      const chatId = typeof data.chat_id === 'number' ? data.chat_id : null;
      activeRunIdRef.current = data.run_id;
      if (chatId !== null) setStateChatId(chatId);
      dispatch({ type: 'RUN_STARTED', runId: data.run_id, chatId });
    } else if (event === 'delta' && isRecord(data)) {
      dispatch({ type: 'TEXT_DELTA', text: typeof data.text === 'string' ? data.text : '' });
    } else if (event === 'tool_call' && isRecord(data) && typeof data.id === 'string') {
      dispatch({ type: 'TOOL_UPSERT', toolCall: data as unknown as ToolCallUpdate });
    } else if (event === 'artifact' && isRecord(data) && typeof data.id === 'string') {
      dispatch({ type: 'ARTIFACT_UPSERT', artifact: data as unknown as WorkArtifact });
    } else if (event === 'final' && isRecord(data)) {
      const response = data as unknown as CompleteTextResponse;
      if (typeof response.run_id === 'string') {
        activeRunIdRef.current = response.run_id;
      }
      if (typeof response.chat_id === 'number') {
        setStateChatId(response.chat_id);
      }
      dispatch({ type: 'FINAL', prompt: turnPrompt, data: response });
    } else if (event === 'done' && isRecord(data)) {
      const runId = typeof data.run_id === 'string' ? data.run_id : activeRunIdRef.current;
      const chatId = typeof data.chat_id === 'number' ? data.chat_id : null;
      if (chatId !== null) setStateChatId(chatId);
      dispatch({ type: 'DONE', runId, chatId });
      activeRunIdRef.current = null;
    } else if (event === 'error') {
      const chatId = isRecord(data) && typeof data.chat_id === 'number'
        ? data.chat_id
        : null;
      if (chatId !== null) setStateChatId(chatId);
      const message = isRecord(data)
        ? String(data.message ?? data.error ?? 'Chat stream failed')
        : String(data || 'Chat stream failed');
      dispatch({ type: 'ERROR', message });
    } else if (event === 'cancelled') {
      const chatId = isRecord(data) && typeof data.chat_id === 'number'
        ? data.chat_id
        : null;
      if (chatId !== null) setStateChatId(chatId);
      dispatch({ type: 'CANCELLED', chatId });
      activeRunIdRef.current = null;
    }
    return event;
  };

  const completeText = async (inputText: string, chat_id?: number | null) => {
    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    activeRunIdRef.current = null;

    const currentChatId = chat_id === undefined ? state_chat_id : chat_id;
    const prompt = inputText;
    const params = getDefaultParams(userSettings);
    dispatch({ type: 'START', prompt, chatId: currentChatId });
    let terminalEventSeen = false;
    const ownsStream = () => (
      abortControllerRef.current === abortController && !abortController.signal.aborted
    );

    try {
      const endpoint = currentChatId !== null
        ? `/agent/complete_text_stream/${currentChatId}`
        : '/agent/complete_text_stream';

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt, ...params }),
        signal: abortController.signal,
      });

      if (!ownsStream()) return;

      if (!response.ok) {
        throw new Error('Failed to complete text');
      }

      if (!response.body) {
        const data: CompleteTextResponse = await response.json();
        if (!ownsStream()) return;
        if (typeof data.run_id === 'string') activeRunIdRef.current = data.run_id;
        if (typeof data.chat_id === 'number') setStateChatId(data.chat_id);
        dispatch({ type: 'FINAL', prompt, data });
        dispatch({ type: 'DONE', runId: data.run_id, chatId: data.chat_id });
        terminalEventSeen = true;
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        if (!ownsStream()) return;
        const { value, done } = await reader.read();
        if (!ownsStream()) return;
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split(/\r?\n\r?\n/);
        buffer = events.pop() ?? '';

        for (const eventBlock of events) {
          if (!ownsStream()) return;
          const handledEvent = handleSseEvent(eventBlock, prompt);
          if (handledEvent && ['final', 'error', 'cancelled'].includes(handledEvent)) {
            terminalEventSeen = true;
          }
        }
      }

      if (!ownsStream()) return;
      buffer += decoder.decode();
      if (buffer.trim()) {
        if (!ownsStream()) return;
        const handledEvent = handleSseEvent(buffer, prompt);
        if (handledEvent && ['final', 'error', 'cancelled'].includes(handledEvent)) {
          terminalEventSeen = true;
        }
      }
      if (!terminalEventSeen && ownsStream()) {
        dispatch({
          type: 'ERROR',
          message: 'Chat stream ended before a final response.',
        });
      }
    } catch (err) {
      if (abortControllerRef.current !== abortController) {
        return;
      }
      if ((err as { name?: string })?.name === 'AbortError') {
        dispatch({ type: 'CANCELLED' });
      } else {
        dispatch({
          type: 'ERROR',
          message: err instanceof Error ? err.message : 'An unknown error occurred',
        });
      }
    } finally {
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
        dispatch({ type: 'STREAM_CLOSED' });
      }
    }
  };

  const cancelGeneration = async () => {
    const runId = activeRunIdRef.current;
    if (!runId) {
      abortControllerRef.current?.abort();
      dispatch({ type: 'CANCELLED' });
      return;
    }

    dispatch({ type: 'CANCELLING' });

    try {
      const response = await fetch(`/agent/runs/${encodeURIComponent(runId)}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const result = await response.json() as { cancelled?: boolean };
      if (activeRunIdRef.current !== runId) {
        return;
      }
      if (!response.ok || result.cancelled !== true) {
        dispatch({
          type: 'CANCEL_FAILED',
          message: 'The server did not accept the cancellation request.',
        });
        return;
      }
      abortControllerRef.current?.abort();
      dispatch({ type: 'CANCELLED' });
    } catch (error) {
      console.warn(`Failed to cancel run ${runId}`, error);
      if (activeRunIdRef.current !== runId) {
        return;
      }
      dispatch({
        type: 'CANCEL_FAILED',
        message: 'Could not reach the server to cancel this run.',
      });
    }
  };

  const resetChatSession = () => {
    const runId = activeRunIdRef.current;
    if (runId) {
      void fetch(`/agent/runs/${encodeURIComponent(runId)}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }).catch((error) => console.warn(`Failed to cancel run ${runId}`, error));
    }
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    activeRunIdRef.current = null;
    setStateChatId(null);
    dispatch({ type: 'RESET' });
  };

  const prompt = streamState.activeTurn?.prompt ?? streamState.completedTurn?.prompt ?? null;
  const streamingText = streamState.activeTurn?.message ?? '';
  const completedText = streamState.completedTurn?.message ?? null;

  return {
    prompt,
    completeText,
    cancelGeneration,
    resetChatSession,
    loading: streamState.loading,
    error: streamState.error,
    completedText,
    completedTurn: streamState.completedTurn,
    activeTurn: streamState.activeTurn,
    streamingText,
    state_chat_id,
  };
};

export default useCompleteText;
