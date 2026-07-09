import { useState } from 'react';
import { UserSettings } from './useUserSettings';


const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

const AGENT_TYPE_MAPPING = {
  "online": "HTTPAGENT",
  "local": "LOCALAGENT",
  "default": "LOCALAGENT"
}

interface CompleteTextResponse {
  message: string;
  chat_id: number | null;
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
  stop: ["string"],
  temperature: settings?.default_temperature ?? 1,
  top_p: settings?.default_top_p ?? 1,
  frequency_penalty: settings?.default_frequency_penalty ?? 0,
  presence_penalty: settings?.default_presence_penalty ?? 0,
  echo: false,
  best_of: 0,
  prompt_tokens: [0],
  response_format: "text",
  agent_type: getAgentTypeFromSettings(settings)
});

const useCompleteText = (userSettings: UserSettings | null = null) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [completedText, setCompletedText] = useState<string | null>(null);
  const [state_chat_id, setStateChatId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState<string | null>(null);

  const completeText = async (inputText: string, chat_id: number | null = null) => {
    setLoading(true);
    setError(null);
    setCompletedText(null);
    chat_id = chat_id ?? state_chat_id;
    setStateChatId(null);
    const prompt = inputText;
    const params = getDefaultParams(userSettings);
    try {
      const endpoint = chat_id !== null
        ? `/agent/complete_text/${chat_id}`
        : '/agent/complete_text';

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt, ...params }),
      });

      if (!response.ok) {
        throw new Error('Failed to complete text');
      }

      const data: CompleteTextResponse = await response.json();
      if (Array.isArray(data.message) && data.message.length > 0) {
        setCompletedText(data.message[0]);
      } else {
        setCompletedText(data.message as string);
      }
      if (data.chat_id !== null) {
        setStateChatId(data.chat_id);
      }
      setPrompt(prompt);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return { prompt, completeText, loading, error, completedText, state_chat_id };
};

export default useCompleteText;
