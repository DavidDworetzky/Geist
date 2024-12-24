import { useState } from 'react';


const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

interface CompleteTextResponse {
  message: string;
}

const params = {
  max_tokens: 1024,
  n: 1,
  stop: ["string"],
  temperature: 1,
  top_p: 1,
  frequency_penalty: 0,
  presence_penalty: 0,
  echo: false,
  best_of: 0,
  prompt_tokens: [0],
  response_format: "text",
  agent_type: "LLAMA"
};

const useCompleteText = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [completedText, setCompletedText] = useState<string | null>(null);
  const [prompt, setPrompt] = useState<string | null>(null);

  const completeText = async (inputText: string, chat_id: number | null = null) => {
    setLoading(true);
    setError(null);
    setCompletedText(null);
    const prompt = inputText;
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
      setPrompt(prompt);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return { prompt, completeText, loading, error, completedText };
};

export default useCompleteText;
