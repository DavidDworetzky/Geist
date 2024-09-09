import { useState } from 'react';


const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

interface CompleteTextResponse {
  completedText: string;
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

  const completeText = async (inputText: string) => {
    setLoading(true);
    setError(null);
    setCompletedText(null);
    const prompt = inputText;
    try {
      const response = await fetch('/agent/complete_text', {
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
      setCompletedText(data.completedText);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return { completeText, loading, error, completedText };
};

export default useCompleteText;
