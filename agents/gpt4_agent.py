import requests
from agents.base_agent import BaseAgent

class GPT4Agent(BaseAgent):
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/engines/gpt-4/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text"):
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "echo": echo,
            "response_format": response_format
        }

        if stop is not None:
            payload["stop"] = stop

        if best_of is not None:
            payload["best_of"] = best_of

        if prompt_tokens is not None:
            payload["prompt_tokens"] = prompt_tokens

        response = requests.post(self.base_url, json=payload, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
