from typing import List, Optional, Union
from dataclasses import dataclass
from agents.models.gpt4_completion import Gpt4Completion
from agents.models.llama_completion import LlamaCompletion

@dataclass
class AgentCompletion:
    message: List[str]
    id: str

    @classmethod
    def from_completion(cls, completion: Union['Gpt4Completion', 'LlamaCompletion']):
        if isinstance(completion, Gpt4Completion):
            return cls(
                message=[completion.choices[0].message.content],
                id=completion.id
            )
        elif isinstance(completion, LlamaCompletion):
            return cls(
                message=[completion.generations[0][0].text],
                id=completion.llm_output.get('token_usage', {}).get('completion_id', '')
            )
        else:
            raise ValueError(f"Unsupported completion type: {type(completion).__name__}")
