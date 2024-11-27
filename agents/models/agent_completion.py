from typing import List, Optional, Union
from dataclasses import dataclass
from typing import Any
from agents.models.gpt4_completion import Gpt4Completion
from agents.models.llama_completion import LlamaCompletion
import uuid

@dataclass
class AgentCompletion:
    message: List[str]
    id: str
    chat_id: int

    @classmethod
    def from_completion(cls, completion: Union['Gpt4Completion', 'LlamaCompletion']):
        if isinstance(completion, Gpt4Completion):
            return cls(
                message=[completion.choices[0].message.content],
                id=completion.id,
                chat_id=completion.chat_id
            )
        elif isinstance(completion, LlamaCompletion):
            assistant_message = next((gen for gen in completion.messages if gen.role == 'assistant'), None)
            assistant_content = assistant_message.content
            assistant_chat_id = assistant_message.chat_id
            if assistant_message is None:
                raise ValueError("No assistant message found in LlamaCompletion")
            return cls(
                message=[assistant_content],
                id=str(uuid.uuid4()),
                chat_id=assistant_chat_id
            )
        else:
            raise ValueError(f"Unsupported completion type: {type(completion).__name__}")
