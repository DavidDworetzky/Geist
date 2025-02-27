from typing import List, Optional, Union
from dataclasses import dataclass
from typing import Any
import logging
from agents.models.gpt4_completion import Gpt4Completion
from agents.models.llama_completion import LlamaCompletion
import uuid

# Create logger
logger = logging.getLogger(__name__)

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
            if assistant_message is None:
                raise ValueError("No assistant message found in LlamaCompletion")
            return cls(
                message=[assistant_message.content],
                id=str(uuid.uuid4()),
                chat_id=completion.chat_id
            )
        else:
            raise ValueError(f"Unsupported completion type: {type(completion).__name__}")
