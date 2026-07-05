import logging
import uuid
from dataclasses import dataclass
from typing import Union

from agents.models.generic_completion import GenericCompletion
from agents.models.llama_completion import LlamaCompletion


# Create logger
logger = logging.getLogger(__name__)

@dataclass
class AgentCompletion:
    message: list[str]
    id: str
    chat_id: int | None

    @classmethod
    def from_completion(cls, completion: Union['LlamaCompletion', 'GenericCompletion']) -> 'AgentCompletion':
        if isinstance(completion, GenericCompletion):
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
