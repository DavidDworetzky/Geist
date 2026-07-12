from typing import List, Union
from dataclasses import dataclass
import logging
from agents.models.llama_completion import LlamaCompletion
from agents.models.generic_completion import GenericCompletion
import uuid

# Create logger
logger = logging.getLogger(__name__)

@dataclass
class AgentCompletion:
    message: List[str]
    id: str
    chat_id: int

    @classmethod
    def from_completion(cls, completion: Union['LlamaCompletion', 'GenericCompletion']):
        if isinstance(completion, GenericCompletion):
            content = completion.get_assistant_content()
            if content is None:
                raise ValueError("No choices found in GenericCompletion")
            return cls(
                message=[content],
                id=completion.id,
                chat_id=completion.chat_id
            )
        elif isinstance(completion, LlamaCompletion):
            content = completion.get_assistant_content()
            if content is None:
                raise ValueError("No assistant message found in LlamaCompletion")
            return cls(
                message=[content],
                id=str(uuid.uuid4()),
                chat_id=completion.chat_id
            )
        else:
            raise ValueError(f"Unsupported completion type: {type(completion).__name__}")
