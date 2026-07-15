import logging
import uuid
from dataclasses import dataclass, field

from agents.models.chat_result import ToolCallResult, WorkArtifact
from agents.models.generic_completion import GenericCompletion
from agents.models.llama_completion import LlamaCompletion


# Create logger
logger = logging.getLogger(__name__)


@dataclass
class AgentCompletion:
    message: list[str]
    id: str
    chat_id: int | None
    run_id: str | None = None
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    artifacts: list[WorkArtifact] = field(default_factory=list)

    @classmethod
    def from_completion(cls, completion: LlamaCompletion | GenericCompletion) -> "AgentCompletion":
        if isinstance(completion, GenericCompletion):
            content = completion.get_assistant_content()
            if content is None:
                raise ValueError("No choices found in GenericCompletion")
            return cls(
                message=[content],
                id=completion.id,
                chat_id=completion.chat_id,
                run_id=getattr(completion, "run_id", None),
                tool_calls=getattr(completion, "tool_calls", []),
                artifacts=getattr(completion, "artifacts", []),
            )
        elif isinstance(completion, LlamaCompletion):
            content = completion.get_assistant_content()
            if content is None:
                raise ValueError("No assistant message found in LlamaCompletion")
            return cls(
                message=[content],
                id=str(uuid.uuid4()),
                chat_id=completion.chat_id,
                run_id=getattr(completion, "run_id", None),
                tool_calls=getattr(completion, "tool_calls", []),
                artifacts=getattr(completion, "artifacts", []),
            )
        else:
            raise ValueError(f"Unsupported completion type: {type(completion).__name__}")
