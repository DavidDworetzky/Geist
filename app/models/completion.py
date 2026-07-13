
from pydantic import BaseModel

from agents.agent_type import AgentType


class CompleteTextParams(BaseModel):
    prompt: str
    max_tokens: int = 16
    n: int = 1
    stop: list[str] | None = None
    temperature: float = 1.0
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    echo: bool = False
    best_of: int | None = None
    prompt_tokens: list[int] | None = None
    response_format: str = "text"
    #more completion params for LLMs
    agent_type: AgentType | None = AgentType.LLAMA

class InitializeAgentParams(BaseModel):
    prompt: str
    agent_type: str | None = None
