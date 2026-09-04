from pydantic import BaseModel, Field, field_validator

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
    # more completion params for LLMs
    agent_type: AgentType | None = AgentType.LLAMA
    # Existing non-streaming API clients retain text-only behavior unless they
    # explicitly opt into the native model/tool loop.
    enable_tools: bool = False
    # Agentic mode lets the executor plan, act, wait for input, and complete.
    agentic_mode: bool = True
    memory_enabled: bool = True
    memory_mode: str = "public"
    folder_id: int | None = None


class InitializeAgentParams(BaseModel):
    prompt: str
    agent_type: str | None = None


class ToolApprovalParams(BaseModel):
    """Decision for a tool call blocked on user approval."""

    call_id: str
    # 'approve' (once) | 'session' (this chat) | 'always' (persist) | 'deny'
    decision: str


class RunInstructionParams(BaseModel):
    instruction_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Instruction must not be blank")
        return value.strip()
