"""Normalized chat and tool-calling contracts shared by model backends."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from agents.models.chat_result import WorkArtifact


ToolCallStatus = Literal[
    "proposed",
    "awaiting_approval",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]
ToolSideEffect = Literal["read", "external_write", "filesystem_write", "process"]


@dataclass(frozen=True)
class ModelRequestConfig:
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
        return cls(
            id=f"toolcall_{uuid.uuid4().hex}",
            name=name,
            arguments=arguments or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_openai(self, provider_name: str | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": provider_name or self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class ChatMessage:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChatMessage:
        calls = [
            call if isinstance(call, ToolCall) else ToolCall(**call)
            for call in value.get("tool_calls", [])
        ]
        return cls(
            role=value["role"],
            content=value.get("content"),
            tool_calls=calls,
            tool_call_id=value.get("tool_call_id"),
            name=value.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            value["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            value["tool_call_id"] = self.tool_call_id
        if self.name:
            value["name"] = self.name
        return value

    def to_openai(self, tool_name_map: dict[str, str] | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            value["tool_calls"] = [
                call.to_openai((tool_name_map or {}).get(call.name)) for call in self.tool_calls
            ]
        if self.tool_call_id:
            value["tool_call_id"] = self.tool_call_id
        # OpenAI tool-result messages accept only role, content, and
        # tool_call_id. Keep the internal name for transcript hydration and
        # Anthropic serialization, but do not send it on role="tool".
        if self.name and self.role != "tool":
            value["name"] = (tool_name_map or {}).get(self.name, self.name)
        return value


@dataclass
class ToolContext:
    user_id: int
    chat_id: int | None
    run_id: str
    approved_call_ids: frozenset[str] = frozenset()


@dataclass
class ToolExecutionOutput:
    content: str
    summary: str | None = None
    artifacts: list[WorkArtifact] = field(default_factory=list)


# Argument models are validated by ToolRegistry immediately before dispatch.
# Handlers may therefore use their concrete Pydantic subtype without violating
# the registry's public contract.
ToolHandler = Callable[[ToolContext, Any], ToolExecutionOutput]
ToolAvailability = Callable[[ToolContext], bool]


@dataclass
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: ToolHandler
    side_effect: ToolSideEffect = "read"
    requires_approval: bool = False
    enabled_by_default: bool = True
    timeout_seconds: float = 30.0
    max_result_chars: int = 20_000
    source_adapter: str | None = None
    availability: ToolAvailability | None = None

    def parameters_schema(self) -> dict[str, Any]:
        schema = self.arguments_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def to_openai(self, provider_name: str | None = None) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": provider_name or self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    def to_anthropic(self, provider_name: str | None = None) -> dict[str, Any]:
        return {
            "name": provider_name or self.name,
            "description": self.description,
            "input_schema": self.parameters_schema(),
        }


@dataclass
class ToolResult:
    call: ToolCall
    status: ToolCallStatus
    content: str
    summary: str | None = None
    artifacts: list[WorkArtifact] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_message(self) -> ChatMessage:
        return ChatMessage(
            role="tool",
            name=self.call.name,
            tool_call_id=self.call.id,
            content=self.content,
        )


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class ModelEvent:
    kind: Literal["text_delta", "turn_complete"]
    text: str = ""
    turn: ModelTurn | None = None

    @classmethod
    def text_delta(cls, text: str) -> ModelEvent:
        return cls(kind="text_delta", text=text)

    @classmethod
    def turn_complete(cls, turn: ModelTurn) -> ModelEvent:
        return cls(kind="turn_complete", turn=turn)


class ToolCallingBackend:
    """Structural interface documented for chat-capable agents."""

    def stream_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        raise NotImplementedError
