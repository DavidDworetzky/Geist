"""Scoped state for one conversation and one agent run."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from agents.models.chat_result import ToolCallResult, WorkArtifact
from agents.models.tool_calling import ChatMessage


RunStatus = Literal["running", "completed", "failed", "cancelled"]


@dataclass
class ConversationState:
    """Structured state for one chat session."""

    chat_id: int | None
    user_id: int
    messages: list[ChatMessage] = field(default_factory=list)

    def add_system_prompt(self, prompt: str | None) -> None:
        if prompt:
            self.messages.append(ChatMessage(role="system", content=prompt))

    def hydrate(self, messages: list[ChatMessage]) -> None:
        self.messages.extend(messages)

    def begin_run(self, prompt: str) -> AgentRunContext:
        return AgentRunContext(
            conversation=self,
            prompt=prompt,
            transcript=[ChatMessage(role="user", content=prompt)],
        )

    def accept_persisted_chat_id(self, chat_id: int | None) -> None:
        if chat_id is not None:
            self.chat_id = chat_id


@dataclass
class AgentRunContext:
    """Mutable execution state for one model/tool run."""

    conversation: ConversationState
    prompt: str
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    transcript: list[ChatMessage] = field(default_factory=list)
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    artifacts: list[WorkArtifact] = field(default_factory=list)
    assistant_text_parts: list[str] = field(default_factory=list)
    status: RunStatus = "running"
    model_completed: bool = False
    total_tool_calls: int = 0
    tool_result_chars: int = 0
    persisted: bool = False
    persisted_status: RunStatus | None = None
    _persistence_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    @property
    def model_messages(self) -> list[ChatMessage]:
        return [*self.conversation.messages, *self.transcript]

    @property
    def assistant_text(self) -> str:
        return "\n\n".join(part for part in self.assistant_text_parts if part).strip()

    @property
    def persistence_lock(self) -> threading.Lock:
        return self._persistence_lock

    def record_assistant(self, message: ChatMessage) -> None:
        if message.content:
            self.assistant_text_parts.append(message.content)
        self.transcript.append(message)

    def record_tool_call(self, state: ToolCallResult) -> None:
        self.tool_calls.append(state)

    def record_tool_message(self, message: ChatMessage) -> None:
        self.transcript.append(message)

    def record_artifacts(self, artifacts: list[WorkArtifact]) -> None:
        self.artifacts.extend(artifacts)

    def transition(self, status: RunStatus) -> None:
        self.status = status

    def mark_model_completed(self) -> None:
        self.model_completed = True

    def persistence_snapshot(self) -> dict[str, Any]:
        return {
            "new_user_message": self.prompt,
            "new_ai_message": self.assistant_text or None,
            "session_id": self.conversation.chat_id,
            "tool_calls": list(self.tool_calls),
            "transcript": [message.to_dict() for message in self.transcript],
            "user_id": self.conversation.user_id,
            "run_id": self.run_id,
            "status": self.status,
        }

    def mark_persisted(self, chat_id: int | None) -> None:
        self.persisted = True
        self.persisted_status = self.status
        self.conversation.accept_persisted_chat_id(chat_id)
