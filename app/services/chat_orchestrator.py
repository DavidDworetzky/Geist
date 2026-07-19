"""Bounded native tool-calling loop for interactive chat."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from agents.models.agent_completion import AgentCompletion
from agents.models.agent_state import ConversationState, RunStatus
from agents.models.chat_result import ToolCallResult, WorkArtifact
from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ToolCall,
    ToolContext,
)
from app.models.database.chat_session import get_chat_history, update_chat_history
from app.services.tool_registry import ToolRegistry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatStreamEvent:
    event: str
    payload: Any


@dataclass
class _RunControl:
    cancellation: threading.Event
    on_cancel: Callable[[], bool] | None = None
    cancel_callback_claimed: bool = False


class RunControlRegistry:
    """Tracks cooperative cancellation and durable terminal callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, _RunControl] = {}

    def start(
        self,
        run_id: str,
        *,
        on_cancel: Callable[[], bool] | None = None,
    ) -> threading.Event:
        cancellation = threading.Event()
        with self._lock:
            self._runs[run_id] = _RunControl(
                cancellation=cancellation,
                on_cancel=on_cancel,
            )
        return cancellation

    def cancel(self, run_id: str) -> bool:
        callback: Callable[[], bool] | None = None
        control: _RunControl | None = None
        with self._lock:
            control = self._runs.get(run_id)
            if control is None:
                return False
            control.cancellation.set()
            if control.on_cancel is not None and not control.cancel_callback_claimed:
                control.cancel_callback_claimed = True
                callback = control.on_cancel

        # Persist cancellation before acknowledging it to the browser. This is
        # intentionally outside the registry lock because persistence can do
        # database I/O and the stream may finish concurrently.
        if callback is not None:
            try:
                accepted = callback()
            except Exception:
                logger.exception("Could not persist cancellation for chat run %s", run_id)
                accepted = False
            if not accepted:
                with self._lock:
                    if self._runs.get(run_id) is control:
                        control.cancellation.clear()
                        control.cancel_callback_claimed = False
                return False
        return True

    def finish(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)


class ChatOrchestrator:
    """Owns model turns, reviewed tool execution, events, and one-time persistence."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        run_controls: RunControlRegistry | None = None,
        max_rounds: int = 6,
        max_tool_calls: int = 10,
        max_history_entries: int = 20,
        max_history_chars: int = 80_000,
        max_tool_result_chars_total: int = 40_000,
        history_loader: Callable[[int], Any] = get_chat_history,
        history_writer: Callable[..., Any] = update_chat_history,
    ) -> None:
        self.registry = registry
        self.run_controls = run_controls or RunControlRegistry()
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.max_history_entries = max_history_entries
        self.max_history_chars = max_history_chars
        self.max_tool_result_chars_total = max_tool_result_chars_total
        self.history_loader = history_loader
        self.history_writer = history_writer

    @staticmethod
    def _entry_messages(entry: Any) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if isinstance(entry, dict):
            transcript = entry.get("transcript")
            if transcript:
                transcript_messages = [ChatMessage.from_dict(message) for message in transcript]
                if entry.get("status") not in {"failed", "cancelled"} and (
                    ChatOrchestrator._has_complete_tool_sequence(transcript_messages)
                ):
                    return transcript_messages

                # Failed/cancelled legacy runs may end with unmatched assistant
                # tool calls. Replaying those provider-invalid blocks can poison
                # every later turn, so retain only their user-visible text.
                if entry.get("user") is not None:
                    messages.append(ChatMessage(role="user", content=str(entry["user"])))
                if entry.get("ai"):
                    messages.append(ChatMessage(role="assistant", content=str(entry["ai"])))
                return messages
            if entry.get("user") is not None:
                messages.append(ChatMessage(role="user", content=str(entry["user"])))
            if entry.get("ai") is not None:
                messages.append(ChatMessage(role="assistant", content=str(entry["ai"])))
        return messages

    @staticmethod
    def _has_complete_tool_sequence(messages: list[ChatMessage]) -> bool:
        """Validate provider-required adjacency for assistant tool calls/results."""
        pending_call_ids: set[str] = set()
        for message in messages:
            if pending_call_ids:
                if (
                    message.role != "tool"
                    or not message.tool_call_id
                    or message.tool_call_id not in pending_call_ids
                ):
                    return False
                pending_call_ids.remove(message.tool_call_id)
                continue

            if message.role == "tool":
                return False
            if message.tool_calls:
                if message.role != "assistant":
                    return False
                call_ids = [call.id for call in message.tool_calls]
                if not all(call_ids) or len(call_ids) != len(set(call_ids)):
                    return False
                pending_call_ids.update(call_ids)

        return not pending_call_ids

    def _history_messages(self, history: Any) -> list[ChatMessage]:
        """Return recent complete turns without breaking tool-call/result pairs."""
        entries = list(getattr(history, "chat_history", history or []))
        selected_blocks: list[list[ChatMessage]] = []
        selected_chars = 0

        for entry in reversed(entries):
            if len(selected_blocks) >= self.max_history_entries:
                break
            block = self._entry_messages(entry)
            if not block:
                continue
            block_chars = len(
                json.dumps(
                    [message.to_dict() for message in block],
                    ensure_ascii=False,
                    default=str,
                )
            )
            if selected_chars + block_chars > self.max_history_chars:
                if selected_blocks:
                    break

                # A single legacy turn may predate result caps. Preserve only
                # its user request and final text answer so provider tool-call
                # adjacency remains valid.
                compact: list[ChatMessage] = []
                user_message = next((message for message in block if message.role == "user"), None)
                final_message = next(
                    (
                        message
                        for message in reversed(block)
                        if message.role == "assistant" and not message.tool_calls
                    ),
                    None,
                )
                per_message_limit = max(1, self.max_history_chars // 2)
                if user_message:
                    compact.append(
                        ChatMessage(
                            role="user",
                            content=(user_message.content or "")[:per_message_limit],
                        )
                    )
                if final_message:
                    compact.append(
                        ChatMessage(
                            role="assistant",
                            content=(final_message.content or "")[:per_message_limit],
                        )
                    )
                if compact:
                    selected_blocks.append(compact)
                break

            selected_blocks.append(block)
            selected_chars += block_chars

        return [message for block in reversed(selected_blocks) for message in block]

    @staticmethod
    def _tool_state(
        call: ToolCall,
        status: str,
        *,
        result_summary: str | None = None,
        artifact_ids: list[str] | None = None,
        error: str | None = None,
        requires_approval: bool = False,
    ) -> ToolCallResult:
        return ToolCallResult.create(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            status=status,
            result_summary=result_summary,
            artifact_ids=artifact_ids,
            error=error,
            requires_approval=requires_approval,
        )

    @staticmethod
    def _artifact_for_history(artifact: WorkArtifact) -> dict[str, Any]:
        """Persist bounded metadata; inline image bytes stay in the live response only."""
        return {
            "id": artifact.id,
            "kind": artifact.kind,
            "mime_type": artifact.mime_type,
            "sha256": artifact.sha256,
            "filename": artifact.filename,
            "url": artifact.url,
        }

    @staticmethod
    def _public_error_message(error: Exception) -> str:
        message = str(error)
        if message.startswith(
            (
                "Model/tool round limit exceeded",
                "Tool call limit exceeded",
                "Model backend did not complete its turn",
                "Model backend returned an invalid event",
            )
        ):
            return message
        return "Chat completion failed"

    def stream(
        self,
        *,
        backend: Any,
        prompt: str,
        user_id: int,
        chat_id: int | None,
        config: ModelRequestConfig,
        system_prompt: str | None,
        enable_tools: bool = True,
        memory_enabled: bool = True,
        memory_mode: str = "public",
        folder_id: int | None = None,
    ) -> Iterator[ChatStreamEvent]:
        conversation = ConversationState(chat_id=chat_id, user_id=user_id)
        conversation.add_system_prompt(system_prompt)
        if chat_id is not None:
            try:
                conversation.hydrate(self._history_messages(self.history_loader(chat_id)))
            except Exception as error:
                logger.warning("Could not hydrate chat %s: %s", chat_id, error)
        run = conversation.begin_run(prompt)
        context = ToolContext(user_id=user_id, chat_id=chat_id, run_id=run.run_id)
        native_tools = bool(getattr(backend, "supports_native_tool_calling", False))
        tools = (
            self.registry.definitions_for_context(context) if enable_tools and native_tools else []
        )

        def persist_turn(status: RunStatus, ai_message: str | None) -> int | None:
            with run.persistence_lock:
                if run.persisted:
                    return conversation.chat_id

                # Snapshot mutable run state so a cancellation callback can
                # safely persist while a provider/tool thread is unwinding.
                run.transition(status)
                snapshot = run.persistence_snapshot()
                snapshot["new_ai_message"] = ai_message
                snapshot["memory_enabled"] = memory_enabled
                snapshot["memory_mode"] = memory_mode
                snapshot["folder_id"] = folder_id
                snapshot["artifacts"] = [
                    self._artifact_for_history(artifact) for artifact in list(run.artifacts)
                ]
                try:
                    history = self.history_writer(**snapshot)
                except Exception:
                    logger.exception(
                        "Could not persist terminal state for chat run %s",
                        run.run_id,
                    )
                    if status == "completed":
                        raise
                    return conversation.chat_id
                persisted_chat_id = getattr(history, "chat_session_id", conversation.chat_id)
                run.mark_persisted(persisted_chat_id)
                return conversation.chat_id

        def cancelled_payload() -> dict[str, Any]:
            persisted_chat_id = persist_turn(
                "cancelled",
                run.assistant_text or None,
            )
            return {"run_id": run.run_id, "chat_id": persisted_chat_id}

        def persist_cancelled_state() -> bool:
            cancelled_payload()
            return run.persisted_status == "cancelled"

        # Register durable cancellation before exposing the run ID. The cancel
        # endpoint therefore cannot acknowledge a run and then lose its terminal
        # record merely because the browser closes the SSE response.
        cancellation = self.run_controls.start(
            run.run_id,
            on_cancel=persist_cancelled_state,
        )

        try:
            yield ChatStreamEvent(
                "run_started",
                {"run_id": run.run_id, "chat_id": conversation.chat_id},
            )

            for _round_number in range(self.max_rounds):
                if cancellation.is_set():
                    yield ChatStreamEvent("cancelled", cancelled_payload())
                    return

                completed_turn = None
                for event in backend.stream_model_turn(run.model_messages, tools, config):
                    if cancellation.is_set():
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return
                    if not isinstance(event, ModelEvent):
                        raise TypeError("Model backend returned an invalid event")
                    if event.kind == "text_delta" and event.text:
                        yield ChatStreamEvent("delta", {"text": event.text})
                    elif event.kind == "turn_complete":
                        completed_turn = event.turn

                if completed_turn is None:
                    raise RuntimeError("Model backend did not complete its turn")

                assistant_message = ChatMessage(
                    role="assistant",
                    content=completed_turn.text or None,
                    tool_calls=completed_turn.tool_calls,
                )
                run.record_assistant(assistant_message)

                if not completed_turn.tool_calls:
                    run.mark_model_completed()
                    break

                run.total_tool_calls += len(completed_turn.tool_calls)
                if run.total_tool_calls > self.max_tool_calls:
                    raise RuntimeError(f"Tool call limit exceeded ({self.max_tool_calls})")

                for call in completed_turn.tool_calls:
                    definition = self.registry.get(call.name)
                    requires_approval = bool(definition and definition.requires_approval)
                    proposed = self._tool_state(
                        call,
                        "proposed",
                        requires_approval=requires_approval,
                    )
                    yield ChatStreamEvent("tool_call", proposed)

                    if cancellation.is_set():
                        cancelled = self._tool_state(call, "cancelled")
                        run.record_tool_call(cancelled)
                        yield ChatStreamEvent("tool_call", cancelled)
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return

                    if not requires_approval:
                        yield ChatStreamEvent("tool_call", self._tool_state(call, "running"))

                    result = self.registry.execute(call, context)
                    if cancellation.is_set():
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return
                    remaining_result_chars = (
                        self.max_tool_result_chars_total - run.tool_result_chars
                    )
                    if remaining_result_chars <= 0:
                        result.content = ""
                    elif len(result.content) > remaining_result_chars:
                        truncation_marker = "\n[tool result omitted: aggregate budget exhausted]"
                        if remaining_result_chars <= len(truncation_marker):
                            result.content = truncation_marker[:remaining_result_chars]
                        else:
                            retained_chars = remaining_result_chars - len(truncation_marker)
                            result.content = f"{result.content[:retained_chars]}{truncation_marker}"
                    run.tool_result_chars += min(
                        len(result.content),
                        max(0, remaining_result_chars),
                    )
                    state = self._tool_state(
                        call,
                        result.status,
                        result_summary=result.summary,
                        artifact_ids=[artifact.id for artifact in result.artifacts],
                        error=result.error,
                        requires_approval=requires_approval,
                    )
                    run.record_tool_call(state)
                    run.record_artifacts(result.artifacts)
                    for artifact in result.artifacts:
                        yield ChatStreamEvent("artifact", artifact)
                    yield ChatStreamEvent("tool_call", state)

                    tool_message = result.to_message()
                    run.record_tool_message(tool_message)

                    if cancellation.is_set():
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return

            if not run.model_completed:
                raise RuntimeError(f"Model/tool round limit exceeded ({self.max_rounds})")

            if cancellation.is_set():
                yield ChatStreamEvent("cancelled", cancelled_payload())
                return

            final_text = run.assistant_text
            persisted_chat_id = persist_turn("completed", final_text)
            if run.persisted_status == "cancelled":
                yield ChatStreamEvent("cancelled", cancelled_payload())
                return
            completion = AgentCompletion(
                message=[final_text],
                id=f"completion_{uuid.uuid4().hex}",
                chat_id=persisted_chat_id,
                run_id=run.run_id,
                tool_calls=run.tool_calls,
                artifacts=run.artifacts,
            )
            yield ChatStreamEvent("final", completion)
            yield ChatStreamEvent(
                "done",
                {"run_id": run.run_id, "chat_id": persisted_chat_id},
            )
        except Exception as error:
            logger.exception("Chat run %s failed", run.run_id)
            persisted_chat_id = persist_turn(
                "failed",
                run.assistant_text or None,
            )
            yield ChatStreamEvent(
                "error",
                {
                    "run_id": run.run_id,
                    "chat_id": persisted_chat_id,
                    "message": self._public_error_message(error),
                },
            )
            yield ChatStreamEvent(
                "done",
                {"run_id": run.run_id, "chat_id": persisted_chat_id},
            )
        finally:
            self.run_controls.finish(run.run_id)

    def complete(self, **kwargs: Any) -> AgentCompletion:
        completion: AgentCompletion | None = None
        error: str | None = None
        for event in self.stream(**kwargs):
            if event.event == "final":
                completion = event.payload
            elif event.event == "error":
                error = event.payload.get("message", "Chat completion failed")
        if completion is None:
            raise RuntimeError(error or "Chat completion did not produce a result")
        return completion
