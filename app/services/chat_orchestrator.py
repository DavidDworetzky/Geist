"""Bounded native tool-calling loop for interactive chat."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import ValidationError

from agents.model_load_errors import ModelMemoryError
from agents.models.agent_completion import AgentCompletion
from agents.models.agent_state import ConversationState, RunStatus
from agents.models.chat_result import ToolCallResult, WorkArtifact
from agents.models.tool_calling import (
    ChatMessage,
    InvocationApproval,
    MalformedToolCallError,
    ModelEvent,
    ModelRequestConfig,
    ToolCall,
    ToolContext,
    ToolResult,
    tool_requires_approval,
)
from app.models.database.chat_session import get_chat_history, update_chat_history
from app.services.agent_permissions import AgentPermissions, load_agent_permissions
from app.services.goal_runtime import (
    CheckpointRecoveryError,
    GoalRuntime,
    GoalRuntimeRegistry,
    GoalStore,
    NullGoalStore,
)
from app.services.tool_approvals import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    SessionGrantRegistry,
    ToolApprovalRegistry,
    approval_registry,
    persist_always_allow,
    session_grants,
    tool_arguments_fingerprint,
)
from app.services.tool_intent_router import ToolIntentDecision, ToolIntentRouter
from app.services.tool_registry import ToolRegistry


UNATTENDED_DENY_MESSAGE = (
    "BLOCKED: this tool requires user approval, but the run is unattended "
    "(no user is present to approve it). Find an approach that avoids this "
    "tool, and do NOT retry the call."
)

DENIED_MESSAGE = (
    "BLOCKED: the user denied this tool call. The user has NOT consented to "
    "this action. Do NOT retry it, and do NOT attempt the same outcome "
    "through a different tool."
)

TIMEOUT_MESSAGE = (
    "BLOCKED: the approval request timed out without a user response. "
    "Silence is not consent. Do NOT retry the call."
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatStreamEvent:
    event: str
    payload: Any


@dataclass
class _RunControl:
    workspace_id: int
    cancellation: threading.Event
    on_cancel: Callable[[], bool] | None = None
    cancel_callback_claimed: bool = False
    chat_id: int | None = None
    accepting_input: bool = True
    inbox: list[dict[str, str]] = field(default_factory=list)
    input_pending: threading.Event = field(default_factory=threading.Event)
    received: dict[str, dict[str, str]] = field(default_factory=dict)
    on_instruction: Callable[[dict[str, str]], None] | None = None
    enqueue_lock: threading.Lock = field(default_factory=threading.Lock)
    pending_accepts: int = 0


class RunAlreadyActiveError(RuntimeError):
    pass


class RunControlRegistry:
    """Tracks cooperative cancellation and durable terminal callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, _RunControl] = {}

    def start(
        self,
        run_id: str,
        *,
        workspace_id: int,
        cancellation: threading.Event | None = None,
        on_cancel: Callable[[], bool] | None = None,
        chat_id: int | None = None,
        on_instruction: Callable[[dict[str, str]], None] | None = None,
    ) -> threading.Event:
        cancellation = cancellation or threading.Event()
        with self._lock:
            if chat_id is not None and any(
                c.chat_id == chat_id and c.workspace_id == workspace_id for c in self._runs.values()
            ):
                raise RunAlreadyActiveError(
                    "This chat already has an active run; send instructions to that run."
                )
            self._runs[run_id] = _RunControl(
                workspace_id=workspace_id,
                cancellation=cancellation,
                on_cancel=on_cancel,
                chat_id=chat_id,
                on_instruction=on_instruction,
            )
        return cancellation

    def enqueue(
        self, run_id: str, workspace_id: int, instruction_id: str, text: str
    ) -> dict[str, str] | None:
        with self._lock:
            control = self._runs.get(run_id)
        if control is None or control.workspace_id != workspace_id:
            return None
        with control.enqueue_lock:
            with self._lock:
                if (
                    self._runs.get(run_id) is not control
                    or not control.accepting_input
                    or control.cancellation.is_set()
                ):
                    return None
                existing = control.received.get(instruction_id)
                if existing:
                    if existing["text"] != text:
                        raise ValueError("Instruction ID was already used for different text")
                    return existing
                if len(control.inbox) >= 20:
                    raise ValueError(
                        "Too many queued instructions; wait for the agent to read them."
                    )
                control.pending_accepts += 1
            instruction = {"id": instruction_id, "text": text, "status": "queued"}
            try:
                if control.on_instruction:
                    control.on_instruction(instruction)
                with self._lock:
                    control.inbox.append(instruction)
                    control.received[instruction_id] = instruction
                    control.input_pending.set()
                return instruction
            finally:
                with self._lock:
                    control.pending_accepts -= 1

    def drain(self, run_id: str) -> list[dict[str, str]]:
        with self._lock:
            control = self._runs[run_id]
            instructions, control.inbox = control.inbox, []
            control.input_pending.clear()
            return instructions

    def pending_input(self, run_id: str) -> threading.Event:
        with self._lock:
            return self._runs[run_id].input_pending

    def seal(self, run_id: str, *, force: bool = False) -> bool:
        with self._lock:
            control = self._runs[run_id]
            if (control.inbox or control.pending_accepts) and not force:
                return False
            control.accepting_input = False
            return True

    def cancel(self, run_id: str, *, workspace_id: int) -> bool:
        callback: Callable[[], bool] | None = None
        control: _RunControl | None = None
        with self._lock:
            control = self._runs.get(run_id)
            if control is None or control.workspace_id != workspace_id:
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
        doom_loop_threshold: int = 3,
        max_history_entries: int = 20,
        max_history_chars: int = 80_000,
        max_tool_result_chars_total: int = 40_000,
        history_loader: Callable[[int], Any] = get_chat_history,
        history_writer: Callable[..., Any] = update_chat_history,
        permissions_loader: Callable[[int], AgentPermissions] = load_agent_permissions,
        approvals: ToolApprovalRegistry = approval_registry,
        grants: SessionGrantRegistry = session_grants,
        always_allow_persister: Callable[[int, str], None] = persist_always_allow,
        orchestration_runs: GoalRuntimeRegistry | None = None,
        goal_store: GoalStore | None = None,
        goal_max_turns: int = 48,
        approval_timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        intent_router: ToolIntentRouter | None = None,
    ) -> None:
        self.registry = registry
        self.permissions_loader = permissions_loader
        self.approvals = approvals
        self.grants = grants
        self.approval_timeout_seconds = approval_timeout_seconds
        self.always_allow_persister = always_allow_persister
        self.run_controls = run_controls or RunControlRegistry()
        self.intent_router = intent_router
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.doom_loop_threshold = doom_loop_threshold
        self.max_history_entries = max_history_entries
        self.max_history_chars = max_history_chars
        self.max_tool_result_chars_total = max_tool_result_chars_total
        self.history_loader = history_loader
        self.history_writer = history_writer
        self.orchestration_runs = orchestration_runs or GoalRuntimeRegistry()
        self.goal_store = goal_store or NullGoalStore()
        self.goal_max_turns = max(1, min(goal_max_turns, 200))

    def _new_invocation(self, call: ToolCall, context: ToolContext) -> ToolCall:
        arguments = json.loads(json.dumps(call.arguments))
        definition = self.registry.get(call.name, context)
        if definition and definition.arguments_model is not None:
            # Dispatch returns a structured error for invalid arguments.
            with suppress(ValidationError):
                arguments = definition.arguments_model.model_validate(arguments).model_dump(
                    mode="json"
                )
        return ToolCall.create(call.name, arguments)

    @staticmethod
    def _bounded_result(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        marker = "\n[tool result truncated by context budget]"
        return content[: max(0, limit - len(marker))] + marker[:limit]

    def _model_context(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        # Keep protocol IDs and call/result pairs intact. Retain recent output
        # preferentially instead of spending a lifetime quota on the first calls.
        remaining = self.max_tool_result_chars_total
        context = []
        for message in reversed(messages):
            if message.role == "tool" and not message.preserve_content:
                content = self._bounded_result(message.content or "", remaining)
                remaining -= len(content)
                message = replace(message, content=content)
            context.append(message)
        ordered = list(reversed(context))
        systems = [message for message in ordered if message.role == "system"]
        blocks: list[list[ChatMessage]] = []
        for message in ordered:
            if message.role == "system":
                continue
            if message.role == "tool" and blocks and blocks[-1][0].tool_calls:
                blocks[-1].append(message)
            else:
                blocks.append([message])
        selected: list[tuple[int, list[ChatMessage]]] = []
        remaining_chars = self.max_history_chars
        protected = [
            index
            for index, block in enumerate(blocks)
            if any(item.preserve_content for item in block)
        ][-4:]
        indices = sorted(
            range(len(blocks)), key=lambda index: (index in protected, index), reverse=True
        )
        for index in indices:
            block = blocks[index]
            if len(selected) >= self.max_history_entries:
                break
            size = len(json.dumps([item.to_dict() for item in block], ensure_ascii=False))
            if size > remaining_chars:
                if len(block) != 1 or block[0].tool_calls:
                    continue  # Drop the whole protocol block, never orphan a result.
                if remaining_chars <= 128:
                    continue
                block = [
                    replace(
                        block[0],
                        content=self._bounded_result(block[0].content or "", remaining_chars - 128),
                    )
                ]
                size = len(json.dumps([item.to_dict() for item in block], ensure_ascii=False))
            remaining_chars -= size
            selected.append((index, block))
        return systems + [item for _, block in sorted(selected) for item in block]

    def _checkpoint_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [
            {**item.to_dict(), "_preserve_control_result": item.preserve_content}
            for item in self._model_context(messages)
            if item.role != "system"
        ]

    @staticmethod
    def _checkpoint_message(value: dict[str, Any]) -> ChatMessage:
        message = ChatMessage.from_dict(value)
        # Only trusted server-side goal checkpoints may restore this hint.
        message.preserve_content = (
            message.role == "tool" and value.get("_preserve_control_result") is True
        )
        return message

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
        requires_per_call_approval: bool = False,
        can_grant: bool = False,
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
            requires_per_call_approval=requires_per_call_approval,
            can_grant=can_grant,
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
        if isinstance(error, ModelMemoryError):
            return str(error)
        if isinstance(error, MalformedToolCallError):
            return "The model returned malformed tool calls and exhausted its recovery budget. Please try again."
        if isinstance(error, CheckpointRecoveryError):
            return str(error)
        message = str(error)
        if message.startswith(
            (
                "Model/tool round limit exceeded",
                "Tool call limit exceeded",
                "Model backend did not complete its turn",
                "Model backend returned an invalid event",
                "Doom loop detected",
            )
        ):
            return message
        return "Chat completion failed"

    def stream(
        self,
        *,
        backend: Any,
        prompt: str,
        workspace_id: int,
        chat_id: int | None,
        config: ModelRequestConfig,
        system_prompt: str | None,
        enable_tools: bool = True,
        enable_intent_router: bool = False,
        memory_enabled: bool = True,
        memory_mode: str = "public",
        folder_id: int | None = None,
        interactive: bool = True,
        agentic_mode: bool = False,
        yield_approval_wait: bool = False,
        cancellation: threading.Event | None = None,
    ) -> Iterator[ChatStreamEvent]:
        conversation = ConversationState(chat_id=chat_id, user_id=workspace_id)
        conversation.add_system_prompt(system_prompt)
        if chat_id is not None:
            try:
                conversation.hydrate(self._history_messages(self.history_loader(chat_id)))
            except Exception as error:
                logger.warning("Could not hydrate chat %s: %s", chat_id, error)
        run = conversation.begin_run(prompt)
        user_id = workspace_id
        cancellation = cancellation or threading.Event()
        denied_tools: set[str] = set()
        pending_text: list[str] = []
        try:
            permissions = self.permissions_loader(user_id)
        except Exception:
            logger.exception("Could not load run permissions; requiring approval for every tool")
            permissions = AgentPermissions(mode="require_approval")
        native_tools = bool(getattr(backend, "supports_native_tool_calling", False))
        goal_loop_enabled = agentic_mode and native_tools
        if agentic_mode and not native_tools:
            logger.info(
                "Agentic goal loop disabled for run %s because the backend does not "
                "support native tool calling",
                run.run_id,
            )
        context = ToolContext(
            workspace_id=workspace_id,
            cancellation=cancellation,
            chat_id=chat_id,
            run_id=run.run_id,
            permission_mode=permissions.mode,
            always_allow_tools=frozenset(permissions.always_allow),
            agentic_mode=goal_loop_enabled,
        )
        runtime: GoalRuntime | None = None
        run_instructions: list[dict[str, str]] = []
        available_tools = self.registry.definitions_for_context(context) if native_tools else []
        tools = [
            definition
            for definition in available_tools
            if enable_tools or definition.approval_exempt
        ]

        def persist_turn(status: RunStatus, ai_message: str | None) -> int | None:
            with run.persistence_lock:
                if run.persisted:
                    return conversation.chat_id

                if status in {"failed", "cancelled"} and pending_text:
                    # Persist only emitted prose, never an unvalidated tool call.
                    # Cancellation can append from its endpoint thread; mutable
                    # transcript reads/writes therefore share this lock.
                    run.record_assistant(
                        ChatMessage(role="assistant", content="".join(pending_text))
                    )
                    pending_text.clear()
                    ai_message = run.assistant_text or None

                # Snapshot mutable run state so a cancellation callback can
                # safely persist while a provider/tool thread is unwinding.
                run.transition(status)
                snapshot = run.persistence_snapshot()
                snapshot["new_ai_message"] = ai_message
                snapshot["memory_enabled"] = memory_enabled
                snapshot["memory_mode"] = memory_mode
                snapshot["folder_id"] = folder_id
                snapshot["orchestration"] = runtime.snapshot() if runtime is not None else None
                snapshot["instructions"] = list(run_instructions)
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
                if conversation.chat_id is None and persisted_chat_id is not None:
                    self.grants.promote_run(workspace_id, run.run_id, persisted_chat_id)
                    if self.registry.session_manager is not None:
                        self.registry.session_manager.promote_scope(
                            f"workspace:{workspace_id}:run:{run.run_id}",
                            f"workspace:{workspace_id}:chat:{persisted_chat_id}",
                        )
                run.mark_persisted(persisted_chat_id)
                if (
                    runtime is not None
                    and runtime.state.goal_id is not None
                    and persisted_chat_id is not None
                ):
                    self.goal_store.attach_chat(runtime.state.goal_id, persisted_chat_id)
                return conversation.chat_id

        def cancelled_payload() -> dict[str, Any]:
            persisted_chat_id = persist_turn(
                "cancelled",
                run.assistant_text or None,
            )
            return {"run_id": run.run_id, "chat_id": persisted_chat_id}

        def persist_cancelled_state() -> bool:
            if runtime is not None:
                runtime.pause()
            cancelled_payload()
            return run.persisted_status == "cancelled"

        # Register durable cancellation before exposing the run ID. The cancel
        # endpoint therefore cannot acknowledge a run and then lose its terminal
        # record merely because the browser closes the SSE response.
        def receive_instruction(item: dict[str, str]) -> None:
            if runtime:
                runtime.add_instruction(item)
            run_instructions.append(item)

        try:
            cancellation = self.run_controls.start(
                run.run_id,
                workspace_id=workspace_id,
                cancellation=cancellation,
                on_cancel=persist_cancelled_state,
                chat_id=chat_id,
                on_instruction=receive_instruction,
            )
        except RunAlreadyActiveError as error:
            yield ChatStreamEvent(
                "error",
                {
                    "code": "run_already_active",
                    "message": str(error),
                    "chat_id": chat_id,
                },
            )
            yield ChatStreamEvent("done", {"run_id": None, "chat_id": chat_id})
            return
        pending_input = self.run_controls.pending_input(run.run_id)

        def apply_instructions(*, mark_applied: bool = True) -> list[dict[str, str]]:
            instructions = self.run_controls.drain(run.run_id)
            for item in instructions:
                item["status"] = "applied" if mark_applied else "queued"
                run.transcript.append(ChatMessage(role="user", content=item["text"]))
            if runtime and instructions:
                runtime.checkpoint(self._checkpoint_messages(run.model_messages))
            return instructions

        try:
            if goal_loop_enabled:
                saved = (
                    self.goal_store.load_latest(user_id, chat_id) if chat_id is not None else None
                )
                if saved is not None and not isinstance(saved, dict):
                    raise CheckpointRecoveryError(
                        "Saved goal is invalid. Start a new chat; the checkpoint is preserved."
                    )
                resume = saved if saved and saved.get("goal_status") != "complete" else None
                if cancellation.is_set():
                    yield ChatStreamEvent("cancelled", cancelled_payload())
                    return
                runtime = GoalRuntime(
                    objective=prompt,
                    tasks=[],
                    max_turns=self.goal_max_turns,
                    saved=resume,
                    on_change=self.goal_store.update,
                    workspace_id=(saved or {}).get("workspace_id")
                    or f"user-{user_id}-chat-{chat_id or run.run_id}",
                )
                if resume:
                    conversation.messages = [
                        message for message in conversation.messages if message.role == "system"
                    ]
                    try:
                        run.transcript = [
                            self._checkpoint_message(message) for message in runtime.transcript
                        ]
                    except (TypeError, ValueError, KeyError) as error:
                        raise CheckpointRecoveryError(
                            "Saved goal transcript is invalid. Start a new chat; the checkpoint is preserved."
                        ) from error
                    if not self._has_complete_tool_sequence(run.transcript):
                        raise CheckpointRecoveryError(
                            "Saved goal transcript is incomplete. Start a new chat; the checkpoint is preserved."
                        )
                    run.transcript = self._model_context(run.transcript)
                    for item in runtime.state.instructions:
                        if item.get("status") == "queued":
                            run.transcript.append(ChatMessage(role="user", content=item["text"]))
                            item["status"] = "applied"
                    run.transcript.append(ChatMessage(role="user", content=prompt))
                    runtime.state.instructions.append(
                        {"id": run.run_id, "text": prompt, "status": "applied"}
                    )
                else:
                    self.goal_store.create(
                        {**runtime.snapshot(), "transcript": []}, user_id, run.run_id
                    )
                if chat_id is not None and runtime.state.goal_id is not None:
                    self.goal_store.attach_chat(runtime.state.goal_id, chat_id)
                context = replace(context, coding_workspace_id=runtime.state.workspace_id)
                self.orchestration_runs.start(run.run_id, runtime)
                orchestration_prompt = runtime.execution_prompt()
                if conversation.messages and conversation.messages[0].role == "system":
                    current_prompt = conversation.messages[0].content or ""
                    conversation.messages[
                        0
                    ].content = f"{current_prompt}\n\n{orchestration_prompt}".strip()
                else:
                    conversation.messages.insert(
                        0, ChatMessage(role="system", content=orchestration_prompt)
                    )
                snapshot = runtime.snapshot()
                yield ChatStreamEvent(
                    "plan",
                    {
                        "tasks": snapshot["tasks"],
                        "warning": snapshot["decomposition_warning"],
                    },
                )
                yield ChatStreamEvent("goal", snapshot)

            if enable_tools and native_tools and not goal_loop_enabled:
                if self.intent_router is None or not enable_intent_router:
                    tools = self.registry.definitions_for_context(context)
                else:
                    try:
                        route_decision = self.intent_router.classify(backend, run.model_messages)
                    except Exception:
                        logger.exception(
                            "Tool intent classification failed for run %s; using Action",
                            run.run_id,
                        )
                        route_decision = ToolIntentDecision()
                    tools = self.registry.definitions_for_intent(
                        context,
                        route_decision.intent,
                        include_retrieval=route_decision.needs_retrieval,
                    )
                    logger.info(
                        "Tool intent for run %s: %s, retrieval=%s (%s)",
                        run.run_id,
                        route_decision.intent,
                        route_decision.needs_retrieval,
                        ", ".join(tool.name for tool in tools) or "no tools",
                    )

            yield ChatStreamEvent(
                "run_started", {"run_id": run.run_id, "chat_id": conversation.chat_id}
            )
            model_limit = self.goal_max_turns if runtime else self.max_rounds
            malformed_attempts = 0
            format_correction = None
            doom_signature: str | None = None
            doom_count = 0
            prose_turns = 0
            for model_index in range(model_limit):
                for instruction in apply_instructions():
                    yield ChatStreamEvent("user_instruction", instruction)
                if cancellation.is_set():
                    yield ChatStreamEvent("cancelled", cancelled_payload())
                    return

                completed_turn = None
                with run.persistence_lock:
                    model_messages = self._model_context(run.model_messages)
                if format_correction:
                    model_messages.append(ChatMessage(role="user", content=format_correction))
                responses = None
                malformed_error = None
                try:
                    responses = backend.stream_model_turn(model_messages, tools, config)
                    for event in responses:
                        if cancellation.is_set():
                            yield ChatStreamEvent("cancelled", cancelled_payload())
                            return
                        if not isinstance(event, ModelEvent):
                            raise TypeError("Model backend returned an invalid event")
                        if event.kind == "text_delta" and event.text:
                            with run.persistence_lock:
                                if cancellation.is_set() or run.persisted:
                                    continue
                                pending_text.append(event.text)
                            yield ChatStreamEvent("delta", {"text": event.text})
                        elif event.kind == "turn_complete":
                            completed_turn = event.turn
                except MalformedToolCallError as error:
                    malformed_error = error
                finally:
                    close = getattr(responses, "close", None)
                    if callable(close):
                        close()

                if completed_turn is not None and any(
                    call.name not in {tool.name for tool in tools}
                    for call in completed_turn.tool_calls
                ):
                    malformed_error = MalformedToolCallError(
                        "Model requested a tool not offered for this turn"
                    )
                if malformed_error is not None:
                    if cancellation.is_set():
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return
                    malformed_attempts += 1
                    if runtime:
                        runtime.finish_turn()
                    if malformed_attempts > 2 or model_index + 1 == model_limit:
                        raise malformed_error
                    logger.warning(
                        "Chat run %s retrying malformed model output (%s/2)",
                        run.run_id,
                        malformed_attempts,
                    )
                    notice = f"\n\nRetrying malformed tool output ({malformed_attempts}/2)…\n\n"
                    with run.persistence_lock:
                        if not run.persisted and "".join(pending_text).strip():
                            run.record_assistant(
                                ChatMessage(role="assistant", content="".join(pending_text))
                            )
                        pending_text.clear()
                    format_correction = (
                        "Your previous generation contained malformed tool-call syntax. "
                        "No tools from that attempted turn executed. Regenerate the intended "
                        "call using the provided tool format and schema, with exactly one JSON "
                        "value per non-string parameter and no trailing prose inside JSON. "
                        "Do not repeat tools whose successful results are already in this conversation."
                    )
                    if runtime:
                        yield ChatStreamEvent("goal", runtime.snapshot())
                    yield ChatStreamEvent("delta", {"text": notice})
                    continue

                format_correction = None
                if completed_turn is None:
                    raise RuntimeError("Model backend did not complete its turn")

                completed_turn.tool_calls = [
                    self._new_invocation(call, context) for call in completed_turn.tool_calls
                ]
                assistant_message = ChatMessage(
                    role="assistant",
                    content=completed_turn.text or None,
                    tool_calls=completed_turn.tool_calls,
                )
                with run.persistence_lock:
                    if not run.persisted:
                        run.record_assistant(assistant_message)
                    pending_text.clear()

                run.total_tool_calls += len(completed_turn.tool_calls)
                call_count = len(completed_turn.tool_calls) if runtime else run.total_tool_calls
                if call_count > self.max_tool_calls:
                    raise RuntimeError(f"Tool call limit exceeded ({self.max_tool_calls})")

                for call in completed_turn.tool_calls:
                    grant_scope = (
                        f"workspace:{workspace_id}:chat:{conversation.chat_id}"
                        if conversation.chat_id is not None
                        else f"workspace:{workspace_id}:run:{run.run_id}"
                    )
                    # Interrupt runs stuck re-issuing the same call: the model
                    # has stopped making progress and each repeat burns tokens
                    # (and possibly side effects) for an identical answer.
                    signature = (
                        f"{call.name}:" f"{json.dumps(call.arguments, sort_keys=True, default=str)}"
                    )
                    if signature == doom_signature:
                        doom_count += 1
                    else:
                        doom_signature = signature
                        doom_count = 1
                    if doom_count >= self.doom_loop_threshold:
                        raise RuntimeError(
                            f"Doom loop detected: tool '{call.name}' was called "
                            f"{doom_count} times in a row with identical arguments"
                        )

                    definition = self.registry.get(call.name, context)
                    invocation_approval = None
                    pending = None
                    requires_per_call_approval = bool(
                        definition and definition.requires_per_call_approval
                    )
                    fingerprint = definition.approval_fingerprint() if definition else ""
                    can_grant = bool(
                        definition
                        and not definition.requires_per_call_approval
                        and definition.allows_standing_grant
                    )
                    session_approved = can_grant and self.grants.allows(
                        grant_scope, call.name, fingerprint
                    )
                    requires_approval = bool(
                        definition
                        and tool_requires_approval(definition, context)
                        and not session_approved
                    )
                    proposed = self._tool_state(
                        call,
                        "proposed",
                        requires_approval=requires_approval,
                        requires_per_call_approval=requires_per_call_approval,
                    )
                    yield ChatStreamEvent("tool_call", proposed)

                    if cancellation.is_set():
                        cancelled = self._tool_state(call, "cancelled")
                        run.record_tool_call(cancelled)
                        yield ChatStreamEvent("tool_call", cancelled)
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return

                    denial_message: str | None = None
                    if pending_input.is_set():
                        denial_message = "New user instructions superseded this unstarted call."
                    elif runtime and runtime.state.goal_status != "active":
                        denial_message = (
                            "The goal has yielded or completed; no further calls will run."
                        )
                    if denial_message is None and call.name in denied_tools:
                        denial_message = DENIED_MESSAGE
                    approval_warning: str | None = None
                    invocation_approved = session_approved
                    expected_fingerprint = fingerprint if session_approved else None
                    if requires_approval and denial_message is None:
                        if not interactive:
                            denial_message = UNATTENDED_DENY_MESSAGE
                        else:
                            assert definition is not None
                            pending = self.approvals.request(
                                run.run_id,
                                call.id,
                                call.name,
                                workspace_id=workspace_id,
                                arguments_fingerprint=tool_arguments_fingerprint(
                                    call.name,
                                    call.arguments,
                                ),
                                definition_fingerprint=fingerprint,
                                can_grant=can_grant,
                                timeout_seconds=self.approval_timeout_seconds,
                            )
                            awaiting = self._tool_state(
                                call,
                                "awaiting_approval",
                                requires_approval=True,
                                can_grant=can_grant,
                                requires_per_call_approval=requires_per_call_approval,
                            )
                            run.record_tool_call(awaiting)
                            yield ChatStreamEvent("tool_call", awaiting)
                            if yield_approval_wait:
                                while (
                                    decision := self.approvals.poll(
                                        pending, cancellation, interruption=pending_input
                                    )
                                ) is None:
                                    yield ChatStreamEvent("approval_wait", {"run_id": run.run_id})
                            else:
                                decision = self.approvals.wait(
                                    pending,
                                    self.approval_timeout_seconds,
                                    cancellation,
                                    interruption=pending_input,
                                )
                            if cancellation.is_set():
                                cancelled = self._tool_state(call, "cancelled")
                                run.record_tool_call(cancelled)
                                yield ChatStreamEvent("tool_call", cancelled)
                                yield ChatStreamEvent("cancelled", cancelled_payload())
                                return
                            if decision == "interrupted":
                                denial_message = (
                                    "New user instructions superseded this unstarted call."
                                )
                            elif decision == "deny":
                                denial_message = (
                                    TIMEOUT_MESSAGE
                                    if pending.denial_reason == "timeout"
                                    else DENIED_MESSAGE
                                )
                            else:
                                invocation_approved = True
                                expected_fingerprint = (
                                    pending.definition_fingerprint
                                    if pending.arguments_fingerprint
                                    == tool_arguments_fingerprint(call.name, call.arguments)
                                    else "invalidated"
                                )
                                if decision == "session":
                                    self.grants.grant(grant_scope, call.name, fingerprint)
                                elif decision == "always":
                                    try:
                                        self.always_allow_persister(workspace_id, call.name)
                                    except Exception:
                                        logger.exception(
                                            "Could not persist always-allow for %s",
                                            call.name,
                                        )
                                        approval_warning = (
                                            "Approved once; the standing grant could not be saved."
                                        )
                                    else:
                                        self.grants.grant(grant_scope, call.name, fingerprint)

                    if denial_message is None:
                        yield ChatStreamEvent(
                            "tool_call",
                            self._tool_state(
                                call,
                                "running",
                                requires_per_call_approval=requires_per_call_approval,
                            ),
                        )
                    if cancellation.is_set():
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return
                    if pending_input.is_set():
                        denial_message = "New user instructions superseded this unstarted call."
                    if denial_message is not None:
                        if denial_message in {
                            DENIED_MESSAGE,
                            TIMEOUT_MESSAGE,
                            UNATTENDED_DENY_MESSAGE,
                        }:
                            denied_tools.add(call.name)
                        result = ToolResult(
                            call=call,
                            status="failed",
                            content=denial_message,
                            summary="Tool call denied",
                            error="approval_timeout"
                            if denial_message == TIMEOUT_MESSAGE
                            else "approval_denied",
                        )
                    else:
                        invocation_approval = (
                            InvocationApproval(call) if invocation_approved else None
                        )
                        result = self.registry.execute(
                            call,
                            replace(
                                context,
                                invocation_approval=invocation_approval,
                            ),
                            expected_approval_fingerprint=expected_fingerprint,
                        )
                    if cancellation.is_set():
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return
                    if approval_warning:
                        result.summary = f"{result.summary or ''} {approval_warning}".strip()
                    preserve_result = result.error in {
                        "approval_denied",
                        "approval_timeout",
                        "approval_stale",
                    }
                    if not preserve_result:
                        result.content = self._bounded_result(
                            result.content, self.max_tool_result_chars_total
                        )
                    state = self._tool_state(
                        call,
                        result.status,
                        result_summary=result.summary,
                        artifact_ids=[artifact.id for artifact in result.artifacts],
                        error=result.error,
                        requires_approval=requires_approval,
                        requires_per_call_approval=requires_per_call_approval,
                    )
                    run.record_tool_call(state)
                    run.record_artifacts(result.artifacts)
                    for artifact in result.artifacts:
                        yield ChatStreamEvent("artifact", artifact)
                    yield ChatStreamEvent("tool_call", state)

                    tool_message = result.to_message()
                    tool_message.preserve_content = preserve_result
                    run.record_tool_message(tool_message)
                    if runtime and definition and not definition.approval_exempt:
                        runtime.observe(
                            call.id,
                            call.name,
                            result.status,
                            result.summary or result.content[:1000],
                        )

                    if runtime is not None and call.name == "agent.plan.update":
                        yield ChatStreamEvent(
                            "plan",
                            {
                                "tasks": runtime.snapshot()["tasks"],
                                "warning": runtime.snapshot()["decomposition_warning"],
                            },
                        )
                    if runtime is not None and call.name == "agent.goal.complete":
                        yield ChatStreamEvent("goal", runtime.snapshot())

                    if cancellation.is_set():
                        yield ChatStreamEvent("cancelled", cancelled_payload())
                        return

                prose_turns = 0 if completed_turn.tool_calls else prose_turns + 1
                if runtime and not completed_turn.tool_calls and not pending_input.is_set():
                    if (
                        not runtime.state.tasks
                        and not runtime.state.observations
                        and completed_turn.text.strip()
                    ):
                        runtime.complete(
                            completed_turn.text[:4000], ["Direct answer; no tool work required."]
                        )
                    elif prose_turns >= 2:
                        runtime.pause()

                # Close the inbox atomically at the hard budget. Accepted late
                # instructions remain queued in the checkpoint for the next run.
                if model_index + 1 == model_limit:
                    self.run_controls.seal(run.run_id, force=True)
                    if runtime:
                        runtime.finish_turn()
                        runtime.checkpoint(self._checkpoint_messages(run.model_messages))
                        yield ChatStreamEvent("goal", runtime.snapshot())
                        run.mark_model_completed()
                    elif not completed_turn.tool_calls:
                        for instruction in apply_instructions(mark_applied=False):
                            yield ChatStreamEvent("user_instruction", instruction)
                        run.mark_model_completed()
                    break

                if runtime:
                    runtime.finish_turn(self._checkpoint_messages(run.model_messages))
                instructions = apply_instructions()
                for instruction in instructions:
                    yield ChatStreamEvent("user_instruction", instruction)
                terminal = (
                    runtime.state.goal_status in {"complete", "waiting_for_user", "paused"}
                    if runtime
                    else not completed_turn.tool_calls and not instructions
                )
                if runtime:
                    yield ChatStreamEvent("goal", runtime.snapshot())
                if terminal and self.run_controls.seal(run.run_id):
                    run.mark_model_completed()
                    break
                if runtime and not completed_turn.tool_calls:
                    run.transcript.append(
                        ChatMessage(role="user", content=runtime.continuation_prompt())
                    )
                    yield ChatStreamEvent("delta", {"text": "\n\n"})

            if not run.model_completed:
                raise RuntimeError(f"Model/tool round limit exceeded ({model_limit})")

            if cancellation.is_set():
                yield ChatStreamEvent("cancelled", cancelled_payload())
                return

            final_text = run.assistant_text
            if runtime and runtime.state.goal_status == "waiting_for_user":
                final_text = runtime.state.waiting_question or final_text
            elif runtime and runtime.state.goal_status == "complete":
                final_text = final_text or runtime.state.completion_summary or ""
            elif runtime and runtime.state.goal_status == "paused":
                final_text = (
                    f"{final_text}\n\nPaused after two replies without tool progress. "
                    "Send a message to continue or redirect this goal."
                ).strip()
            if runtime is not None and runtime.state.goal_status == "budget_limited":
                final_text = (
                    f"{final_text}\n\nAgentic mode paused after "
                    f"{runtime.state.turns_used} model calls without claiming completion."
                ).strip()
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
                orchestration=runtime.snapshot() if runtime is not None else None,
                instructions=list(run_instructions),
            )
            yield ChatStreamEvent("final", completion)
            yield ChatStreamEvent(
                "done",
                {"run_id": run.run_id, "chat_id": persisted_chat_id},
            )
        except GeneratorExit:
            persist_turn("cancelled", run.assistant_text or None)
            raise
        except Exception as error:
            logger.exception("Chat run %s failed", run.run_id)
            if runtime is not None and not isinstance(error, CheckpointRecoveryError):
                runtime.fail()
            persisted_chat_id = persist_turn(
                "failed",
                run.assistant_text or None,
            )
            memory_failure = {}
            if isinstance(error, ModelMemoryError):
                memory_failure["model_load"] = error.to_status(getattr(backend, "model_id", None))
            yield ChatStreamEvent(
                "error",
                {
                    "run_id": run.run_id,
                    "chat_id": persisted_chat_id,
                    "message": self._public_error_message(error),
                    **memory_failure,
                    "code": "checkpoint_recovery_required"
                    if isinstance(error, CheckpointRecoveryError)
                    else "malformed_tool_call"
                    if isinstance(error, MalformedToolCallError)
                    else "chat_error",
                },
            )
            yield ChatStreamEvent(
                "done",
                {"run_id": run.run_id, "chat_id": persisted_chat_id},
            )
        finally:
            self.approvals.cancel_run(run.run_id)
            self.orchestration_runs.finish(run.run_id)
            self.run_controls.finish(run.run_id)
            self.grants.clear(f"workspace:{workspace_id}:run:{run.run_id}")

    def complete(self, **kwargs: Any) -> AgentCompletion:
        kwargs.setdefault("interactive", False)
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
