"""Explicit, schema-validated chat tool registry."""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adapters.image_generation_adapter import ImageGenerationAdapter
from adapters.job_status_adapter import JobStatusAdapter
from adapters.markdown_file_adapter import MarkdownFileAdapter
from adapters.search_adapter import SearchAdapter
from agents.models.tool_calling import (
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutionOutput,
    ToolIntent,
    ToolResult,
    ToolSemanticTag,
    tool_requires_approval,
)
from app.runtime_config import default_markdown_root
from app.services.coding_workspace import CodingWorkspace, WorkspaceOperationError
from app.services.document_search import DocumentSearchService
from app.services.execution import (
    create_execution_environment,
)
from app.services.execution.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    MAX_COMMAND_TIMEOUT_SECONDS,
)
from app.services.execution.docker import DockerExecutionEnvironment
from app.services.execution.session import DockerSessionManager
from app.services.goal_runtime import GoalRuntimeRegistry


logger = logging.getLogger(__name__)


class StrictToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebSearchArguments(StrictToolArguments):
    query: str = Field(min_length=1, max_length=512)
    max_results: int = Field(default=5, ge=1, le=10)
    recency: Literal["day", "week", "month", "year"] | None = None


class DocumentSearchArguments(StrictToolArguments):
    query: str = Field(default="", max_length=512)
    limit: int = Field(default=10, ge=1, le=50)


class ImageGenerationArguments(StrictToolArguments):
    prompt: str = Field(min_length=1, max_length=4000)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1024x1024"
    quality: Literal["low", "medium", "high", "auto"] | None = None
    style: Literal["vivid", "natural"] | None = None


class MarkdownPathArguments(StrictToolArguments):
    path: str = Field(min_length=1, max_length=1024)


class MarkdownListArguments(StrictToolArguments):
    path: str | None = Field(default=None, max_length=1024)
    limit: int = Field(default=100, ge=1, le=500)


@runtime_checkable
class ToolSource(Protocol):
    """A dynamic provider of tool definitions (MCP servers, adapter bridges).

    Sources are consulted on every catalog/dispatch so their tool lists can
    change at runtime without rebuilding the registry. A failing source must
    degrade to an empty list rather than break chat.
    """

    name: str

    def definitions(self, context: ToolContext | None = None) -> list[ToolDefinition]: ...


class TerminalRunArguments(StrictToolArguments):
    command: str = Field(min_length=1, max_length=8_000)
    timeout_seconds: int = Field(
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        ge=1,
        le=MAX_COMMAND_TIMEOUT_SECONDS,
    )


class MarkdownWriteArguments(MarkdownPathArguments):
    content: str = Field(max_length=100_000)


class SmsSendArguments(StrictToolArguments):
    number: str = Field(pattern=r"^\+[1-9]\d{7,14}$")
    message: str = Field(min_length=1, max_length=1600)
    idempotency_key: str = Field(min_length=8, max_length=128)


class WorkspaceListArguments(StrictToolArguments):
    path: str = Field(default="", max_length=1024)
    pattern: str = Field(default="*", min_length=1, max_length=256)
    limit: int = Field(default=200, ge=1, le=1000)


class WorkspaceReadArguments(StrictToolArguments):
    path: str = Field(min_length=1, max_length=1024)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    max_chars: int = Field(default=100_000, ge=1, le=200_000)


class WorkspaceSearchArguments(StrictToolArguments):
    query: str = Field(min_length=1, max_length=1000)
    path: str = Field(default="", max_length=1024)
    pattern: str = Field(default="*", min_length=1, max_length=256)
    case_sensitive: bool = False
    limit: int = Field(default=100, ge=1, le=500)


class WorkspaceWriteArguments(StrictToolArguments):
    path: str = Field(min_length=1, max_length=1024)
    content: str = Field(max_length=200_000)


class WorkspaceEditArguments(StrictToolArguments):
    path: str = Field(min_length=1, max_length=1024)
    old_text: str = Field(min_length=1, max_length=100_000)
    new_text: str = Field(max_length=100_000)
    expected_replacements: int = Field(default=1, ge=1, le=100)


class EmailSendArguments(StrictToolArguments):
    to_email: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=998)
    content: str = Field(min_length=1, max_length=100_000)
    to_name: str | None = Field(default=None, max_length=200)
    idempotency_key: str = Field(min_length=8, max_length=128)


_JSON_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_argument_errors(arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Minimal structural validation for raw-JSON-schema tools.

    Required keys and primitive property types are checked here so obviously
    malformed calls fail fast with a message the model can act on; full
    constraint enforcement stays with the tool's own backend.
    """
    errors: list[str] = []
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in arguments:
            errors.append(f"Missing required argument '{name}'")
    for name, value in arguments.items():
        declared = properties.get(name)
        if not isinstance(declared, dict):
            continue
        expected_type = declared.get("type")
        if not isinstance(expected_type, str):
            continue
        expected = _JSON_TYPE_CHECKS.get(expected_type)
        if expected is None or value is None:
            continue
        if (
            isinstance(value, bool)
            and expected_type in ("integer", "number")
            or not isinstance(value, expected)
        ):
            errors.append(f"Argument '{name}' must be of type {expected_type}")
    return errors


class PlanTaskUpdate(StrictToolArguments):
    task_id: str = Field(min_length=1, max_length=64)
    status: Literal["pending", "in_progress", "completed", "blocked", "skipped"] | None = None
    title: str | None = Field(default=None, min_length=1, max_length=240)
    acceptance_criteria: list[str] | None = Field(default=None, max_length=10)
    evidence: str | None = Field(default=None, max_length=2000)
    evidence_refs: list[Annotated[str, Field(max_length=128)]] | None = Field(
        default=None, max_length=50
    )
    skip_reason: str | None = Field(default=None, max_length=2000)


class PlanUpdateArguments(StrictToolArguments):
    updates: list[PlanTaskUpdate] = Field(default_factory=list, max_length=12)
    objective: str | None = Field(default=None, min_length=1, max_length=4000)


class GoalCompleteArguments(StrictToolArguments):
    summary: str = Field(min_length=1, max_length=4000)
    evidence: list[str] = Field(min_length=1, max_length=20)
    evidence_refs: list[Annotated[str, Field(max_length=128)]] = Field(
        default_factory=list, max_length=50
    )


class GoalWaitArguments(StrictToolArguments):
    question: str = Field(min_length=1, max_length=4000)


class ToolRegistry:
    def __init__(
        self,
        explicitly_enabled: set[str] | None = None,
        max_concurrent_executions: int = 4,
    ):
        self._definitions: dict[str, ToolDefinition] = {}
        self._sources: list[ToolSource] = []
        self.session_manager: DockerSessionManager | None = None
        self._explicitly_enabled = explicitly_enabled or set()
        self._max_concurrent_executions = max_concurrent_executions
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self._pending_futures: set[Future] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_executions,
            thread_name_prefix="geist-tool",
        )

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)
        if self.session_manager is not None:
            self.session_manager.shutdown()

    def startup(self) -> None:
        with self._lifecycle_lock:
            if not self._closed:
                return
            if any(not future.done() for future in self._pending_futures):
                raise RuntimeError("Previous tool execution is still stopping; retry startup later")
            if self.session_manager is not None:
                self.session_manager.startup()
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_concurrent_executions, thread_name_prefix="geist-tool"
            )
            self._closed = False

    def _forget_future(self, future: Future) -> None:
        with self._lifecycle_lock:
            self._pending_futures.discard(future)

    def close_chat(self, workspace_id: int, chat_id: int) -> None:
        from app.services.tool_approvals import session_grants

        scope = f"workspace:{workspace_id}:chat:{chat_id}"
        session_grants.clear(scope)
        if self.session_manager is not None:
            self.session_manager.close_scope(scope)

    def add_source(self, source: ToolSource) -> None:
        if any(existing.name == source.name for existing in self._sources):
            raise ValueError(f"Tool source already registered: {source.name}")
        self._sources.append(source)

    def remove_source(self, name: str) -> None:
        self._sources = [source for source in self._sources if source.name != name]

    def _source_definitions(self, context: ToolContext | None = None) -> dict[str, ToolDefinition]:
        merged: dict[str, ToolDefinition] = {}
        for source in self._sources:
            try:
                definitions = source.definitions(context)
            except Exception:
                logger.exception("Tool source %s failed; skipping its tools", source.name)
                continue
            for definition in definitions:
                if definition.name in self._definitions or definition.name in merged:
                    logger.warning(
                        "Tool source %s tool %s collides with an existing tool; skipping",
                        source.name,
                        definition.name,
                    )
                    continue
                # A mutable source may redefine a name after it was granted.
                # Name-only grants are reserved for static built-in definitions.
                merged[definition.name] = replace(
                    definition, allows_standing_grant=False, approval_exempt=False
                )
        return merged

    def get(self, name: str, context: ToolContext | None = None) -> ToolDefinition | None:
        definition = self._definitions.get(name)
        if definition is not None:
            return definition
        if not self._sources:
            return None
        return self._source_definitions(context).get(name)

    def catalog(self, context: ToolContext | None = None) -> list[ToolDefinition]:
        return list(self._definitions.values()) + list(self._source_definitions(context).values())

    def is_enabled(self, definition: ToolDefinition) -> bool:
        return definition.enabled_by_default or definition.name in self._explicitly_enabled

    def definitions_for_context(self, context: ToolContext) -> list[ToolDefinition]:
        definitions = []
        for definition in self.catalog(context):
            enabled = self.is_enabled(definition)
            available = definition.availability is None or definition.availability(context)
            if enabled and available:
                definitions.append(definition)
        return definitions

    def definitions_for_intent(
        self,
        context: ToolContext,
        intent: ToolIntent,
        *,
        include_retrieval: bool = True,
    ) -> list[ToolDefinition]:
        definitions = self.definitions_for_context(context)
        if intent in {"answer", "sensitive_answer"} and not include_retrieval:
            return []
        if intent == "answer":
            allowed_tags: frozenset[ToolSemanticTag] = frozenset(
                {"public_retrieval", "local_retrieval"}
            )
            return [
                definition for definition in definitions if definition.semantic_tags & allowed_tags
            ]
        if intent == "sensitive_answer":
            return [
                definition
                for definition in definitions
                if "local_retrieval" in definition.semantic_tags
            ]
        if intent == "image_generation":
            return [
                definition
                for definition in definitions
                if "image_generation" in definition.semantic_tags
            ]
        return [
            definition
            for definition in definitions
            if "image_generation" not in definition.semantic_tags
        ]

    def execute(
        self,
        call: ToolCall,
        context: ToolContext,
        *,
        expected_approval_fingerprint: str | None = None,
    ) -> ToolResult:
        definition = self.get(call.name, context)
        if definition is None:
            return ToolResult(
                call=call,
                status="failed",
                content=f"Unknown tool: {call.name}",
                error="unknown_tool",
            )
        if (
            expected_approval_fingerprint is not None
            and definition.approval_fingerprint() != expected_approval_fingerprint
        ):
            return ToolResult(
                call=call,
                status="failed",
                content=(
                    "BLOCKED: the tool definition changed after approval. "
                    "Review the updated tool before trying again."
                ),
                error="approval_stale",
            )
        if not self.is_enabled(definition):
            return ToolResult(
                call=call,
                status="failed",
                content=f"Tool is not enabled: {call.name}",
                error="tool_disabled",
            )
        if definition.availability is not None and not definition.availability(context):
            return ToolResult(
                call=call,
                status="failed",
                content=f"Tool is not configured: {call.name}",
                error="tool_unavailable",
            )
        if tool_requires_approval(definition, context) and not (
            context.invocation_approval is not None and context.invocation_approval.consume(call)
        ):
            return ToolResult(
                call=call,
                status="awaiting_approval",
                content=f"Tool requires approval before execution: {call.name}",
                error="approval_required",
            )

        arguments: Any
        if definition.arguments_model is not None:
            try:
                arguments = definition.arguments_model.model_validate(call.arguments)
            except ValidationError as error:
                return ToolResult(
                    call=call,
                    status="failed",
                    content=f"Invalid arguments for {call.name}: {error}",
                    error="invalid_arguments",
                )
        else:
            errors = _schema_argument_errors(call.arguments, definition.parameters_schema())
            if errors:
                return ToolResult(
                    call=call,
                    status="failed",
                    content=f"Invalid arguments for {call.name}: {'; '.join(errors)}",
                    error="invalid_arguments",
                )
            arguments = call.arguments

        handler = definition.handler
        assert handler is not None  # guaranteed by ToolDefinition.__post_init__
        with self._lifecycle_lock:
            if self._closed:
                return ToolResult(
                    call=call,
                    status="failed",
                    content="Tool execution is stopped",
                    error="tool_execution_stopped",
                )
            future = self._executor.submit(handler, context, arguments)
            self._pending_futures.add(future)
            future.add_done_callback(self._forget_future)
        try:
            output = future.result(timeout=definition.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            return ToolResult(
                call=call,
                status="failed",
                content=f"Tool timed out after {definition.timeout_seconds:g} seconds",
                error="tool_timeout",
            )
        except WorkspaceOperationError as error:
            return ToolResult(
                call=call, status="failed", content=str(error), error="workspace_unavailable"
            )
        except Exception:
            logger.exception("Chat tool %s failed", call.name)
            return ToolResult(
                call=call,
                status="failed",
                content=f"Tool failed: {call.name}",
                error="tool_execution_failed",
            )
        content = output.content
        if len(content) > definition.max_result_chars:
            content = f"{content[:definition.max_result_chars]}\n[tool result truncated]"
        return ToolResult(
            call=call,
            status="failed" if output.error else "succeeded",
            content=content,
            summary=(output.summary or content[:500])[:500],
            artifacts=output.artifacts,
            error=output.error,
        )


def build_default_tool_registry(
    orchestration_runs: GoalRuntimeRegistry | None = None,
) -> ToolRegistry:
    explicitly_enabled = {
        name.strip()
        for name in os.getenv("GEIST_ENABLED_CHAT_TOOLS", "").split(",")
        if name.strip()
    }
    registry = ToolRegistry(explicitly_enabled=explicitly_enabled)
    search_adapter = SearchAdapter(base_url=os.getenv("WEB_SEARCH_BASE_URL"))
    image_adapter = ImageGenerationAdapter()
    markdown_adapter = MarkdownFileAdapter(file_root=str(default_markdown_root()))
    try:
        execution_environment = create_execution_environment()
        workspace = CodingWorkspace(execution_environment)
        registry.session_manager = workspace.sessions
    except ValueError as error:
        logger.error("Coding tools disabled: %s", error)
        execution_environment = None
        workspace = None

    if orchestration_runs is not None:

        def plan_update(
            context: ToolContext, arguments: PlanUpdateArguments
        ) -> ToolExecutionOutput:
            result = orchestration_runs.update_plan(
                context.run_id,
                [update.model_dump(exclude_none=True) for update in arguments.updates],
                objective=arguments.objective,
            )
            return ToolExecutionOutput(
                content=json.dumps(result, ensure_ascii=False),
                summary=(
                    "Updated agentic plan"
                    if result.get("accepted")
                    else str(result.get("error", "Plan update rejected"))
                ),
            )

        def goal_complete(
            context: ToolContext, arguments: GoalCompleteArguments
        ) -> ToolExecutionOutput:
            result = orchestration_runs.complete_goal(
                context.run_id,
                arguments.summary,
                arguments.evidence,
                arguments.evidence_refs,
            )
            return ToolExecutionOutput(
                content=json.dumps(result, ensure_ascii=False),
                summary=(
                    "Goal completion accepted"
                    if result.get("accepted")
                    else str(result.get("error", "Goal completion rejected"))
                ),
            )

        registry.register(
            ToolDefinition(
                name="agent.plan.update",
                description=(
                    "Revise the current objective when the user redirects the work, or add/revise "
                    "tasks in its plan. This preserves the goal and its budget. "
                    "New task IDs require a title; mark removed requirements skipped with a reason. "
                    "Change titles, acceptance criteria, status, or evidence as you learn. Mark a task "
                    "completed only after its acceptance criteria have direct evidence."
                ),
                arguments_model=PlanUpdateArguments,
                handler=plan_update,
                approval_exempt=True,
                availability=lambda context: context.agentic_mode,
            )
        )
        registry.register(
            ToolDefinition(
                name="agent.goal.complete",
                description=(
                    "Claim that the active goal is fully complete after auditing every "
                    "deliverable. Completion is rejected while plan tasks remain open."
                ),
                arguments_model=GoalCompleteArguments,
                handler=goal_complete,
                approval_exempt=True,
                availability=lambda context: context.agentic_mode,
            )
        )

        def goal_wait(context: ToolContext, arguments: GoalWaitArguments) -> ToolExecutionOutput:
            result = orchestration_runs.wait_for_user(context.run_id, arguments.question)
            return ToolExecutionOutput(content=json.dumps(result), summary=arguments.question)

        registry.register(
            ToolDefinition(
                name="agent.goal.wait",
                description="Pause execution to ask the user a question or request missing authorization. The next user message resumes this goal.",
                arguments_model=GoalWaitArguments,
                handler=goal_wait,
                approval_exempt=True,
                availability=lambda context: context.agentic_mode,
            )
        )

    def web_search(context: ToolContext, arguments: WebSearchArguments) -> ToolExecutionOutput:
        results = search_adapter.search(
            search_term=arguments.query,
            max_results=arguments.max_results,
            recency=arguments.recency,
        )
        return ToolExecutionOutput(
            content=json.dumps({"query": arguments.query, "results": results}, ensure_ascii=False),
            summary=f"Found {len(results)} web results for {arguments.query!r}",
        )

    def document_search(
        context: ToolContext, arguments: DocumentSearchArguments
    ) -> ToolExecutionOutput:
        results = DocumentSearchService.search(
            user_id=context.workspace_id,
            query=arguments.query,
            limit=arguments.limit,
        )
        return ToolExecutionOutput(
            content=json.dumps(
                {"query": arguments.query, "documents": results}, ensure_ascii=False
            ),
            summary=f"Found {len(results)} matching uploaded documents",
        )

    def image_generate(
        context: ToolContext, arguments: ImageGenerationArguments
    ) -> ToolExecutionOutput:
        result = image_adapter.generate_image(**arguments.model_dump(exclude_none=True))
        artifact = result.get("artifact")
        artifacts = [artifact] if artifact is not None else []
        return ToolExecutionOutput(
            content=result.get("summary") or "Image generated",
            summary=result.get("summary") or "Image generated",
            artifacts=artifacts,
        )

    def markdown_list(
        context: ToolContext, arguments: MarkdownListArguments
    ) -> ToolExecutionOutput:
        files = markdown_adapter.get_files(arguments.path)[: arguments.limit]
        return ToolExecutionOutput(
            content=json.dumps({"files": files}, ensure_ascii=False),
            summary=f"Found {len(files)} Markdown files",
        )

    def markdown_read(
        context: ToolContext, arguments: MarkdownPathArguments
    ) -> ToolExecutionOutput:
        content = markdown_adapter.read_file(arguments.path)
        return ToolExecutionOutput(content=content, summary=f"Read {arguments.path}")

    def markdown_write(
        context: ToolContext, arguments: MarkdownWriteArguments
    ) -> ToolExecutionOutput:
        written = markdown_adapter.write_file(arguments.path, arguments.content)
        if not written:
            raise RuntimeError(f"Could not write {arguments.path}")
        return ToolExecutionOutput(content="File written", summary=f"Wrote {arguments.path}")

    def workspace_list(
        context: ToolContext, arguments: WorkspaceListArguments
    ) -> ToolExecutionOutput:
        assert workspace is not None
        files = workspace.file_operation(context, "list_files", arguments.model_dump())
        return ToolExecutionOutput(
            content=json.dumps({"files": files}, ensure_ascii=False),
            summary=f"Found {len(files)} workspace files",
        )

    def workspace_read(
        context: ToolContext, arguments: WorkspaceReadArguments
    ) -> ToolExecutionOutput:
        assert workspace is not None
        result = workspace.file_operation(context, "read_file", arguments.model_dump())
        return ToolExecutionOutput(
            content=json.dumps(result, ensure_ascii=False),
            summary=(f"Read {result['path']} lines {result['start_line']}-{result['end_line']}"),
        )

    def workspace_search(
        context: ToolContext, arguments: WorkspaceSearchArguments
    ) -> ToolExecutionOutput:
        assert workspace is not None
        matches = workspace.file_operation(context, "search_text", arguments.model_dump())
        return ToolExecutionOutput(
            content=json.dumps({"matches": matches}, ensure_ascii=False),
            summary=f"Found {len(matches)} matching lines",
        )

    def workspace_write(
        context: ToolContext, arguments: WorkspaceWriteArguments
    ) -> ToolExecutionOutput:
        assert workspace is not None
        result = workspace.file_operation(context, "write_file", arguments.model_dump())
        return ToolExecutionOutput(
            content=json.dumps(result, ensure_ascii=False),
            summary=f"Wrote {result['path']}",
        )

    def workspace_edit(
        context: ToolContext, arguments: WorkspaceEditArguments
    ) -> ToolExecutionOutput:
        assert workspace is not None
        result = workspace.file_operation(context, "edit_file", arguments.model_dump())
        return ToolExecutionOutput(
            content=json.dumps(result, ensure_ascii=False),
            summary=f"Edited {result['path']} ({result['replacements']} replacement(s))",
        )

    def email_send(context: ToolContext, arguments: EmailSendArguments) -> ToolExecutionOutput:
        from adapters.sendgrid_adapter import SendGridAdapter

        api_key = os.getenv("SENDGRID_API_KEY")
        from_email = os.getenv("SENDGRID_FROM_EMAIL")
        if not api_key or not from_email:
            raise RuntimeError("SendGrid is not configured")
        adapter = SendGridAdapter(
            sendgrid_api_key=api_key,
            from_email=from_email,
            from_name=os.getenv("SENDGRID_FROM_NAME"),
        )
        result = adapter.send_email(
            to_email=arguments.to_email,
            subject=arguments.subject,
            content=arguments.content,
            to_name=arguments.to_name,
        )
        return ToolExecutionOutput(content=result, summary=result)

    def sms_send(context: ToolContext, arguments: SmsSendArguments) -> ToolExecutionOutput:
        from adapters.sms_adapter import SMSAdapter

        token = os.getenv("TWILIO_TOKEN")
        sid = os.getenv("TWILIO_SID")
        source = os.getenv("TWILIO_SOURCE")
        if not token or not sid or not source:
            raise RuntimeError("Twilio is not configured")
        adapter = SMSAdapter(twilio_key=token, twilio_sid=sid, twilio_source=source)
        message_id = adapter.send_text(message=arguments.message, number=arguments.number)
        return ToolExecutionOutput(
            content=json.dumps({"message_id": message_id}),
            summary="SMS sent",
        )

    registry.register(
        ToolDefinition(
            name="web.search",
            description=(
                "Search current public web information. Use for today's news, recent events, "
                "changing facts, schedules, prices, laws, or other time-sensitive public information."
            ),
            arguments_model=WebSearchArguments,
            handler=web_search,
            timeout_seconds=20,
            source_adapter="SearchAdapter.search",
            semantic_tags=frozenset({"public_retrieval"}),
        )
    )
    registry.register(
        ToolDefinition(
            name="documents.search",
            description=(
                "Search the current workspace's uploaded documents by filename and extracted content. "
                "Use when asked to find, list, or inspect workspace files; do not use for public web facts."
            ),
            arguments_model=DocumentSearchArguments,
            handler=document_search,
            source_adapter="DocumentSearchService.search",
            semantic_tags=frozenset({"local_retrieval"}),
        )
    )
    registry.register(
        ToolDefinition(
            name="image.generate",
            description="Generate an image from a text description when the user explicitly asks for an image.",
            arguments_model=ImageGenerationArguments,
            handler=image_generate,
            side_effect="external_write",
            timeout_seconds=120,
            source_adapter="ImageGenerationAdapter.generate_image",
            availability=lambda context: bool(image_adapter.api_key),
            semantic_tags=frozenset({"image_generation"}),
        )
    )

    registry.register(
        ToolDefinition(
            name="workspace.list_files",
            description=(
                "List source and text files beneath the coding workspace. Supports "
                "a relative directory, glob pattern, and bounded result count."
            ),
            arguments_model=WorkspaceListArguments,
            handler=workspace_list,
            source_adapter="WorkspaceFileAdapter.list_files",
            availability=lambda _: workspace is not None and workspace.available,
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.read_file",
            description=(
                "Read a bounded line range from a UTF-8 source or text file beneath "
                "the coding workspace."
            ),
            arguments_model=WorkspaceReadArguments,
            handler=workspace_read,
            source_adapter="WorkspaceFileAdapter.read_file",
            availability=lambda _: workspace is not None and workspace.available,
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.search",
            description=(
                "Search source and text files beneath the coding workspace, like a "
                "bounded rg/grep using a literal query."
            ),
            arguments_model=WorkspaceSearchArguments,
            handler=workspace_search,
            source_adapter="WorkspaceFileAdapter.search_text",
            availability=lambda _: workspace is not None and workspace.available,
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.write_file",
            description=(
                "Create or replace a UTF-8 source or text file beneath the coding "
                "workspace. Requires user approval."
            ),
            arguments_model=WorkspaceWriteArguments,
            handler=workspace_write,
            side_effect="filesystem_write",
            requires_approval=True,
            source_adapter="WorkspaceFileAdapter.write_file",
            availability=lambda _: workspace is not None and workspace.available,
            requires_per_call_approval=workspace.requires_per_call_approval if workspace else True,
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.edit_file",
            description=(
                "Replace an exact text block in a workspace file. The edit fails "
                "without changing the file unless the expected match count is exact. "
                "Requires user approval."
            ),
            arguments_model=WorkspaceEditArguments,
            handler=workspace_edit,
            side_effect="filesystem_write",
            requires_approval=True,
            source_adapter="WorkspaceFileAdapter.edit_file",
            availability=lambda _: workspace is not None and workspace.available,
            requires_per_call_approval=workspace.requires_per_call_approval if workspace else True,
        )
    )

    # Reviewed mappings that are intentionally opt-in. They are in the catalog,
    # but are not sent to models unless the server explicitly enables them.
    registry.register(
        ToolDefinition(
            name="workspace.list_markdown",
            description="List Markdown files under the configured workspace root.",
            arguments_model=MarkdownListArguments,
            handler=markdown_list,
            source_adapter="MarkdownFileAdapter.get_files",
            semantic_tags=frozenset({"local_retrieval"}),
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.read_markdown",
            description="Read a Markdown file under the configured workspace root.",
            arguments_model=MarkdownPathArguments,
            handler=markdown_read,
            source_adapter="MarkdownFileAdapter.read_file",
            semantic_tags=frozenset({"local_retrieval"}),
        )
    )
    # Native tool execution backend (GEIST_EXEC_BACKEND). Sandboxed Docker
    # runs approval-free; local or host-mounted Docker requires approval —
    # isolation and approval are two implementations of the same safety
    # budget, so a backend must hold at least one of them.
    if execution_environment is not None and workspace is not None and workspace.available:

        def terminal_run(
            context: ToolContext, arguments: TerminalRunArguments
        ) -> ToolExecutionOutput:
            assert workspace is not None
            result = workspace.run(context, arguments.command, arguments.timeout_seconds)
            summary = (
                ("blocked by policy; " if result.blocked else "")
                + f"exit {result.exit_code}"
                + (" (timed out)" if result.timed_out else "")
                + f" in {result.duration_seconds:.1f}s"
            )
            return ToolExecutionOutput(
                content=json.dumps(
                    {
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "timed_out": result.timed_out,
                        "truncated": result.truncated,
                        "blocked": result.blocked,
                    },
                    ensure_ascii=False,
                ),
                summary=summary,
                error="policy_blocked" if result.blocked else None,
            )

        def _execution_available(context: ToolContext) -> bool:
            if isinstance(execution_environment, DockerExecutionEnvironment):
                return execution_environment.is_available()
            return True

        registry.register(
            ToolDefinition(
                name="terminal.run",
                description=(
                    "Run a shell command in the configured execution backend "
                    f"({execution_environment.describe()}). Returns exit code, "
                    "stdout, and stderr."
                ),
                arguments_model=TerminalRunArguments,
                handler=terminal_run,
                side_effect="process",
                requires_approval=execution_environment.requires_per_call_approval,
                requires_per_call_approval=execution_environment.requires_per_call_approval,
                allows_standing_grant=not execution_environment.requires_per_call_approval,
                enabled_by_default=True,
                timeout_seconds=MAX_COMMAND_TIMEOUT_SECONDS + 30,
                source_adapter=f"execution.{execution_environment.name}",
                source_revision=execution_environment.describe(),
                availability=_execution_available,
            )
        )

    # Reflected adapter actions ride through the same registry as the curated
    # tools above (one registry, several sources) but stay disabled until an
    # operator opts in by name via GEIST_ENABLED_CHAT_TOOLS, e.g.
    # adapter.JobStatusAdapter.check_async_tool.
    from app.services.adapter_tool_source import AdapterToolSource

    registry.add_source(AdapterToolSource([JobStatusAdapter()]))
    return registry
