"""Explicit, schema-validated chat tool registry."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adapters.image_generation_adapter import ImageGenerationAdapter
from adapters.markdown_file_adapter import MarkdownFileAdapter
from adapters.search_adapter import SearchAdapter
from agents.models.tool_calling import (
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutionOutput,
    ToolResult,
)
from app.services.document_search import DocumentSearchService


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


class MarkdownWriteArguments(MarkdownPathArguments):
    content: str = Field(max_length=100_000)


class EmailSendArguments(StrictToolArguments):
    to_email: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=998)
    content: str = Field(min_length=1, max_length=100_000)
    to_name: str | None = Field(default=None, max_length=200)
    idempotency_key: str = Field(min_length=8, max_length=128)


class SmsSendArguments(StrictToolArguments):
    number: str = Field(pattern=r"^\+[1-9]\d{7,14}$")
    message: str = Field(min_length=1, max_length=1600)
    idempotency_key: str = Field(min_length=8, max_length=128)


class ToolRegistry:
    def __init__(
        self,
        explicitly_enabled: set[str] | None = None,
        max_concurrent_executions: int = 4,
    ):
        self._definitions: dict[str, ToolDefinition] = {}
        self._explicitly_enabled = explicitly_enabled or set()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_executions,
            thread_name_prefix="geist-tool",
        )

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def catalog(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def is_enabled(self, definition: ToolDefinition) -> bool:
        return definition.enabled_by_default or definition.name in self._explicitly_enabled

    def definitions_for_context(self, context: ToolContext) -> list[ToolDefinition]:
        definitions = []
        for definition in self._definitions.values():
            enabled = self.is_enabled(definition)
            available = definition.availability is None or definition.availability(context)
            if enabled and available:
                definitions.append(definition)
        return definitions

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        definition = self.get(call.name)
        if definition is None:
            return ToolResult(
                call=call,
                status="failed",
                content=f"Unknown tool: {call.name}",
                error="unknown_tool",
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
        if definition.requires_approval and call.id not in context.approved_call_ids:
            return ToolResult(
                call=call,
                status="awaiting_approval",
                content=f"Tool requires approval before execution: {call.name}",
                error="approval_required",
            )

        try:
            arguments = definition.arguments_model.model_validate(call.arguments)
        except ValidationError as error:
            return ToolResult(
                call=call,
                status="failed",
                content=f"Invalid arguments for {call.name}: {error}",
                error="invalid_arguments",
            )

        future = self._executor.submit(definition.handler, context, arguments)
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
            status="succeeded",
            content=content,
            summary=(output.summary or content[:500])[:500],
            artifacts=output.artifacts,
        )


def build_default_tool_registry() -> ToolRegistry:
    explicitly_enabled = {
        name.strip()
        for name in os.getenv("GEIST_ENABLED_CHAT_TOOLS", "").split(",")
        if name.strip()
    }
    registry = ToolRegistry(explicitly_enabled=explicitly_enabled)
    search_adapter = SearchAdapter(base_url=os.getenv("WEB_SEARCH_BASE_URL"))
    image_adapter = ImageGenerationAdapter()
    markdown_adapter = MarkdownFileAdapter(file_root=os.getenv("GEIST_MARKDOWN_ROOT", "."))

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
            user_id=context.user_id,
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
        )
    )
    registry.register(
        ToolDefinition(
            name="documents.search",
            description=(
                "Search the current user's uploaded documents by filename and extracted content. "
                "Use when asked to find, list, or inspect the user's files; do not use for public web facts."
            ),
            arguments_model=DocumentSearchArguments,
            handler=document_search,
            source_adapter="DocumentSearchService.search",
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
            enabled_by_default=False,
            source_adapter="MarkdownFileAdapter.get_files",
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.read_markdown",
            description="Read a Markdown file under the configured workspace root.",
            arguments_model=MarkdownPathArguments,
            handler=markdown_read,
            enabled_by_default=False,
            source_adapter="MarkdownFileAdapter.read_file",
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.write_markdown",
            description="Write a Markdown file under the configured workspace root.",
            arguments_model=MarkdownWriteArguments,
            handler=markdown_write,
            side_effect="filesystem_write",
            requires_approval=True,
            enabled_by_default=False,
            source_adapter="MarkdownFileAdapter.write_file",
            # Approval/resume and durable idempotency are not implemented yet.
            # Keep the reviewed mapping visible in the catalog but unavailable
            # to model turns even if an operator enables its name.
            availability=lambda context: False,
        )
    )
    registry.register(
        ToolDefinition(
            name="communication.email.send",
            description="Send an email through the configured SendGrid account.",
            arguments_model=EmailSendArguments,
            handler=email_send,
            side_effect="external_write",
            requires_approval=True,
            enabled_by_default=False,
            source_adapter="SendGridAdapter.send_email",
            availability=lambda context: False,
        )
    )
    registry.register(
        ToolDefinition(
            name="communication.sms.send",
            description="Send an SMS through the configured Twilio account.",
            arguments_model=SmsSendArguments,
            handler=sms_send,
            side_effect="external_write",
            requires_approval=True,
            enabled_by_default=False,
            source_adapter="SMSAdapter.send_text",
            availability=lambda context: False,
        )
    )
    return registry
