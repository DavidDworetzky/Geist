"""Explicit, schema-validated chat tool registry."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Literal, Protocol, runtime_checkable

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


@runtime_checkable
class ToolSource(Protocol):
    """A dynamic provider of tool definitions (MCP servers, adapter bridges).

    Sources are consulted on every catalog/dispatch so their tool lists can
    change at runtime without rebuilding the registry. A failing source must
    degrade to an empty list rather than break chat.
    """

    name: str

    def definitions(self, context: ToolContext | None = None) -> list[ToolDefinition]: ...


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


class ToolRegistry:
    def __init__(
        self,
        explicitly_enabled: set[str] | None = None,
        max_concurrent_executions: int = 4,
    ):
        self._definitions: dict[str, ToolDefinition] = {}
        self._sources: list[ToolSource] = []
        self._explicitly_enabled = explicitly_enabled or set()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_executions,
            thread_name_prefix="geist-tool",
        )

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

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
                merged[definition.name] = definition
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
        if definition.requires_approval and call.id not in context.approved_call_ids:
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
        future = self._executor.submit(handler, context, arguments)
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
    # Reflected adapter actions ride through the same registry as the curated
    # tools above (one registry, several sources) but stay disabled until an
    # operator opts in by name via GEIST_ENABLED_CHAT_TOOLS, e.g.
    # adapter.JobStatusAdapter.check_async_tool.
    from app.services.adapter_tool_source import AdapterToolSource

    registry.add_source(AdapterToolSource([JobStatusAdapter()]))
    return registry
