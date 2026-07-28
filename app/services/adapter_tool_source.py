"""Bridge adapter reflection schemas into the unified chat ToolRegistry.

This is the migration path off the legacy ``agents.tool_calling.ToolDispatcher``
split: instead of a second dispatch mechanism, adapter actions become ordinary
``ToolDefinition`` entries served through a registry source. Bridged tools are
disabled by default and named ``adapter.<Adapter>.<action>`` so operators must
opt in per action via ``GEIST_ENABLED_CHAT_TOOLS``.
"""

from __future__ import annotations

import json
import logging

from adapters.async_tool import is_async_tool
from adapters.base_adapter import BaseAdapter
from adapters.tool_schema import ToolSchema, enumerate_tool_schemas
from agents.models.tool_calling import ToolContext, ToolDefinition, ToolExecutionOutput


logger = logging.getLogger(__name__)


class AdapterToolSource:
    """Expose initialized adapter instances as schema-validated chat tools."""

    name = "adapters"

    def __init__(
        self,
        adapters: list[BaseAdapter],
        *,
        enabled_by_default: bool = False,
        timeout_seconds: float = 30.0,
    ):
        self._adapters = list(adapters)
        self._enabled_by_default = enabled_by_default
        self._timeout_seconds = timeout_seconds

    def definitions(self) -> list[ToolDefinition]:
        definitions: list[ToolDefinition] = []
        for adapter in self._adapters:
            try:
                schemas = enumerate_tool_schemas(adapter)
            except Exception:
                logger.exception(
                    "Could not enumerate tool schemas for adapter %s", type(adapter).__name__
                )
                continue
            for schema in schemas:
                definitions.append(self._definition(adapter, schema))
        return definitions

    def _definition(self, adapter: BaseAdapter, schema: ToolSchema) -> ToolDefinition:
        method = getattr(adapter, schema.action)

        def handler(context: ToolContext, arguments: dict) -> ToolExecutionOutput:
            if is_async_tool(method):
                output = self._enqueue_async(schema, arguments)
            else:
                output = method(**arguments)
            try:
                content = json.dumps(output, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                content = str(output)
            return ToolExecutionOutput(
                content=content,
                summary=f"Ran {schema.adapter}.{schema.action}",
            )

        return ToolDefinition(
            name=f"adapter.{schema.adapter}.{schema.action}",
            description=schema.description,
            arguments_schema=schema.parameters,
            handler=handler,
            side_effect="external_write",
            enabled_by_default=self._enabled_by_default,
            timeout_seconds=self._timeout_seconds,
            source_adapter=f"{schema.adapter}.{schema.action}",
        )

    @staticmethod
    def _enqueue_async(schema: ToolSchema, arguments: dict) -> dict:
        from app.services.job_queue import enqueue

        job = enqueue(
            "tool.call",
            payload={
                "adapter": schema.adapter,
                "function": schema.action,
                "arguments": arguments,
            },
        )
        return {
            "async": True,
            "job_id": job.job_id,
            "status": "queued",
            "check_with": "adapter.JobStatusAdapter.check_async_tool",
        }
