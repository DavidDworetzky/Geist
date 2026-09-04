import threading
from unittest.mock import Mock

import pytest

from agents.models.tool_calling import (
    InvocationApproval,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.tool_registry import (
    ToolRegistry,
    WebSearchArguments,
    build_default_tool_registry,
)


def _context(approved_call: ToolCall | None = None) -> ToolContext:
    return ToolContext(
        user_id=42,
        chat_id=7,
        run_id="run-test",
        invocation_approval=InvocationApproval(approved_call) if approved_call else None,
    )


def _definition(
    name: str,
    handler: Mock,
    *,
    enabled_by_default: bool = True,
    requires_approval: bool = False,
    timeout_seconds: float = 1.0,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Test definition for {name}",
        arguments_model=WebSearchArguments,
        handler=handler,
        enabled_by_default=enabled_by_default,
        requires_approval=requires_approval,
        timeout_seconds=timeout_seconds,
    )


def test_default_catalog_and_context_definitions(monkeypatch, tmp_path):
    monkeypatch.delenv("GEIST_ENABLED_CHAT_TOOLS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))

    registry = build_default_tool_registry()
    catalog = {definition.name: definition for definition in registry.catalog()}

    assert set(catalog) == {
        "web.search",
        "documents.search",
        "image.generate",
        "workspace.list_markdown",
        "workspace.read_markdown",
        "workspace.write_markdown",
        "workspace.list_files",
        "workspace.read_file",
        "workspace.search",
        "workspace.write_file",
        "workspace.edit_file",
        "communication.email.send",
        "communication.sms.send",
    }
    assert catalog["web.search"].enabled_by_default is True
    assert catalog["documents.search"].enabled_by_default is True
    assert catalog["image.generate"].enabled_by_default is True
    assert catalog["workspace.read_markdown"].enabled_by_default is False
    assert catalog["workspace.write_markdown"].requires_approval is True
    assert catalog["workspace.read_file"].enabled_by_default is True
    assert catalog["workspace.search"].requires_approval is False
    assert catalog["workspace.write_file"].requires_approval is True
    assert catalog["workspace.edit_file"].requires_approval is True
    assert catalog["workspace.edit_file"].requires_per_call_approval is False
    assert catalog["communication.email.send"].requires_approval is True
    assert catalog["communication.sms.send"].requires_approval is True

    available_names = {
        definition.name for definition in registry.definitions_for_context(_context())
    }
    assert available_names == {
        "web.search",
        "documents.search",
        "workspace.list_files",
        "workspace.read_file",
        "workspace.search",
        "workspace.write_file",
        "workspace.edit_file",
    }


def test_environment_can_explicitly_enable_catalog_tools(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "GEIST_ENABLED_CHAT_TOOLS",
        "workspace.list_markdown, workspace.read_markdown",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))

    registry = build_default_tool_registry()
    available_names = {
        definition.name for definition in registry.definitions_for_context(_context())
    }

    assert available_names == {
        "web.search",
        "documents.search",
        "workspace.list_files",
        "workspace.read_file",
        "workspace.search",
        "workspace.write_file",
        "workspace.edit_file",
        "workspace.list_markdown",
        "workspace.read_markdown",
    }
    assert registry.is_enabled(registry.get("workspace.write_markdown")) is False


def test_side_effect_mappings_stay_unavailable_until_approval_resume_exists(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "GEIST_ENABLED_CHAT_TOOLS",
        "workspace.write_markdown,communication.email.send,communication.sms.send",
    )
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))

    registry = build_default_tool_registry()
    available_names = {
        definition.name for definition in registry.definitions_for_context(_context())
    }

    assert "workspace.write_markdown" not in available_names
    assert "communication.email.send" not in available_names
    assert "communication.sms.send" not in available_names


def test_workspace_write_and_edit_tools_use_existing_approval_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    registry = build_default_tool_registry()

    write_call = ToolCall.create(
        "workspace.write_file",
        {"path": "src/app.py", "content": "value = 1\n"},
    )
    assert registry.execute(write_call, _context()).status == "awaiting_approval"
    assert registry.execute(write_call, _context(write_call)).status == "succeeded"

    edit_call = ToolCall.create(
        "workspace.edit_file",
        {
            "path": "src/app.py",
            "old_text": "value = 1",
            "new_text": "value = 2",
        },
    )
    assert registry.execute(edit_call, _context()).status == "awaiting_approval"
    assert registry.execute(edit_call, _context(edit_call)).status == "succeeded"
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "valid query", "unexpected": True},
        {"query": ""},
        {"query": "valid query", "max_results": 11},
    ],
)
def test_execute_rejects_invalid_or_extra_arguments(arguments):
    handler = Mock(return_value=ToolExecutionOutput(content="should not run"))
    registry = ToolRegistry()
    registry.register(_definition("strict.search", handler))

    result = registry.execute(ToolCall.create("strict.search", arguments), _context())

    assert result.status == "failed"
    assert result.error == "invalid_arguments"
    handler.assert_not_called()


def test_execute_rejects_unknown_and_disabled_tools():
    handler = Mock(return_value=ToolExecutionOutput(content="should not run"))
    registry = ToolRegistry()
    registry.register(_definition("disabled.search", handler, enabled_by_default=False))

    unknown = registry.execute(ToolCall.create("missing.search", {"query": "x"}), _context())
    disabled = registry.execute(
        ToolCall.create("disabled.search", {"query": "x"}),
        _context(),
    )

    assert unknown.status == "failed"
    assert unknown.error == "unknown_tool"
    assert disabled.status == "failed"
    assert disabled.error == "tool_disabled"
    handler.assert_not_called()


def test_execute_requires_matching_call_approval_before_running_handler():
    handler = Mock(return_value=ToolExecutionOutput(content="sent", summary="sent"))
    registry = ToolRegistry()
    registry.register(_definition("approved.search", handler, requires_approval=True))
    call = ToolCall.create("approved.search", {"query": "approved query"})

    awaiting = registry.execute(call, _context())
    succeeded = registry.execute(call, _context(call))

    assert awaiting.status == "awaiting_approval"
    assert awaiting.error == "approval_required"
    assert succeeded.status == "succeeded"
    assert succeeded.content == "sent"
    handler.assert_called_once()
    validated_arguments = handler.call_args.args[1]
    assert validated_arguments.query == "approved query"


def test_execute_returns_timeout_without_waiting_for_handler_completion():
    release_handler = threading.Event()

    def blocked_handler(context, arguments):
        release_handler.wait(timeout=1)
        return ToolExecutionOutput(content="late result")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow.search",
            description="A deliberately blocked test tool",
            arguments_model=WebSearchArguments,
            handler=blocked_handler,
            timeout_seconds=0.01,
        )
    )

    try:
        result = registry.execute(
            ToolCall.create("slow.search", {"query": "wait"}),
            _context(),
        )
    finally:
        release_handler.set()

    assert result.status == "failed"
    assert result.error == "tool_timeout"
    assert "timed out" in result.content


def test_execute_hides_handler_exception_details(caplog):
    def failing_handler(context, arguments):
        raise RuntimeError("provider secret response")

    registry = ToolRegistry()
    registry.register(_definition("failing.search", failing_handler))

    result = registry.execute(
        ToolCall.create("failing.search", {"query": "anything"}),
        _context(),
    )

    assert result.status == "failed"
    assert result.error == "tool_execution_failed"
    assert result.content == "Tool failed: failing.search"
    assert "provider secret response" not in result.content
    assert "provider secret response" in caplog.text
