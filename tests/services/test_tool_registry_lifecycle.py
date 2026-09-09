import threading

import pytest

from agents.models.tool_calling import ToolCall, ToolContext, ToolDefinition, ToolExecutionOutput
from app.services.tool_registry import ToolRegistry


def test_registry_reopens_after_shutdown_without_replacing_definitions():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="test.echo",
            description="Test",
            arguments_schema={"type": "object"},
            handler=lambda *_: ToolExecutionOutput(content="ok"),
        )
    )
    try:
        for _ in range(3):
            registry.startup()
            assert registry.execute(
                ToolCall.create("test.echo", {}), ToolContext(1, 1, "run")
            ).succeeded
            registry.shutdown()
    finally:
        registry.shutdown()


def test_registry_cannot_replace_a_still_running_worker():
    release = threading.Event()
    entered = threading.Event()

    def work(*_):
        entered.set()
        release.wait(2)
        return ToolExecutionOutput(content="finished")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="test.wait",
            description="Test",
            arguments_schema={"type": "object"},
            handler=work,
            timeout_seconds=0.02,
        )
    )
    try:
        result = registry.execute(ToolCall.create("test.wait", {}), ToolContext(1, 1, "run"))
        assert entered.is_set() and result.error == "tool_timeout"
        registry.shutdown()
        with pytest.raises(RuntimeError, match="still stopping"):
            registry.startup()
    finally:
        release.set()
        registry.shutdown()
