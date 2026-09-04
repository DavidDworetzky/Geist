from dataclasses import replace

from agents.models.tool_calling import (
    InvocationApproval,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutionOutput,
)
from app.services.tool_registry import TerminalRunArguments, ToolRegistry


def test_approval_is_bound_to_arguments_and_consumed_once():
    executed = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="terminal.run",
            description="Mock execution only",
            arguments_model=TerminalRunArguments,
            handler=lambda context, args: (
                executed.append(args.command),
                ToolExecutionOutput(content="ok"),
            )[1],
            requires_approval=True,
            requires_per_call_approval=True,
        )
    )
    call = ToolCall.create("terminal.run", {"command": "first"})
    context = ToolContext(
        user_id=1, chat_id=1, run_id="run", invocation_approval=InvocationApproval(call)
    )
    changed = replace(call, arguments={"command": "different"})
    assert registry.execute(changed, context).error == "approval_required"
    assert registry.execute(call, context).succeeded
    assert registry.execute(call, context).error == "approval_required"
    assert executed == ["first"]
    registry._executor.shutdown()


def test_legacy_model_call_id_is_not_authority():
    call = ToolCall.create("terminal.run", {"command": "first"})
    approval = InvocationApproval(call)
    assert not approval.consume(replace(call, name="another.tool"))
    assert not approval.consume(replace(call, id="another-invocation"))
    assert approval.consume(call)
