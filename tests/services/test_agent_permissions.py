from unittest.mock import Mock, patch

import pytest

from agents.models.tool_calling import (
    PERMISSION_MODE_AUTO_APPROVE,
    PERMISSION_MODE_DEFAULT,
    PERMISSION_MODE_REQUIRE_APPROVAL,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecutionOutput,
    tool_requires_approval,
)
from app.services.agent_permissions import (
    AgentPermissions,
    load_agent_permissions,
    normalize_agent_permissions,
)
from app.services.tool_registry import ToolRegistry, WebSearchArguments


def _context(
    mode: str = PERMISSION_MODE_DEFAULT,
    always_allow: frozenset[str] = frozenset(),
    approved_call_ids: frozenset[str] = frozenset(),
) -> ToolContext:
    return ToolContext(
        user_id=42,
        chat_id=7,
        run_id="run-test",
        approved_call_ids=approved_call_ids,
        permission_mode=mode,
        always_allow_tools=always_allow,
    )


def _definition(name: str, handler: Mock, *, requires_approval: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Test definition for {name}",
        arguments_model=WebSearchArguments,
        handler=handler,
        requires_approval=requires_approval,
        timeout_seconds=1.0,
    )


@pytest.mark.parametrize(
    ("mode", "flag", "always_allowed", "expected"),
    [
        (PERMISSION_MODE_DEFAULT, False, False, False),
        (PERMISSION_MODE_DEFAULT, True, False, True),
        (PERMISSION_MODE_DEFAULT, True, True, False),
        (PERMISSION_MODE_AUTO_APPROVE, True, False, False),
        (PERMISSION_MODE_AUTO_APPROVE, False, False, False),
        (PERMISSION_MODE_REQUIRE_APPROVAL, False, False, True),
        (PERMISSION_MODE_REQUIRE_APPROVAL, True, False, True),
        (PERMISSION_MODE_REQUIRE_APPROVAL, True, True, False),
        (PERMISSION_MODE_REQUIRE_APPROVAL, False, True, False),
    ],
)
def test_tool_requires_approval_matrix(mode, flag, always_allowed, expected):
    definition = _definition("tool.example", Mock(), requires_approval=flag)
    always_allow = frozenset({"tool.example"}) if always_allowed else frozenset()
    context = _context(mode=mode, always_allow=always_allow)
    assert tool_requires_approval(definition, context) is expected


def test_registry_auto_approve_executes_approval_gated_tool():
    handler = Mock(return_value=ToolExecutionOutput(content="sent"))
    registry = ToolRegistry()
    registry.register(_definition("email.send", handler, requires_approval=True))

    call = ToolCall.create("email.send", {"query": "hi"})
    result = registry.execute(call, _context(mode=PERMISSION_MODE_AUTO_APPROVE))

    assert result.status == "succeeded"
    handler.assert_called_once()


def test_registry_require_approval_gates_read_only_tool():
    handler = Mock(return_value=ToolExecutionOutput(content="found"))
    registry = ToolRegistry()
    registry.register(_definition("web.search", handler))

    call = ToolCall.create("web.search", {"query": "hi"})
    result = registry.execute(call, _context(mode=PERMISSION_MODE_REQUIRE_APPROVAL))

    assert result.status == "awaiting_approval"
    assert result.error == "approval_required"
    handler.assert_not_called()

    approved = registry.execute(
        call,
        _context(
            mode=PERMISSION_MODE_REQUIRE_APPROVAL,
            approved_call_ids=frozenset({call.id}),
        ),
    )
    assert approved.status == "succeeded"


def test_registry_always_allow_bypasses_per_tool_flag():
    handler = Mock(return_value=ToolExecutionOutput(content="written"))
    registry = ToolRegistry()
    registry.register(_definition("workspace.write", handler, requires_approval=True))

    call = ToolCall.create("workspace.write", {"query": "hi"})
    result = registry.execute(
        call,
        _context(always_allow=frozenset({"workspace.write"})),
    )

    assert result.status == "succeeded"
    handler.assert_called_once()


def test_normalize_agent_permissions_canonicalizes():
    normalized = normalize_agent_permissions(
        {"mode": "require_approval", "always_allow": [" b.tool", "a.tool", "a.tool"], "junk": 1}
    )
    assert normalized == {"mode": "require_approval", "always_allow": ["a.tool", "b.tool"]}


def test_normalize_agent_permissions_defaults_and_errors():
    assert normalize_agent_permissions(None) == {"mode": "default", "always_allow": []}
    assert normalize_agent_permissions({}) == {"mode": "default", "always_allow": []}
    with pytest.raises(ValueError):
        normalize_agent_permissions({"mode": "yolo"})
    with pytest.raises(ValueError):
        normalize_agent_permissions({"always_allow": "web.search"})
    with pytest.raises(ValueError):
        normalize_agent_permissions({"always_allow": ["", "web.search"]})
    with pytest.raises(ValueError):
        normalize_agent_permissions("auto_approve")


def test_load_agent_permissions_reads_stored_settings():
    stored = Mock(agent_permissions={"mode": "auto_approve", "always_allow": ["web.search"]})
    with patch(
        "app.models.database.user_settings.get_user_settings", return_value=stored
    ):
        permissions = load_agent_permissions(1)
    assert permissions == AgentPermissions(
        mode="auto_approve", always_allow=frozenset({"web.search"})
    )


def test_load_agent_permissions_falls_back_on_missing_or_malformed():
    with patch("app.models.database.user_settings.get_user_settings", return_value=None):
        assert load_agent_permissions(1) == AgentPermissions()

    stored = Mock(agent_permissions={"mode": "bogus"})
    with patch(
        "app.models.database.user_settings.get_user_settings", return_value=stored
    ):
        assert load_agent_permissions(1) == AgentPermissions()

    with patch(
        "app.models.database.user_settings.get_user_settings",
        side_effect=RuntimeError("db down"),
    ):
        assert load_agent_permissions(1) == AgentPermissions()
