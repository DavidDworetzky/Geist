import threading
from unittest.mock import Mock, patch

import pytest

from app.services.tool_approvals import (
    SessionGrantRegistry,
    ToolApprovalRegistry,
    persist_always_allow,
)


def test_resolve_unblocks_waiter_with_decision():
    registry = ToolApprovalRegistry()
    pending = registry.request("run_1", "call_1", "web.search")

    resolver = threading.Timer(
        0.05, lambda: registry.resolve("run_1", "call_1", "session")
    )
    resolver.start()
    decision = registry.wait(pending, timeout_seconds=2.0)
    assert decision == "session"
    assert registry.has_pending("run_1") is False


def test_wait_times_out_to_deny():
    registry = ToolApprovalRegistry()
    pending = registry.request("run_1", "call_1", "web.search")
    assert registry.wait(pending, timeout_seconds=0.05) == "deny"
    assert registry.has_pending("run_1") is False


def test_wait_cancellation_denies():
    registry = ToolApprovalRegistry()
    pending = registry.request("run_1", "call_1", "web.search")
    cancellation = threading.Event()
    cancellation.set()
    assert registry.wait(pending, timeout_seconds=5.0, cancellation=cancellation) == "deny"


def test_resolve_unknown_call_returns_false_and_bad_decision_raises():
    registry = ToolApprovalRegistry()
    assert registry.resolve("run_x", "call_x", "approve") is False
    registry.request("run_1", "call_1", "web.search")
    with pytest.raises(ValueError):
        registry.resolve("run_1", "call_1", "yolo")


def test_cancel_run_denies_all_pending():
    registry = ToolApprovalRegistry()
    first = registry.request("run_1", "call_1", "a")
    second = registry.request("run_1", "call_2", "b")
    other = registry.request("run_2", "call_1", "c")

    registry.cancel_run("run_1")
    assert first.decision == "deny" and first.event.is_set()
    assert second.decision == "deny" and second.event.is_set()
    assert other.decision is None
    assert registry.has_pending("run_2") is True


def test_session_grants_are_scoped():
    grants = SessionGrantRegistry()
    grants.grant("chat:1", "web.search")
    assert grants.granted("chat:1") == frozenset({"web.search"})
    assert grants.granted("chat:2") == frozenset()
    grants.clear("chat:1")
    assert grants.granted("chat:1") == frozenset()


def test_persist_always_allow_merges_into_settings():
    stored = Mock(agent_permissions={"mode": "default", "always_allow": ["a.tool"]})
    with (
        patch(
            "app.models.database.user_settings.get_user_settings", return_value=stored
        ),
        patch(
            "app.models.database.user_settings.update_user_settings"
        ) as mock_update,
    ):
        persist_always_allow(1, "web.search")

    mock_update.assert_called_once_with(
        1, {"agent_permissions": {"mode": "default", "always_allow": ["a.tool", "web.search"]}}
    )


def test_persist_always_allow_noop_when_already_listed():
    stored = Mock(agent_permissions={"mode": "default", "always_allow": ["web.search"]})
    with (
        patch(
            "app.models.database.user_settings.get_user_settings", return_value=stored
        ),
        patch(
            "app.models.database.user_settings.update_user_settings"
        ) as mock_update,
    ):
        persist_always_allow(1, "web.search")
    mock_update.assert_not_called()
