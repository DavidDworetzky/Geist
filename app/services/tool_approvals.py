"""In-flight tool approval decisions and session-scoped grants.

The chat orchestrator blocks a run when a tool call needs approval; the
client answers through the approval endpoint with one of four decisions
(Hermes/OpenCode tiers):

- ``approve``: run this one call.
- ``session``: run it and stop asking for this tool in this chat session.
- ``always``: run it and persist the tool onto the user's always-allow list.
- ``deny``: refuse the call; the model is told not to retry.

Waiting fails closed: no decision within the timeout counts as a deny
("silence is not consent").
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

APPROVAL_DECISIONS = frozenset({"approve", "session", "always", "deny"})
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0


@dataclass
class PendingApproval:
    run_id: str
    call_id: str
    tool_name: str
    allow_persistent: bool = True
    event: threading.Event = field(default_factory=threading.Event)
    decision: str | None = None


class ToolApprovalRegistry:
    """Bridges the blocked orchestrator thread and the approval endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], PendingApproval] = {}

    def request(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        *,
        allow_persistent: bool = True,
    ) -> PendingApproval:
        pending = PendingApproval(
            run_id=run_id,
            call_id=call_id,
            tool_name=tool_name,
            allow_persistent=allow_persistent,
        )
        with self._lock:
            self._pending[(run_id, call_id)] = pending
        return pending

    def resolve(self, run_id: str, call_id: str, decision: str) -> bool:
        """Record a decision; False when nothing is waiting for this call."""
        if decision not in APPROVAL_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(APPROVAL_DECISIONS)}")
        with self._lock:
            pending = self._pending.get((run_id, call_id))
            if (
                pending is not None
                and not pending.allow_persistent
                and decision in {"session", "always"}
            ):
                raise ValueError(f"{pending.tool_name} requires an approve-once or deny decision")
            pending = self._pending.pop((run_id, call_id), None)
        if pending is None:
            return False
        pending.decision = decision
        pending.event.set()
        return True

    def wait(
        self,
        pending: PendingApproval,
        timeout_seconds: float,
        cancellation: threading.Event | None = None,
    ) -> str:
        """Block until a decision, cancellation, or timeout (both deny)."""
        remaining = timeout_seconds
        interval = 0.25
        while remaining > 0:
            if pending.event.wait(min(interval, remaining)):
                return pending.decision or "deny"
            if cancellation is not None and cancellation.is_set():
                break
            remaining -= interval
        with self._lock:
            self._pending.pop((pending.run_id, pending.call_id), None)
        return "deny"

    def cancel_run(self, run_id: str) -> None:
        """Deny everything still pending for a finished/cancelled run."""
        with self._lock:
            entries = [
                self._pending.pop(key) for key in [key for key in self._pending if key[0] == run_id]
            ]
        for pending in entries:
            pending.decision = "deny"
            pending.event.set()

    def has_pending(self, run_id: str) -> bool:
        with self._lock:
            return any(key[0] == run_id for key in self._pending)

    def pending(self) -> list[PendingApproval]:
        """Snapshot of everything currently awaiting a decision."""
        with self._lock:
            return list(self._pending.values())


class SessionGrantRegistry:
    """'Allow for this session' grants, scoped to a chat session (in-memory)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._grants: dict[str, set[str]] = {}

    def grant(self, scope_key: str, tool_name: str) -> None:
        with self._lock:
            self._grants.setdefault(scope_key, set()).add(tool_name)

    def granted(self, scope_key: str) -> frozenset[str]:
        with self._lock:
            return frozenset(self._grants.get(scope_key, set()))

    def clear(self, scope_key: str) -> None:
        with self._lock:
            self._grants.pop(scope_key, None)


# Shared singletons: the orchestrator waits on these, the API resolves them.
approval_registry = ToolApprovalRegistry()
session_grants = SessionGrantRegistry()


def persist_always_allow(user_id: int, tool_name: str) -> None:
    """Append a tool to the user's stored always-allow list ("always" tier)."""
    from app.models.database.user_settings import get_user_settings, update_user_settings
    from app.services.agent_permissions import normalize_agent_permissions

    settings = get_user_settings(user_id)
    if settings is None:
        logger.warning("Cannot persist always-allow for unknown user %s", user_id)
        return
    try:
        current = normalize_agent_permissions(settings.agent_permissions or {})
    except ValueError:
        current = {"mode": "default", "always_allow": []}
    if tool_name in current["always_allow"]:
        return
    current["always_allow"] = sorted({*current["always_allow"], tool_name})
    update_user_settings(user_id, {"agent_permissions": current})
