"""Fail-closed approval handoff for in-flight chat tool calls."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal


ToolApprovalDecision = Literal["approve", "session", "always", "deny"]
ApprovalOutcome = ToolApprovalDecision | Literal["interrupted"]
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0


@dataclass
class PendingToolApproval:
    run_id: str
    call_id: str
    tool_name: str
    workspace_id: int
    arguments_fingerprint: str
    definition_fingerprint: str
    can_grant: bool = False
    expires_at: float = field(
        default_factory=lambda: time.monotonic() + DEFAULT_APPROVAL_TIMEOUT_SECONDS
    )
    denial_reason: str | None = None
    event: threading.Event = field(default_factory=threading.Event)
    decision: ApprovalOutcome | None = None


class ToolApprovalRegistry:
    """Connect the blocked orchestrator worker to an approval HTTP request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], PendingToolApproval] = {}

    def request(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        *,
        workspace_id: int,
        arguments_fingerprint: str,
        definition_fingerprint: str,
        can_grant: bool = False,
        timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ) -> PendingToolApproval:
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("Approval timeout must be finite and non-negative")
        pending = PendingToolApproval(
            run_id=run_id,
            call_id=call_id,
            tool_name=tool_name,
            workspace_id=workspace_id,
            arguments_fingerprint=arguments_fingerprint,
            definition_fingerprint=definition_fingerprint,
            can_grant=can_grant,
            expires_at=time.monotonic() + max(0.0, timeout_seconds),
        )
        with self._lock:
            if (run_id, call_id) in self._pending:
                raise ValueError("A pending approval already exists for this tool call")
            if len(self._pending) >= 256:
                raise ValueError("Too many pending tool approvals")
            self._pending[(run_id, call_id)] = pending
        return pending

    def resolve(
        self,
        run_id: str,
        call_id: str,
        decision: ToolApprovalDecision,
        *,
        workspace_id: int,
    ) -> bool:
        if decision not in {"approve", "session", "always", "deny"}:
            raise ValueError("Invalid tool approval decision")
        with self._lock:
            pending = self._pending.get((run_id, call_id))
            if pending is None or pending.workspace_id != workspace_id:
                return False
            if time.monotonic() >= pending.expires_at:
                self._settle(pending, "deny", "timeout")
                return False
            if decision in {"session", "always"} and not pending.can_grant:
                raise ValueError(
                    "This tool requires invocation approval; standing grants are unavailable"
                )
            self._settle(pending, decision, "denied" if decision == "deny" else None)
            return True

    def _settle(
        self, pending: PendingToolApproval, decision: ApprovalOutcome, reason: str | None
    ) -> None:
        """Caller holds the registry lock; settlement and removal are atomic."""
        key = (pending.run_id, pending.call_id)
        if self._pending.get(key) is pending:
            self._pending.pop(key)
            pending.decision = decision
            pending.denial_reason = reason
            pending.event.set()

    def poll(
        self,
        pending: PendingToolApproval,
        cancellation: threading.Event | None = None,
        interruption: threading.Event | None = None,
    ) -> ApprovalOutcome | None:
        with self._lock:
            if cancellation is not None and cancellation.is_set():
                self._settle(pending, "deny", "cancelled")
            elif interruption is not None and interruption.is_set():
                self._settle(pending, "interrupted", "superseded")
            elif time.monotonic() >= pending.expires_at:
                self._settle(pending, "deny", "timeout")
            return pending.decision

    def wait(
        self,
        pending: PendingToolApproval,
        timeout_seconds: float,
        cancellation: threading.Event | None = None,
        interruption: threading.Event | None = None,
    ) -> ApprovalOutcome:
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("Approval timeout must be finite and non-negative")
        with self._lock:
            pending.expires_at = min(
                pending.expires_at, time.monotonic() + max(0.0, timeout_seconds)
            )
        while (decision := self.poll(pending, cancellation, interruption)) is None:
            pending.event.wait(min(0.25, max(0.0, pending.expires_at - time.monotonic())))
        return decision

    def cancel_run(self, run_id: str) -> None:
        with self._lock:
            for pending in list(self._pending.values()):
                if pending.run_id == run_id:
                    self._settle(pending, "deny", "cancelled")

    def has_pending(self, run_id: str) -> bool:
        with self._lock:
            return any(key[0] == run_id for key in self._pending)

    def pending(self) -> list[PendingToolApproval]:
        with self._lock:
            return list(self._pending.values())


class SessionGrantRegistry:
    """Bounded, expiring chat grants tied to the reviewed tool definition."""

    def __init__(
        self, *, ttl_seconds: float = 3600.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("Session grant TTL must be finite and positive")
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._grants: dict[str, tuple[float, dict[str, str]]] = {}

    def _prune(self) -> None:
        now = self._clock()
        for scope, (expires, _) in list(self._grants.items()):
            if expires <= now:
                self._grants.pop(scope, None)

    def grant(self, scope_key: str, tool_name: str, fingerprint: str = "") -> None:
        with self._lock:
            self._prune()
            if scope_key not in self._grants and len(self._grants) >= 256:
                self._grants.pop(next(iter(self._grants)))
            _, tools = self._grants.get(scope_key, (0.0, {}))
            if tool_name not in tools and len(tools) >= 256:
                raise ValueError("Too many session tool grants")
            tools[tool_name] = fingerprint
            self._grants[scope_key] = (self._clock() + self._ttl_seconds, tools)

    def allows(self, scope_key: str, tool_name: str, fingerprint: str) -> bool:
        with self._lock:
            self._prune()
            _, tools = self._grants.get(scope_key, (0.0, {}))
            return bool(fingerprint) and tools.get(tool_name) == fingerprint

    def granted(self, scope_key: str) -> frozenset[str]:
        with self._lock:
            self._prune()
            return frozenset(self._grants.get(scope_key, (0.0, {}))[1])

    def clear(self, scope_key: str) -> None:
        with self._lock:
            self._grants.pop(scope_key, None)

    def promote_run(self, workspace_id: int, run_id: str, chat_id: int) -> None:
        with self._lock:
            self._prune()
            grants = self._grants.pop(f"workspace:{workspace_id}:run:{run_id}", None)
            if grants is not None:
                self._grants[f"workspace:{workspace_id}:chat:{chat_id}"] = grants


def persist_always_allow(workspace_id: int, tool_name: str) -> None:
    """Append a reviewed static-tool grant without losing concurrent settings edits."""
    from sqlalchemy import Text, cast

    from app.models.database.database import SessionLocal
    from app.models.database.user_settings import UserSettings
    from app.services.agent_permissions import normalize_agent_permissions

    normalize_agent_permissions({"always_allow": [tool_name]})
    for _ in range(3):
        with SessionLocal() as session:
            row = (
                session.query(UserSettings, cast(UserSettings.agent_permissions, Text))
                .filter(UserSettings.user_id == workspace_id)
                .first()
            )
            if row is None:
                raise ValueError("Workspace settings are unavailable")
            settings, original_json = row
            current = normalize_agent_permissions(settings.agent_permissions)
            if tool_name in current["always_allow"]:
                return
            current["always_allow"] = sorted({*current["always_allow"], tool_name})
            updated = normalize_agent_permissions(current)
            changed = (
                session.query(UserSettings)
                .filter(
                    UserSettings.user_settings_id == settings.user_settings_id,
                    cast(UserSettings.agent_permissions, Text) == original_json,
                )
                .update({"agent_permissions": updated}, synchronize_session=False)
            )
            if changed == 1:
                session.commit()
                return
            session.rollback()
    raise RuntimeError("Permissions changed concurrently; retry the standing grant")


approval_registry = ToolApprovalRegistry()
session_grants = SessionGrantRegistry()


def tool_arguments_fingerprint(tool_name: str, arguments: dict) -> str:
    payload = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
