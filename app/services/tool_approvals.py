"""Fail-closed approval handoff for in-flight chat tool calls."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal


ToolApprovalDecision = Literal["approve", "deny"]
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0


@dataclass
class PendingToolApproval:
    run_id: str
    call_id: str
    tool_name: str
    event: threading.Event = field(default_factory=threading.Event)
    decision: ToolApprovalDecision | None = None


class ToolApprovalRegistry:
    """Connect the blocked orchestrator worker to an approval HTTP request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], PendingToolApproval] = {}

    def request(self, run_id: str, call_id: str, tool_name: str) -> PendingToolApproval:
        pending = PendingToolApproval(run_id=run_id, call_id=call_id, tool_name=tool_name)
        with self._lock:
            self._pending[(run_id, call_id)] = pending
        return pending

    def resolve(
        self,
        run_id: str,
        call_id: str,
        decision: ToolApprovalDecision,
    ) -> bool:
        if decision not in {"approve", "deny"}:
            raise ValueError("decision must be 'approve' or 'deny'")
        with self._lock:
            pending = self._pending.pop((run_id, call_id), None)
        if pending is None:
            return False
        pending.decision = decision
        pending.event.set()
        return True

    def wait(
        self,
        pending: PendingToolApproval,
        timeout_seconds: float,
        cancellation: threading.Event | None = None,
    ) -> ToolApprovalDecision:
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
        with self._lock:
            pending = [self._pending.pop(key) for key in list(self._pending) if key[0] == run_id]
        for request in pending:
            request.decision = "deny"
            request.event.set()

    def pending(self) -> list[PendingToolApproval]:
        with self._lock:
            return list(self._pending.values())


approval_registry = ToolApprovalRegistry()
