"""Background scheduler that runs due agent routines.

A routine is a stored prompt run through the chat orchestrator on a fixed
interval with nobody watching. Runs are therefore non-interactive: the
orchestrator is invoked with ``interactive=False``, so any tool call that
would normally wait for user approval is denied immediately (the Hermes
``cron_mode: deny`` posture) instead of blocking on a user who is not there.

The scheduler is a single daemon thread polling for due routines, mirroring
the JobWorker pattern. The actual run is delegated to an injected runner
callable so the scheduler itself stays free of model/agent wiring.
"""

from __future__ import annotations

import datetime
import logging
import threading
from collections.abc import Callable

from app.models.database.agent_routine import (
    AgentRoutineModel,
    get_due_routines,
    mark_routine_run,
)


logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 60.0

RoutineRunner = Callable[[AgentRoutineModel], None]


class RoutineScheduler:
    def __init__(
        self,
        runner: RoutineRunner,
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        due_loader: Callable[[], list[AgentRoutineModel]] | None = None,
        run_marker: Callable[[int], None] | None = None,
    ):
        self.runner = runner
        self.poll_interval_seconds = poll_interval_seconds
        self.due_loader = due_loader or (
            lambda: get_due_routines(datetime.datetime.utcnow())
        )
        self.run_marker = run_marker or mark_routine_run
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def run_due_once(self) -> int:
        """Run everything currently due; returns how many routines ran.

        A routine is marked as run BEFORE the runner executes, so a crash or
        restart mid-run delays the next occurrence instead of re-firing the
        same routine in a loop.
        """
        try:
            due = self.due_loader()
        except Exception:
            logger.exception("Could not load due routines")
            return 0

        ran = 0
        for routine in due:
            if self._stop_event.is_set():
                break
            try:
                self.run_marker(routine.routine_id)
            except Exception:
                logger.exception(
                    "Could not mark routine %s as run; skipping this cycle",
                    routine.routine_id,
                )
                continue
            try:
                logger.info(
                    "Running routine %s (%r)", routine.routine_id, routine.name
                )
                self.runner(routine)
                ran += 1
            except Exception:
                logger.exception("Routine %s failed", routine.routine_id)
        return ran

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_due_once()
            self._stop_event.wait(self.poll_interval_seconds)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="geist-routine-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
