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
import math
import os
import threading
import time
from collections.abc import Callable

from app.models.database.agent_routine import (
    AgentRoutineModel,
    claim_routine_run,
    finish_routine_run,
    get_due_routines,
)


logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 60.0

RoutineRunner = Callable[[AgentRoutineModel, threading.Event], None]
RoutineClaimer = Callable[[AgentRoutineModel], bool]


class RoutineScheduler:
    def __init__(
        self,
        runner: RoutineRunner,
        *,
        poll_interval_seconds: float | None = None,
        run_timeout_seconds: float | None = None,
        enabled: bool | None = None,
        batch_size: int = 8,
        due_loader: Callable[[], list[AgentRoutineModel]] | None = None,
        routine_claimer: RoutineClaimer | None = None,
    ):
        self.runner = runner
        self.poll_interval_seconds = (
            float(os.getenv("GEIST_ROUTINE_POLL_SECONDS") or DEFAULT_POLL_INTERVAL_SECONDS)
            if poll_interval_seconds is None
            else poll_interval_seconds
        )
        self.run_timeout_seconds = (
            float(os.getenv("GEIST_ROUTINE_RUN_TIMEOUT_SECONDS") or "600")
            if run_timeout_seconds is None
            else run_timeout_seconds
        )
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.poll_interval_seconds,
                self.run_timeout_seconds,
            )
        ):
            raise ValueError("Routine scheduler deadlines must be finite and positive")
        if not 1 <= batch_size <= 64:
            raise ValueError("Routine batch size must be between 1 and 64")
        self.batch_size = batch_size
        self.enabled = (
            os.getenv("GEIST_ROUTINE_SCHEDULER_ENABLED", "1").lower() in {"1", "true", "yes"}
            if enabled is None
            else enabled
        )
        self.due_loader = due_loader or (
            lambda: get_due_routines(datetime.datetime.utcnow(), limit=self.batch_size)
        )
        self.routine_claimer = routine_claimer or claim_routine_run
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycle_lock = threading.Lock()
        self._active_run: threading.Thread | None = None
        self._run_cancellation = threading.Event()

    @property
    def blocked(self) -> bool:
        return bool(
            self._active_run and self._active_run.is_alive() and self._run_cancellation.is_set()
        )

    def _record_outcome(
        self, routine: AgentRoutineModel, status: str, error: str | None = None
    ) -> None:
        try:
            finish_routine_run(routine, status, error)
        except Exception:
            logger.exception("Could not record outcome for routine %s", routine.routine_id)

    def _execute(self, routine: AgentRoutineModel) -> bool:
        finished = threading.Event()
        succeeded = []
        cancellation = threading.Event()
        self._run_cancellation = cancellation

        def execute() -> None:
            try:
                self.runner(routine, cancellation)
                if not cancellation.is_set():
                    succeeded.append(True)
                    self._record_outcome(routine, "succeeded")
            except Exception:
                logger.exception("Routine %s failed", routine.routine_id)
                if not cancellation.is_set():
                    self._record_outcome(
                        routine, "failed", "Routine failed; check its chat history or server logs."
                    )
            finally:
                finished.set()

        self._active_run = threading.Thread(target=execute, name="geist-routine-run", daemon=True)
        if self._stop_event.is_set():
            cancellation.set()
            self._record_outcome(routine, "cancelled", "Stopped before execution began.")
            return False
        self._active_run.start()
        deadline = time.monotonic() + self.run_timeout_seconds
        while not finished.wait(min(0.1, max(0, deadline - time.monotonic()))):
            if self._stop_event.is_set() or time.monotonic() >= deadline:
                cancellation.set()
                self._record_outcome(
                    routine,
                    "cancelled" if self._stop_event.is_set() else "timed_out",
                    "Execution stopped. Further routines wait until the active worker exits.",
                )
                logger.warning(
                    "Routine %s stopped or exceeded its execution budget", routine.routine_id
                )
                return False
        self._active_run.join(timeout=0.1)
        return bool(succeeded)

    def run_due_once(self) -> int:
        if not self.enabled or not self._cycle_lock.acquire(blocking=False):
            return 0
        try:
            if self._active_run is not None and self._active_run.is_alive():
                return 0  # Never replace a wedged worker with another background run.
            return self._run_due_batch()
        finally:
            self._cycle_lock.release()

    def _run_due_batch(self) -> int:
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
        for routine in due[: self.batch_size]:
            if self._stop_event.is_set():
                break
            try:
                claimed = self.routine_claimer(routine)
            except Exception:
                logger.exception(
                    "Could not claim routine %s; skipping this cycle",
                    routine.routine_id,
                )
                continue
            if not claimed:
                logger.info(
                    "Routine %s was already claimed or changed; skipping",
                    routine.routine_id,
                )
                continue
            try:
                logger.info("Running routine %s (%r)", routine.routine_id, routine.name)
                if self._execute(routine):
                    ran += 1
                if self._active_run is not None and self._active_run.is_alive():
                    break
            except Exception:
                logger.exception("Routine %s failed", routine.routine_id)
        return ran

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_due_once()
            self._stop_event.wait(self.poll_interval_seconds)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="geist-routine-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._run_cancellation.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if not self._thread.is_alive():
                self._thread = None
