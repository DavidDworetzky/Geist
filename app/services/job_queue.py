"""
Compact, durable job queue on the application database.

Jobs are rows in the `job` table (see app.models.database.job). A single
worker thread claims one job at a time and dispatches it to a handler looked
up by the job's `kind`. Handlers take the payload dict and return a
JSON-serializable result. Failures requeue with exponential backoff until
max_attempts, then land in the terminal failed state.

No broker, no new dependencies: SQLite gets a safe single-writer queue, and
PostgreSQL claiming uses FOR UPDATE SKIP LOCKED so several workers can share
one queue if ever needed.
"""
import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from app.models.database.job import (
    Job,
    claim_next_job,
    enqueue_job,
    mark_job_failed,
    mark_job_succeeded,
)


logger = logging.getLogger(__name__)

JobHandler = Callable[[dict[str, Any]], Any]

_handlers: dict[str, JobHandler] = {}


def register_job_handler(kind: str, handler: JobHandler) -> None:
    """Register the callable that executes jobs of the given kind."""
    if kind in _handlers:
        logger.warning(f"Job handler for kind '{kind}' is being replaced")
    _handlers[kind] = handler


def job_handler(kind: str) -> Callable[[JobHandler], JobHandler]:
    """Decorator form of register_job_handler."""
    def decorator(handler: JobHandler) -> JobHandler:
        register_job_handler(kind, handler)
        return handler
    return decorator


def registered_kinds() -> list:
    """Kinds that currently have a handler."""
    return sorted(_handlers.keys())


def enqueue(
    kind: str,
    payload: dict[str, Any] | None = None,
    max_attempts: int = 3,
    delay_seconds: int = 0,
) -> Job:
    """Queue one job for background execution."""
    return enqueue_job(kind, payload=payload, max_attempts=max_attempts, delay_seconds=delay_seconds)


class JobWorker:
    """Single-threaded queue consumer."""

    def __init__(self, poll_interval: float = 1.0, retry_backoff_seconds: int = 30):
        self.poll_interval = poll_interval
        self.retry_backoff_seconds = retry_backoff_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> bool:
        """Claim and execute at most one job. Returns True if one was processed."""
        job = claim_next_job()
        if job is None:
            return False

        handler = _handlers.get(job.kind)
        if handler is None:
            logger.error(f"No handler registered for job kind '{job.kind}' (job_id={job.job_id})")
            mark_job_failed(
                job.job_id,
                f"No handler registered for kind '{job.kind}'",
                retry_backoff_seconds=self.retry_backoff_seconds,
            )
            return True

        try:
            result = handler(job.to_dict()["payload"])
        except Exception as e:
            logger.exception(f"Job {job.job_id} ({job.kind}) failed on attempt {job.attempts}")
            mark_job_failed(job.job_id, str(e), retry_backoff_seconds=self.retry_backoff_seconds)
            return True

        mark_job_succeeded(job.job_id, result)
        logger.info(f"Job {job.job_id} ({job.kind}) succeeded on attempt {job.attempts}")
        return True

    def _run_loop(self) -> None:
        logger.info("Job worker started")
        while not self._stop_event.is_set():
            try:
                processed = self.run_once()
            except Exception:
                # Claiming itself failed (e.g. transient DB error); back off.
                logger.exception("Job worker loop error")
                processed = False
            if not processed:
                self._stop_event.wait(self.poll_interval)
        logger.info("Job worker stopped")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="geist-job-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


_worker: JobWorker | None = None


def start_worker() -> JobWorker | None:
    """
    Start the process-wide worker thread unless disabled via
    GEIST_JOB_WORKER_ENABLED=false. Poll cadence comes from
    GEIST_JOB_POLL_INTERVAL (seconds, default 1.0).
    """
    global _worker
    enabled = os.getenv("GEIST_JOB_WORKER_ENABLED", "true").strip().lower() not in ("false", "0", "no")
    if not enabled:
        logger.info("Job worker disabled via GEIST_JOB_WORKER_ENABLED")
        return None
    if _worker is None:
        try:
            poll_interval = float(os.getenv("GEIST_JOB_POLL_INTERVAL", "1.0"))
        except ValueError:
            poll_interval = 1.0
        _worker = JobWorker(poll_interval=poll_interval)
    _worker.start()
    return _worker


def stop_worker() -> None:
    """Stop the process-wide worker thread if it is running."""
    global _worker
    if _worker is not None:
        _worker.stop()
        _worker = None


@job_handler("tool.call")
def _run_tool_call_job(payload: dict[str, Any]) -> Any:
    """
    Execute an @async_tool adapter action in the background.

    Payload: {"adapter": str, "function": str, "arguments": dict}, enqueued
    exclusively by ToolDispatcher._dispatch_async after schema validation.
    The adapter is re-initialized in the worker from the environment, and
    only public actions marked @async_tool are executed.
    """
    # Imported here to keep module import light and avoid circular imports.
    from adapters.adapter_registry import init_adapter_class
    from adapters.async_tool import is_async_tool
    from app.environment import LoadEnvironmentDictionary

    adapter_name = payload["adapter"]
    function_name = payload["function"]
    arguments = payload.get("arguments") or {}

    if function_name.startswith("_"):
        raise ValueError(f"Refusing to run non-public adapter action '{function_name}'")

    wrapper = init_adapter_class(adapter_name, LoadEnvironmentDictionary())
    if wrapper is None:
        raise ValueError(f"Adapter '{adapter_name}' could not be initialized in the worker")

    method = getattr(wrapper.instance, function_name, None)
    if not callable(method):
        raise ValueError(f"Adapter '{adapter_name}' has no callable action '{function_name}'")
    if not is_async_tool(method):
        raise ValueError(f"Adapter action '{adapter_name}.{function_name}' is not an @async_tool")

    return method(**arguments)


@job_handler("workflow.run")
def _run_workflow_job(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a stored workflow in the background.

    Payload: {"workflow_id": int, "user_id": int, "input_data": dict}.
    Returns the WorkflowRun summary so the job row carries the run outcome.
    """
    # Imported here to keep module import light and avoid circular imports.
    from sqlalchemy.orm import selectinload

    from app.models.database.database import SessionLocal
    from app.models.database.workflow import Workflow
    from app.services.workflow_execution import WorkflowExecutor

    workflow_id = payload["workflow_id"]
    user_id = payload["user_id"]
    input_data = payload.get("input_data") or {}

    with SessionLocal() as session:
        workflow = (
            session.query(Workflow)
            .options(selectinload(Workflow.steps))
            .filter(Workflow.workflow_id == workflow_id, Workflow.user_id == user_id)
            .first()
        )
        if workflow is None:
            raise ValueError(f"Workflow {workflow_id} not found for user {user_id}")
        session.expunge_all()

    run = WorkflowExecutor().execute_workflow(workflow=workflow, user_id=user_id, input_data=input_data)
    return {
        "run_id": run.run_id,
        "workflow_id": workflow_id,
        "status": run.status,
        "error_message": run.error_message,
    }
