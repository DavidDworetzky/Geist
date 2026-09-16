import datetime
import threading
from types import SimpleNamespace

import pytest

from app.models.database.agent_routine import AgentRoutineModel
from app.services.routine_scheduler import RoutineScheduler


@pytest.fixture(autouse=True)
def enable_test_schedulers(monkeypatch):
    monkeypatch.setenv("GEIST_ROUTINE_SCHEDULER_ENABLED", "1")


def _routine(routine_id=1, name="digest", user_id=1) -> AgentRoutineModel:
    now = datetime.datetime(2026, 8, 2, 12, 0, 0)
    return AgentRoutineModel(
        routine_id=routine_id,
        user_id=user_id,
        name=name,
        prompt="Summarize the week",
        interval_minutes=60,
        enabled=True,
        last_run_at=None,
        next_run_at=now,
        create_date=now,
        update_date=now,
    )


def test_run_due_once_claims_before_running():
    order = []

    def claim(routine):
        order.append(("claim", routine.routine_id))
        return True

    scheduler = RoutineScheduler(
        runner=lambda routine, cancellation: order.append(("run", routine.routine_id)),
        due_loader=lambda: [_routine(1), _routine(2)],
        routine_claimer=claim,
    )

    ran = scheduler.run_due_once()

    assert ran == 2
    # claim-before-run: a crash mid-run delays the next occurrence instead of
    # re-firing the same routine forever.
    assert order == [("claim", 1), ("run", 1), ("claim", 2), ("run", 2)]


def test_stop_after_claim_records_cancelled_before_worker_start(monkeypatch):
    outcomes = []
    scheduler = RoutineScheduler(runner=lambda *_: pytest.fail("Started after stop"))
    monkeypatch.setattr(scheduler, "_record_outcome", lambda *args: outcomes.append(args))
    scheduler.stop()
    assert not scheduler._execute(_routine())
    assert outcomes[0][1] == "cancelled"


def test_runner_failure_does_not_stop_other_routines():
    ran = []

    def flaky_runner(routine, cancellation):
        if routine.routine_id == 1:
            raise RuntimeError("model unavailable")
        ran.append(routine.routine_id)

    scheduler = RoutineScheduler(
        runner=flaky_runner,
        due_loader=lambda: [_routine(1), _routine(2)],
        routine_claimer=lambda routine: True,
    )

    assert scheduler.run_due_once() == 1
    assert ran == [2]


def test_claim_failure_skips_that_routine():
    ran = []

    def failing_claimer(routine):
        if routine.routine_id == 1:
            raise RuntimeError("db down")
        return True

    scheduler = RoutineScheduler(
        runner=lambda routine, cancellation: ran.append(routine.routine_id),
        due_loader=lambda: [_routine(1), _routine(2)],
        routine_claimer=failing_claimer,
    )

    assert scheduler.run_due_once() == 1
    assert ran == [2]


def test_due_loader_failure_is_contained():
    def broken_loader():
        raise RuntimeError("db down")

    scheduler = RoutineScheduler(
        runner=lambda routine, cancellation: None,
        due_loader=broken_loader,
        routine_claimer=lambda routine: True,
    )
    assert scheduler.run_due_once() == 0


def test_lost_claim_does_not_run_duplicate_occurrence():
    ran = []
    scheduler = RoutineScheduler(
        runner=lambda routine, cancellation: ran.append(routine.routine_id),
        due_loader=lambda: [_routine(1)],
        routine_claimer=lambda routine: False,
    )

    assert scheduler.run_due_once() == 0
    assert ran == []


def test_start_stop_lifecycle():
    scheduler = RoutineScheduler(
        runner=lambda routine, cancellation: None,
        poll_interval_seconds=3600,
        due_loader=lambda: [],
        routine_claimer=lambda routine: True,
    )
    scheduler.start()
    assert scheduler._thread is not None and scheduler._thread.is_alive()
    scheduler.stop(timeout=2)
    assert scheduler._thread is None


def test_run_routine_uses_owning_user(monkeypatch):
    import app.main as main

    captured = {}

    class FakeAgent:
        def stream_model_turn(self):
            raise AssertionError("test seam only")

    def stream(**kwargs):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(main, "get_active_agent", lambda agent_type: FakeAgent())
    monkeypatch.setattr(main, "get_default_workspace", lambda: SimpleNamespace(workspace_id=42))
    monkeypatch.setattr(
        main.UserSettingsService,
        "get_or_create_workspace_settings_by_id",
        lambda workspace_id: SimpleNamespace(
            default_max_tokens=77,
            default_temperature=0.2,
            default_top_p=0.8,
            default_frequency_penalty=0.1,
            default_presence_penalty=0.3,
        ),
    )
    monkeypatch.setattr(main.chat_orchestrator, "stream", stream)

    main.run_routine(_routine(user_id=42))

    assert captured["workspace_id"] == 42
    assert captured["config"].max_tokens == 77
    assert captured["config"].temperature == 0.2
    assert captured["prompt"].startswith("[Scheduled routine #1: digest]")


def test_disabled_scheduler_and_bounded_batch():
    calls = []
    scheduler = RoutineScheduler(
        runner=lambda routine, cancellation: calls.append(routine.routine_id),
        due_loader=lambda: [_routine(1), _routine(2)],
        routine_claimer=lambda routine: True,
        enabled=False,
        batch_size=1,
    )
    scheduler.start()
    assert scheduler._thread is None
    assert scheduler.run_due_once() == 0
    scheduler.enabled = True
    assert scheduler.run_due_once() == 1
    assert calls == [1]


def test_timed_out_runner_is_cancelled_and_never_replaced_while_alive():
    release = threading.Event()
    cancellations = []

    def blocked(routine, cancellation):
        cancellations.append(cancellation)
        release.wait(2)

    scheduler = RoutineScheduler(
        runner=blocked,
        due_loader=lambda: [_routine(1), _routine(2)],
        routine_claimer=lambda routine: True,
        run_timeout_seconds=0.02,
    )
    try:
        assert scheduler.run_due_once() == 0
        assert cancellations[0].is_set()
        assert scheduler.blocked
        worker = scheduler._active_run
        assert scheduler.run_due_once() == 0
        assert scheduler._active_run is worker
        assert len(cancellations) == 1
    finally:
        release.set()
        scheduler._active_run.join(1)
    assert not scheduler.blocked
