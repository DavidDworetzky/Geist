import datetime

from app.models.database.agent_routine import AgentRoutineModel
from app.services.routine_scheduler import RoutineScheduler


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
        runner=lambda routine: order.append(("run", routine.routine_id)),
        due_loader=lambda: [_routine(1), _routine(2)],
        routine_claimer=claim,
    )

    ran = scheduler.run_due_once()

    assert ran == 2
    # claim-before-run: a crash mid-run delays the next occurrence instead of
    # re-firing the same routine forever.
    assert order == [("claim", 1), ("run", 1), ("claim", 2), ("run", 2)]


def test_runner_failure_does_not_stop_other_routines():
    ran = []

    def flaky_runner(routine):
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
        runner=lambda routine: ran.append(routine.routine_id),
        due_loader=lambda: [_routine(1), _routine(2)],
        routine_claimer=failing_claimer,
    )

    assert scheduler.run_due_once() == 1
    assert ran == [2]


def test_due_loader_failure_is_contained():
    def broken_loader():
        raise RuntimeError("db down")

    scheduler = RoutineScheduler(
        runner=lambda routine: None,
        due_loader=broken_loader,
        routine_claimer=lambda routine: True,
    )
    assert scheduler.run_due_once() == 0


def test_lost_claim_does_not_run_duplicate_occurrence():
    ran = []
    scheduler = RoutineScheduler(
        runner=lambda routine: ran.append(routine.routine_id),
        due_loader=lambda: [_routine(1)],
        routine_claimer=lambda routine: False,
    )

    assert scheduler.run_due_once() == 0
    assert ran == []


def test_start_stop_lifecycle():
    scheduler = RoutineScheduler(
        runner=lambda routine: None,
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
    monkeypatch.setattr(main.chat_orchestrator, "stream", stream)

    main.run_routine(_routine(user_id=42))

    assert captured["user_id"] == 42
