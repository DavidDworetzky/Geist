import datetime

from app.models.database.agent_routine import AgentRoutineModel
from app.services.routine_scheduler import RoutineScheduler


def _routine(routine_id=1, name="digest") -> AgentRoutineModel:
    now = datetime.datetime(2026, 8, 2, 12, 0, 0)
    return AgentRoutineModel(
        routine_id=routine_id,
        user_id=1,
        name=name,
        prompt="Summarize the week",
        interval_minutes=60,
        enabled=True,
        last_run_at=None,
        next_run_at=now,
        create_date=now,
        update_date=now,
    )


def test_run_due_once_marks_before_running():
    order = []
    scheduler = RoutineScheduler(
        runner=lambda routine: order.append(("run", routine.routine_id)),
        due_loader=lambda: [_routine(1), _routine(2)],
        run_marker=lambda routine_id: order.append(("mark", routine_id)),
    )

    ran = scheduler.run_due_once()

    assert ran == 2
    # mark-before-run: a crash mid-run delays the next occurrence instead of
    # re-firing the same routine forever.
    assert order == [("mark", 1), ("run", 1), ("mark", 2), ("run", 2)]


def test_runner_failure_does_not_stop_other_routines():
    ran = []

    def flaky_runner(routine):
        if routine.routine_id == 1:
            raise RuntimeError("model unavailable")
        ran.append(routine.routine_id)

    scheduler = RoutineScheduler(
        runner=flaky_runner,
        due_loader=lambda: [_routine(1), _routine(2)],
        run_marker=lambda routine_id: None,
    )

    assert scheduler.run_due_once() == 1
    assert ran == [2]


def test_mark_failure_skips_that_routine():
    ran = []

    def failing_marker(routine_id):
        if routine_id == 1:
            raise RuntimeError("db down")

    scheduler = RoutineScheduler(
        runner=lambda routine: ran.append(routine.routine_id),
        due_loader=lambda: [_routine(1), _routine(2)],
        run_marker=failing_marker,
    )

    assert scheduler.run_due_once() == 1
    assert ran == [2]


def test_due_loader_failure_is_contained():
    def broken_loader():
        raise RuntimeError("db down")

    scheduler = RoutineScheduler(
        runner=lambda routine: None,
        due_loader=broken_loader,
        run_marker=lambda routine_id: None,
    )
    assert scheduler.run_due_once() == 0


def test_start_stop_lifecycle():
    scheduler = RoutineScheduler(
        runner=lambda routine: None,
        poll_interval_seconds=3600,
        due_loader=lambda: [],
        run_marker=lambda routine_id: None,
    )
    scheduler.start()
    assert scheduler._thread is not None and scheduler._thread.is_alive()
    scheduler.stop(timeout=2)
    assert scheduler._thread is None
