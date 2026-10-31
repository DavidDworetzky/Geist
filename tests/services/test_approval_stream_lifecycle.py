import asyncio
import threading
from types import SimpleNamespace

import anyio

from app.services.chat_orchestrator import ChatStreamEvent


def test_approval_keepalive_releases_worker_and_disconnect_cancels(monkeypatch):
    from app import main

    cancelled = []
    closed = []
    monkeypatch.setattr(main, "get_default_workspace", lambda: SimpleNamespace(workspace_id=7))
    monkeypatch.setattr(
        main,
        "run_controls",
        SimpleNamespace(
            cancel=lambda run_id, **kwargs: cancelled.append((run_id, kwargs)),
        ),
    )

    def events(*args, **kwargs):
        assert kwargs["yield_approval_wait"] is True
        try:
            yield ChatStreamEvent("run_started", {"run_id": "run", "chat_id": 1})
            while True:
                yield ChatStreamEvent("approval_wait", {"run_id": "run"})
        finally:
            closed.append(True)

    monkeypatch.setattr(main, "stream_chat_events", events)

    async def exercise():
        stream = main.async_stream_chat_completion(main.CompleteTextParams(prompt="hello"))
        assert "run_started" in await anext(stream)
        assert await anext(stream) == ": approval pending\n\n"
        assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
        await stream.aclose()

    asyncio.run(exercise())
    assert cancelled == [("run", {"workspace_id": 7})]
    assert closed == [True]


def test_finished_stream_does_not_cancel_completed_run(monkeypatch):
    from app import main

    cancelled = threading.Event()
    monkeypatch.setattr(main, "get_default_workspace", lambda: SimpleNamespace(workspace_id=1))
    monkeypatch.setattr(
        main,
        "run_controls",
        SimpleNamespace(
            cancel=lambda *args, **kwargs: cancelled.set(),
        ),
    )
    monkeypatch.setattr(
        main,
        "stream_chat_events",
        lambda *args, **kwargs: (
            event
            for event in [
                ChatStreamEvent("run_started", {"run_id": "run", "chat_id": 1}),
                ChatStreamEvent("done", {"run_id": "run", "chat_id": 1}),
            ]
        ),
    )

    async def exercise():
        return [
            event
            async for event in main.async_stream_chat_completion(
                main.CompleteTextParams(prompt="hello")
            )
        ]

    assert len(asyncio.run(exercise())) == 2
    assert not cancelled.is_set()
