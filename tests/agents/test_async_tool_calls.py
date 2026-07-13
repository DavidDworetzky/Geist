"""Tests for asynchronous tool calls dispatched through the job queue."""
import importlib

import pytest

from adapters.adapter_registry import AdapterWrapper
from adapters.async_tool import async_tool, is_async_tool
from adapters.base_adapter import BaseAdapter
from adapters.job_status_adapter import JobStatusAdapter
from agents.tool_calling import ToolCall, ToolDispatcher
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.job import JobStatus, get_job, get_jobs
from app.services.job_queue import JobWorker


@pytest.fixture()
def sqlite_database(tmp_path):
    original_config = DATABASE_CONFIG
    sqlite_config = DatabaseConfig(
        provider="sqlite",
        database_url=f"sqlite:///{tmp_path / 'geist.sqlite3'}",
    )

    engine = configure_database(sqlite_config)

    importlib.import_module("app.models.database")

    Base.metadata.create_all(bind=engine)
    try:
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


class SlowFastAdapter(BaseAdapter):
    """Adapter with one async (slow) and one sync (fast) action."""

    def __init__(self):
        self.calls = []

    def enumerate_actions(self):
        return ["render", "echo"]

    @async_tool
    def render(self, prompt: str) -> dict:
        """Slow action executed through the job queue."""
        self.calls.append(prompt)
        return {"rendered": prompt}

    def echo(self, text: str) -> str:
        """Fast action executed inline."""
        self.calls.append(text)
        return text


def make_dispatcher(adapter):
    wrapper = AdapterWrapper(name="SlowFastAdapter", instance=adapter)
    return ToolDispatcher([wrapper], function_log=[])


def test_async_tool_marker():
    adapter = SlowFastAdapter()
    assert is_async_tool(adapter.render)
    assert not is_async_tool(adapter.echo)


def test_sync_tool_still_executes_inline(sqlite_database):
    adapter = SlowFastAdapter()
    dispatcher = make_dispatcher(adapter)

    result = dispatcher.dispatch(ToolCall(adapter="SlowFastAdapter", function="echo", arguments={"text": "hi"}))

    assert result.success
    assert result.result == "hi"
    assert adapter.calls == ["hi"]
    assert get_jobs() == []


def test_async_tool_returns_job_handle_without_executing(sqlite_database):
    adapter = SlowFastAdapter()
    dispatcher = make_dispatcher(adapter)

    result = dispatcher.dispatch(
        ToolCall(adapter="SlowFastAdapter", function="render", arguments={"prompt": "a cat"})
    )

    assert result.success
    assert result.result["async"] is True
    assert result.result["status"] == "queued"
    assert result.result["check_with"] == "JobStatusAdapter__check_async_tool"
    # Nothing executed inline.
    assert adapter.calls == []

    job = get_job(result.result["job_id"])
    assert job is not None
    assert job.kind == "tool.call"
    assert job.to_dict()["payload"] == {
        "adapter": "SlowFastAdapter",
        "function": "render",
        "arguments": {"prompt": "a cat"},
    }


def test_worker_executes_async_tool_and_agent_can_poll(sqlite_database, monkeypatch):
    executed = []

    class WorkerSideAdapter(SlowFastAdapter):
        @async_tool
        def render(self, prompt: str) -> dict:
            executed.append(prompt)
            return {"rendered": prompt}

        def enumerate_actions(self):
            return ["render", "echo"]

    def fake_init(classname, args):
        assert classname == "SlowFastAdapter"
        return AdapterWrapper(name=classname, instance=WorkerSideAdapter())

    monkeypatch.setattr("adapters.adapter_registry.init_adapter_class", fake_init)

    dispatcher = make_dispatcher(SlowFastAdapter())
    handle = dispatcher.dispatch(
        ToolCall(adapter="SlowFastAdapter", function="render", arguments={"prompt": "a dog"})
    )
    job_id = handle.result["job_id"]

    # Agent polls before completion: not done yet.
    checker = JobStatusAdapter()
    pending = checker.check_async_tool(job_id=job_id)
    assert pending["status"] == "queued"
    assert pending["done"] is False

    # Worker picks it up and executes the real action.
    assert JobWorker().run_once() is True
    assert executed == ["a dog"]

    finished = checker.check_async_tool(job_id=job_id)
    assert finished["status"] == JobStatus.SUCCEEDED.value
    assert finished["done"] is True
    assert finished["result"] == {"rendered": "a dog"}


def test_handler_refuses_non_async_actions(sqlite_database, monkeypatch):
    monkeypatch.setattr(
        "adapters.adapter_registry.init_adapter_class",
        lambda classname, args: AdapterWrapper(name=classname, instance=SlowFastAdapter()),
    )
    from app.services.job_queue import enqueue

    # A forged payload targeting a sync action must fail, not execute.
    job = enqueue(
        "tool.call",
        payload={"adapter": "SlowFastAdapter", "function": "echo", "arguments": {"text": "hi"}},
        max_attempts=1,
    )
    assert JobWorker().run_once() is True
    failed = get_job(job.job_id)
    assert failed.status == JobStatus.FAILED.value
    assert "not an @async_tool" in failed.error


def test_check_async_tool_unknown_job(sqlite_database):
    assert JobStatusAdapter().check_async_tool(job_id=424242) == {
        "job_id": 424242,
        "status": "not_found",
    }
