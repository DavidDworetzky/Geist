"""
Marker for adapter actions that run asynchronously through the job queue.

Decorate a slow adapter action with @async_tool and the agent tool dispatcher
will enqueue it as a background job instead of executing it inline. The
dispatch immediately returns a job handle; the agent checks completion later
via JobStatusAdapter.check_async_tool(job_id=...).
"""
from collections.abc import Callable


ASYNC_TOOL_ATTR = "__geist_async_tool__"


def async_tool(fn: Callable) -> Callable:
    """Mark an adapter action for queued (asynchronous) execution."""
    setattr(fn, ASYNC_TOOL_ATTR, True)
    return fn


def is_async_tool(fn: Callable) -> bool:
    """True if the callable was marked with @async_tool."""
    return bool(getattr(fn, ASYNC_TOOL_ATTR, False))
