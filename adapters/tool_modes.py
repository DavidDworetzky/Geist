"""
Execution-mode decorators for adapter actions.

@online_tool marks an action as depending on an external hosted API;
@offline_tool marks an action as running fully locally. The marker is
declarative metadata read with tool_mode(), so agents, dispatch policy, or
configuration can select between online and offline implementations of the
same capability (e.g. Gemini vs FLUX image generation) without inspecting
adapter internals. Composes with @async_tool.
"""
from collections.abc import Callable


TOOL_MODE_ATTR = "__geist_tool_mode__"

ONLINE = "online"
OFFLINE = "offline"


def online_tool(fn: Callable) -> Callable:
    """Mark an adapter action as backed by an external hosted API."""
    setattr(fn, TOOL_MODE_ATTR, ONLINE)
    return fn


def offline_tool(fn: Callable) -> Callable:
    """Mark an adapter action as running fully locally."""
    setattr(fn, TOOL_MODE_ATTR, OFFLINE)
    return fn


def tool_mode(fn: Callable) -> str | None:
    """The declared execution mode of an action ("online" | "offline" | None)."""
    return getattr(fn, TOOL_MODE_ATTR, None)
