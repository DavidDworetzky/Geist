from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import main
from app.models.completion import CompleteTextParams


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "requested,saved,expected",
    [
        (None, False, False),
        (None, True, True),
        (True, False, True),
        (False, True, False),
    ],
)
def test_chat_resolves_setting_default_but_allows_explicit_override(
    monkeypatch, streaming, requested, saved, expected
):
    monkeypatch.setattr(main, "get_default_workspace", lambda: SimpleNamespace(workspace_id=1))
    monkeypatch.setattr(main, "get_active_agent", lambda _: SimpleNamespace(stream_model_turn=True))
    monkeypatch.setattr(main, "resolved_memory_settings", lambda *_: (False, "public", None))
    monkeypatch.setattr(main, "build_memory_context", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        main.UserSettingsService,
        "get_or_create_workspace_settings_by_id",
        lambda _: SimpleNamespace(agentic_mode_enabled=saved, ui_preferences={}),
    )
    orchestrator = Mock()
    orchestrator.stream.return_value = []
    monkeypatch.setattr(main, "chat_orchestrator", orchestrator)
    params = CompleteTextParams(prompt="New objective", agentic_mode=requested, goal_action="new")
    if streaming:
        list(main.stream_chat_completion(params))
        call = orchestrator.stream.call_args
    else:
        main.run_chat_completion(params)
        call = orchestrator.complete.call_args
    assert call.kwargs["agentic_mode"] is expected
    # Older clients cannot bypass an incomplete goal with this removed field.
    assert "goal_action" not in call.kwargs


def test_routines_do_not_inherit_interactive_goal_budget(monkeypatch):
    monkeypatch.setattr(main, "get_active_agent", lambda _: SimpleNamespace(stream_model_turn=True))
    orchestrator = Mock()
    orchestrator.stream.return_value = []
    monkeypatch.setattr(main, "chat_orchestrator", orchestrator)
    main.run_routine(
        SimpleNamespace(prompt="Scheduled work", user_id=1, routine_id=1, name="Test routine")
    )
    assert orchestrator.stream.call_args.kwargs["agentic_mode"] is False
    assert orchestrator.stream.call_args.kwargs["interactive"] is False
