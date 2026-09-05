"""Coverage for the chat-callable prompt scheduling tool."""

from agents.models.tool_calling import ToolContext
from app.services.tool_registry import build_default_tool_registry


def test_schedule_creation_tool_is_available_to_chat(monkeypatch, tmp_path):
    monkeypatch.delenv("GEIST_ENABLED_CHAT_TOOLS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))

    registry = build_default_tool_registry()
    context = ToolContext(workspace_id=1, chat_id=None, run_id="schedule-tool-test")
    tool_name = "adapter.CronScheduleAdapter.create_prompt_schedule"

    assert registry.get(tool_name) is not None
    assert tool_name in {
        definition.name for definition in registry.definitions_for_context(context)
    }
