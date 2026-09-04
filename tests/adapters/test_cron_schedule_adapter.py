"""Tests for the chat-callable prompt scheduling adapter."""

from adapters.cron_schedule_adapter import CronScheduleAdapter
from adapters.tool_schema import enumerate_tool_schemas


def test_cron_schedule_adapter_exposes_create_action_only():
    adapter = CronScheduleAdapter()
    schemas = enumerate_tool_schemas(adapter)

    assert adapter.enumerate_actions() == ["create_prompt_schedule"]
    assert [schema.qualified_name for schema in schemas] == [
        "CronScheduleAdapter__create_prompt_schedule"
    ]
    properties = schemas[0].parameters["properties"]
    assert "user_id" not in properties
    assert set(schemas[0].parameters["required"]) == {
        "name",
        "prompt",
        "cron_expression",
    }
