"""Agent tool for creating recurring prompt schedules from chat."""

from typing import Any, Literal

from adapters.base_adapter import BaseAdapter


class CronScheduleAdapter(BaseAdapter):
    """Create cron schedules for Geist's current workspace."""

    def enumerate_actions(self) -> list[str]:
        return ["create_prompt_schedule"]

    def create_prompt_schedule(
        self,
        name: str,
        prompt: str,
        cron_expression: str,
        timezone: str = "UTC",
        agent_type: Literal["local", "online"] | None = None,
    ) -> dict[str, Any]:
        """Schedule a prompt to run repeatedly using five-field cron syntax.

        Times are interpreted in the supplied IANA time zone. Examples include
        `0 9 * * 1-5` for weekdays at 09:00 and `30 7 * * *` for every day at
        07:30. The schedule always belongs to Geist's current workspace.
        """
        from app.models.database.geist_user import get_default_workspace
        from app.schemas.prompt_schedule import InferenceConfig, PromptScheduleCreate
        from app.services.prompt_scheduler import create_prompt_schedule

        workspace = get_default_workspace()
        request = PromptScheduleCreate(
            name=name,
            prompt=prompt,
            cron_expression=cron_expression,
            timezone=timezone,
            inference_config=InferenceConfig(agent_type=agent_type),
        )
        schedule = create_prompt_schedule(workspace.workspace_id, request)
        return {
            "prompt_schedule_id": schedule.prompt_schedule_id,
            "name": schedule.name,
            "cron_expression": schedule.cron_expression,
            "timezone": schedule.timezone,
            "next_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
            "enabled": schedule.enabled,
        }
