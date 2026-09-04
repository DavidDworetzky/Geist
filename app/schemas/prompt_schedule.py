"""Request and response schemas for recurring prompt schedules."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.services.cron_schedule import parse_cron, validate_timezone


class InferenceConfig(BaseModel):
    """Optional per-schedule overrides; omitted values follow user settings."""

    agent_type: Literal["local", "online"] | None = None
    model: str | None = Field(default=None, max_length=255)
    runner_type: str | None = Field(default=None, max_length=100)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)


class PromptScheduleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=100000)
    cron_expression: str = Field(min_length=1, max_length=100)
    timezone: str = Field(default="UTC", min_length=1, max_length=100)
    enabled: bool = True
    inference_config: InferenceConfig = Field(default_factory=InferenceConfig)

    @field_validator("name", "prompt")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value

    @field_validator("cron_expression")
    @classmethod
    def validate_cron_expression(cls, value: str) -> str:
        return parse_cron(value).expression

    @field_validator("timezone")
    @classmethod
    def validate_timezone_name(cls, value: str) -> str:
        normalized = value.strip()
        validate_timezone(normalized)
        return normalized


class PromptScheduleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    prompt: str | None = Field(default=None, min_length=1, max_length=100000)
    cron_expression: str | None = Field(default=None, min_length=1, max_length=100)
    timezone: str | None = Field(default=None, min_length=1, max_length=100)
    enabled: bool | None = None
    inference_config: InferenceConfig | None = None

    @field_validator("name", "prompt")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value

    @field_validator("cron_expression")
    @classmethod
    def validate_optional_cron(cls, value: str | None) -> str | None:
        return parse_cron(value).expression if value is not None else None

    @field_validator("timezone")
    @classmethod
    def validate_optional_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        validate_timezone(normalized)
        return normalized


class PromptScheduleResponse(BaseModel):
    prompt_schedule_id: int
    user_id: int
    name: str
    prompt: str
    cron_expression: str
    timezone: str
    enabled: bool
    inference_config: dict[str, Any]
    next_run_at: str | None = None
    last_enqueued_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class PromptScheduleRunResponse(BaseModel):
    job_id: int
    status: str
    scheduled_for: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
