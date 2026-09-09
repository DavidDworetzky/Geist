"""DTO models for agent routines API."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class RoutineCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=8_000)
    interval_minutes: int = Field(default=60, ge=5, le=7 * 24 * 60)
    enabled: bool = True


class RoutineUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    prompt: str | None = Field(default=None, min_length=1, max_length=8_000)
    interval_minutes: int | None = Field(default=None, ge=5, le=7 * 24 * 60)
    enabled: bool | None = None


class RoutineResponse(BaseModel):
    routine_id: int
    user_id: int
    name: str
    prompt: str
    interval_minutes: int
    enabled: bool
    run_once_requested: bool = False
    last_status: str | None = None
    last_error: str | None = None
    scheduler_blocked: bool = False
    last_run_at: datetime | None
    next_run_at: datetime | None
    create_date: datetime
    update_date: datetime

    @field_serializer("last_run_at", "next_run_at", "create_date", "update_date")
    def serialize_timestamp(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()

    model_config = ConfigDict(from_attributes=True)
