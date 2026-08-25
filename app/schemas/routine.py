"""DTO models for agent routines API."""

from datetime import datetime

from pydantic import BaseModel, Field


class RoutineCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=8_000)
    interval_minutes: int = Field(default=60, ge=5, le=7 * 24 * 60)
    enabled: bool = True


class RoutineUpdate(BaseModel):
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
    last_run_at: datetime | None
    next_run_at: datetime | None
    create_date: datetime
    update_date: datetime

    class Config:
        from_attributes = True
