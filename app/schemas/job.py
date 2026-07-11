from typing import Any

from pydantic import BaseModel, Field


class JobEnqueueRequest(BaseModel):
    """Schema for enqueueing a background job."""
    kind: str = Field(..., description="Registered handler kind that will execute this job")
    payload: dict[str, Any] = Field(default_factory=dict, description="JSON payload passed to the handler")
    max_attempts: int = Field(3, ge=1, le=10, description="Attempts before the job is marked failed")
    delay_seconds: int = Field(0, ge=0, description="Seconds to delay the job before it becomes visible")


class JobResponse(BaseModel):
    """Schema for job status responses."""
    job_id: int
    kind: str
    payload: dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    run_after: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
