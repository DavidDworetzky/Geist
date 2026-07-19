from typing import Any

from pydantic import BaseModel


class JobResponse(BaseModel):
    """Schema for job status responses."""
    job_id: int
    user_id: int | None = None
    kind: str
    dedupe_key: str | None = None
    payload: dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    run_after: str | None = None
    locked_at: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
