from typing import Any

from pydantic import BaseModel


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
