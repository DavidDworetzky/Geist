
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.utils import get_current_user
from app.models.database.job import get_job, get_jobs
from app.schemas.job import JobResponse


router = APIRouter()

# Read-only observability endpoints. Enqueueing is intentionally not exposed
# over HTTP: jobs are queued internally via app.services.job_queue.enqueue by
# feature code that enforces its own authorization (e.g. the workflow run
# endpoint's background=true path), never directly by API callers.


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_endpoint(
    job_id: int,
    current_user: dict = Depends(get_current_user),
) -> JobResponse:
    """Get the status and result of a background job."""
    job = get_job(job_id, user_id=int(current_user["user_id"]))
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobResponse(**job.to_dict())


@router.get("/", response_model=list[JobResponse])
async def list_jobs_endpoint(
    status_filter: str | None = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
) -> list[JobResponse]:
    """List background jobs, newest first, optionally filtered by status."""
    jobs = get_jobs(
        status=status_filter,
        limit=min(limit, 200),
        user_id=int(current_user["user_id"]),
    )
    return [JobResponse(**job.to_dict()) for job in jobs]
