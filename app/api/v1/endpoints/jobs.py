
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.utils import get_current_user
from app.models.database.job import get_job, get_jobs
from app.schemas.job import JobEnqueueRequest, JobResponse
from app.services.job_queue import enqueue, registered_kinds


router = APIRouter()


@router.post("/", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def enqueue_job_endpoint(
    request: JobEnqueueRequest,
    current_user: dict = Depends(get_current_user),
) -> JobResponse:
    """Queue a background job for the worker to execute."""
    if request.kind not in registered_kinds():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown job kind '{request.kind}'. Registered kinds: {registered_kinds()}",
        )
    job = enqueue(
        request.kind,
        payload=request.payload,
        max_attempts=request.max_attempts,
        delay_seconds=request.delay_seconds,
    )
    return JobResponse(**job.to_dict())


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_endpoint(
    job_id: int,
    current_user: dict = Depends(get_current_user),
) -> JobResponse:
    """Get the status and result of a job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobResponse(**job.to_dict())


@router.get("/", response_model=list[JobResponse])
async def list_jobs_endpoint(
    status_filter: str | None = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
) -> list[JobResponse]:
    """List jobs, newest first, optionally filtered by status."""
    jobs = get_jobs(status=status_filter, limit=min(limit, 200))
    return [JobResponse(**job.to_dict()) for job in jobs]
