"""HTTP API for cron-scheduled prompt inference."""

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.api.utils import get_current_workspace
from app.models.database.geist_user import WorkspaceModel
from app.models.database.prompt_schedule import PromptSchedule, PromptScheduleRun
from app.schemas.prompt_schedule import (
    PromptScheduleCreate,
    PromptScheduleResponse,
    PromptScheduleRunResponse,
    PromptScheduleUpdate,
)
from app.services.prompt_scheduler import (
    create_prompt_schedule,
    delete_prompt_schedule,
    enqueue_prompt_schedule_now,
    get_prompt_schedule,
    list_prompt_schedule_runs,
    list_prompt_schedules,
    update_prompt_schedule,
)


router = APIRouter()


def _response(schedule: PromptSchedule) -> PromptScheduleResponse:
    return PromptScheduleResponse(**schedule.to_dict())


def _scheduled_for_iso(run: PromptScheduleRun) -> str:
    if run.scheduled_for is None:
        raise RuntimeError("Persisted schedule run is missing scheduled_for")
    return run.scheduled_for.isoformat()


@router.post("/", response_model=PromptScheduleResponse, status_code=status.HTTP_201_CREATED)
async def create_schedule(
    request: PromptScheduleCreate,
    current_workspace: WorkspaceModel = Depends(get_current_workspace),
) -> PromptScheduleResponse:
    """Create a recurring prompt schedule."""
    return _response(create_prompt_schedule(current_workspace.workspace_id, request))


@router.get("/", response_model=list[PromptScheduleResponse])
async def list_schedules(
    current_workspace: WorkspaceModel = Depends(get_current_workspace),
) -> list[PromptScheduleResponse]:
    """List configured schedules for the current user."""
    return [
        _response(schedule) for schedule in list_prompt_schedules(current_workspace.workspace_id)
    ]


@router.get("/{prompt_schedule_id}", response_model=PromptScheduleResponse)
async def get_schedule(
    prompt_schedule_id: int,
    current_workspace: WorkspaceModel = Depends(get_current_workspace),
) -> PromptScheduleResponse:
    """Return one configured schedule."""
    schedule = get_prompt_schedule(prompt_schedule_id, current_workspace.workspace_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Prompt schedule not found")
    return _response(schedule)


@router.patch("/{prompt_schedule_id}", response_model=PromptScheduleResponse)
async def update_schedule(
    prompt_schedule_id: int,
    request: PromptScheduleUpdate,
    current_workspace: WorkspaceModel = Depends(get_current_workspace),
) -> PromptScheduleResponse:
    """Update a schedule and recalculate its next occurrence when needed."""
    schedule = update_prompt_schedule(
        prompt_schedule_id,
        current_workspace.workspace_id,
        request,
    )
    if schedule is None:
        raise HTTPException(status_code=404, detail="Prompt schedule not found")
    return _response(schedule)


@router.delete("/{prompt_schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    prompt_schedule_id: int,
    current_workspace: WorkspaceModel = Depends(get_current_workspace),
) -> Response:
    """Delete a schedule while preserving prior run records."""
    if not delete_prompt_schedule(prompt_schedule_id, current_workspace.workspace_id):
        raise HTTPException(status_code=404, detail="Prompt schedule not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{prompt_schedule_id}/run",
    response_model=PromptScheduleRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_schedule_now(
    prompt_schedule_id: int,
    current_workspace: WorkspaceModel = Depends(get_current_workspace),
) -> PromptScheduleRunResponse:
    """Queue an immediate background run without changing the cron cadence."""
    schedule = get_prompt_schedule(prompt_schedule_id, current_workspace.workspace_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Prompt schedule not found")
    job = enqueue_prompt_schedule_now(schedule)
    return PromptScheduleRunResponse(**job.to_dict())


@router.get("/{prompt_schedule_id}/runs", response_model=list[PromptScheduleRunResponse])
async def list_schedule_runs(
    prompt_schedule_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    current_workspace: WorkspaceModel = Depends(get_current_workspace),
) -> list[PromptScheduleRunResponse]:
    """List recent automatic and manual runs for one schedule."""
    if get_prompt_schedule(prompt_schedule_id, current_workspace.workspace_id) is None:
        raise HTTPException(status_code=404, detail="Prompt schedule not found")
    return [
        PromptScheduleRunResponse(
            **job.to_dict(),
            scheduled_for=_scheduled_for_iso(run),
        )
        for job, run in list_prompt_schedule_runs(
            prompt_schedule_id,
            current_workspace.workspace_id,
            limit,
        )
    ]
