"""API endpoints for scheduled agent routines."""

import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.utils import get_current_workspace
from app.models.database.agent_routine import (
    create_routine,
    delete_routine,
    get_routine,
    list_routines,
    schedule_routine_once,
    update_routine,
)
from app.schemas.routine import RoutineCreate, RoutineResponse, RoutineUpdate
from app.security.operator import OperatorCapability, require_operator_capability


logger = logging.getLogger(__name__)

router = APIRouter()
_require_execution = require_operator_capability(OperatorCapability.TOOLS_EXECUTE)


def _owned_routine_or_404(routine_id: int, user_id: int):
    routine = get_routine(routine_id)
    if routine is None or routine.user_id != user_id:
        raise HTTPException(status_code=404, detail="Routine not found")
    return routine


@router.get("/", response_model=list[RoutineResponse])
def get_routines(request: Request, current_workspace=Depends(get_current_workspace)):
    try:
        scheduler = request.app.state.routine_scheduler
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=scheduler.run_timeout_seconds
        )
        result = []
        for routine in list_routines(current_workspace.workspace_id):
            response = RoutineResponse.model_validate(routine)
            response.scheduler_blocked = scheduler.blocked
            if (
                routine.last_status == "running"
                and routine.last_run_at
                and routine.last_run_at < cutoff
            ):
                response.last_status = "outcome_unknown"
                response.last_error = "No confirmed outcome within the execution budget. The server may have restarted; check the chat or logs before retrying."
            result.append(response)
        return result
    except Exception as error:
        logger.error(f"Error listing routines: {error}")
        raise HTTPException(status_code=500, detail="Internal server error") from error


@router.post("/", response_model=RoutineResponse, dependencies=[Depends(_require_execution)])
def create_routine_endpoint(
    params: RoutineCreate, current_workspace=Depends(get_current_workspace)
):
    try:
        return create_routine(
            user_id=current_workspace.workspace_id,
            name=params.name,
            prompt=params.prompt,
            interval_minutes=params.interval_minutes,
            enabled=params.enabled,
        )
    except Exception as error:
        logger.error(f"Error creating routine: {error}")
        raise HTTPException(status_code=500, detail="Internal server error") from error


@router.put(
    "/{routine_id}", response_model=RoutineResponse, dependencies=[Depends(_require_execution)]
)
def update_routine_endpoint(
    routine_id: int, params: RoutineUpdate, current_workspace=Depends(get_current_workspace)
):
    _owned_routine_or_404(routine_id, current_workspace.workspace_id)
    try:
        updated = update_routine(routine_id, params.model_dump(exclude_unset=True))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if updated is None:
        raise HTTPException(status_code=404, detail="Routine not found")
    return updated


@router.delete("/{routine_id}", dependencies=[Depends(_require_execution)])
def delete_routine_endpoint(routine_id: int, current_workspace=Depends(get_current_workspace)):
    _owned_routine_or_404(routine_id, current_workspace.workspace_id)
    delete_routine(routine_id)
    return {"deleted": routine_id}


@router.post(
    "/{routine_id}/run_now",
    response_model=RoutineResponse,
    dependencies=[Depends(_require_execution)],
)
def run_routine_now(routine_id: int, current_workspace=Depends(get_current_workspace)):
    """Schedule the routine for the scheduler's next poll (within a minute)."""
    _owned_routine_or_404(routine_id, current_workspace.workspace_id)
    updated = schedule_routine_once(routine_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Routine not found")
    return updated
