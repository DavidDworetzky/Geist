"""API endpoints for scheduled agent routines."""
import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.database.agent_routine import (
    create_routine,
    delete_routine,
    get_routine,
    list_routines,
    update_routine,
)
from app.models.database.geist_user import get_default_user
from app.schemas.routine import RoutineCreate, RoutineResponse, RoutineUpdate


logger = logging.getLogger(__name__)

router = APIRouter()


def get_current_user():
    """Placeholder auth shim, mirroring the user-settings endpoints."""
    return get_default_user()


def _owned_routine_or_404(routine_id: int, user_id: int):
    routine = get_routine(routine_id)
    if routine is None or routine.user_id != user_id:
        raise HTTPException(status_code=404, detail="Routine not found")
    return routine


@router.get("/", response_model=list[RoutineResponse])
async def get_routines(current_user=Depends(get_current_user)):
    try:
        return list_routines(current_user.user_id)
    except Exception as error:
        logger.error(f"Error listing routines: {error}")
        raise HTTPException(status_code=500, detail="Internal server error") from error


@router.post("/", response_model=RoutineResponse)
async def create_routine_endpoint(
    params: RoutineCreate, current_user=Depends(get_current_user)
):
    try:
        return create_routine(
            user_id=current_user.user_id,
            name=params.name,
            prompt=params.prompt,
            interval_minutes=params.interval_minutes,
            enabled=params.enabled,
        )
    except Exception as error:
        logger.error(f"Error creating routine: {error}")
        raise HTTPException(status_code=500, detail="Internal server error") from error


@router.put("/{routine_id}", response_model=RoutineResponse)
async def update_routine_endpoint(
    routine_id: int, params: RoutineUpdate, current_user=Depends(get_current_user)
):
    _owned_routine_or_404(routine_id, current_user.user_id)
    updated = update_routine(routine_id, params.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Routine not found")
    return updated


@router.delete("/{routine_id}")
async def delete_routine_endpoint(
    routine_id: int, current_user=Depends(get_current_user)
):
    _owned_routine_or_404(routine_id, current_user.user_id)
    delete_routine(routine_id)
    return {"deleted": routine_id}


@router.post("/{routine_id}/run_now", response_model=RoutineResponse)
async def run_routine_now(routine_id: int, current_user=Depends(get_current_user)):
    """Schedule the routine for the scheduler's next poll (within a minute)."""
    _owned_routine_or_404(routine_id, current_user.user_id)
    updated = update_routine(
        routine_id,
        {"next_run_at": datetime.datetime.utcnow(), "enabled": True},
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Routine not found")
    return updated
