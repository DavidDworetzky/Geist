"""
API endpoints for managing provider API keys.

Keys are write-only over HTTP: responses carry configuration state and a
masked hint, never the stored key itself.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.models.database.geist_user import get_default_user
from app.services.provider_keys import (
    provider_key_statuses,
    remove_provider_key,
    store_provider_key,
)


logger = logging.getLogger(__name__)

router = APIRouter()


def get_current_user():
    """
    Get current user (placeholder - should integrate with actual auth system).
    For now, returns the default user.
    """
    return get_default_user()


class ProviderKeyStatus(BaseModel):
    """Key configuration state for one provider; never includes the raw key."""
    id: str
    name: str
    description: str
    api_key_env: str
    env_configured: bool
    has_stored_key: bool
    key_hint: str | None = None
    supports_base_url: bool = False
    base_url: str | None = None
    updated_at: str | None = None


class ProviderKeyUpdate(BaseModel):
    """Request body for storing a provider API key."""
    api_key: str = Field(min_length=1, max_length=4096)
    base_url: str | None = Field(default=None, max_length=2048)


@router.get("/", response_model=list[ProviderKeyStatus])
async def list_providers(current_user = Depends(get_current_user)):
    """List managed providers with their key configuration state."""
    try:
        return provider_key_statuses(current_user.user_id)
    except Exception as e:
        logger.error(f"Error listing provider key statuses: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.put("/{provider_id}/key", response_model=ProviderKeyStatus)
async def put_provider_key(
    provider_id: str,
    update: ProviderKeyUpdate,
    current_user = Depends(get_current_user),
):
    """Store or replace the API key for one provider."""
    try:
        store_provider_key(
            current_user.user_id,
            provider_id,
            update.api_key,
            base_url=update.base_url,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error storing provider key for '{provider_id}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    for status in provider_key_statuses(current_user.user_id):
        if status["id"] == provider_id.strip().lower():
            return status
    raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")


@router.delete("/{provider_id}/key", response_model=ProviderKeyStatus)
async def delete_provider_key(
    provider_id: str,
    current_user = Depends(get_current_user),
):
    """Remove the stored API key for one provider."""
    try:
        removed = remove_provider_key(current_user.user_id, provider_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error removing provider key for '{provider_id}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    if not removed:
        raise HTTPException(status_code=404, detail="No stored key for this provider")

    for status in provider_key_statuses(current_user.user_id):
        if status["id"] == provider_id.strip().lower():
            return status
    raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")
