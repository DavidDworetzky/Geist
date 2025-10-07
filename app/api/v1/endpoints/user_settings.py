"""
API endpoints for user settings management.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging
from app.models.user_settings import (
    UserSettingsResponse,
    UserSettingsUpdate,
    AgentConfigRequest
)
from app.services.user_settings_service import UserSettingsService
from app.models.database.geist_user import get_default_user

logger = logging.getLogger(__name__)

router = APIRouter()

def get_current_user():
    """
    Get current user (placeholder - should integrate with actual auth system).
    For now, returns the default user.
    """
    return get_default_user()

@router.get("/", response_model=UserSettingsResponse)
async def get_user_settings(current_user = Depends(get_current_user)):
    """
    Get user settings for the current user.
    
    Returns:
        UserSettingsResponse: User settings
    """
    try:
        settings = UserSettingsService.get_or_create_user_settings_by_id(current_user.user_id)
        return settings
    except Exception as e:
        logger.error(f"Error getting user settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{user_id}", response_model=UserSettingsResponse)
async def get_user_settings_by_id(user_id: int):
    """
    Get user settings by user ID.
    
    Args:
        user_id: User ID to get settings for
        
    Returns:
        UserSettingsResponse: User settings
    """
    try:
        settings = UserSettingsService.get_user_settings_by_id(user_id)
        if not settings:
            raise HTTPException(status_code=404, detail="User settings not found")
        return settings
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user settings for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/", response_model=UserSettingsResponse)
async def update_user_settings(
    updates: UserSettingsUpdate,
    current_user = Depends(get_current_user)
):
    """
    Update user settings for the current user.
    
    Args:
        updates: Settings updates
        
    Returns:
        UserSettingsResponse: Updated user settings
    """
    try:
        settings = UserSettingsService.update_user_settings_by_id(current_user.user_id, updates)
        if not settings:
            # Create settings if they don't exist
            settings = UserSettingsService.get_or_create_user_settings_by_id(current_user.user_id)
            # Try updating again
            settings = UserSettingsService.update_user_settings_by_id(current_user.user_id, updates)
        
        if not settings:
            raise HTTPException(status_code=500, detail="Failed to update user settings")
        
        return settings
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{user_id}", response_model=UserSettingsResponse)
async def update_user_settings_by_id(
    user_id: int,
    updates: UserSettingsUpdate
):
    """
    Update user settings by user ID.
    
    Args:
        user_id: User ID to update settings for
        updates: Settings updates
        
    Returns:
        UserSettingsResponse: Updated user settings
    """
    try:
        settings = UserSettingsService.update_user_settings_by_id(user_id, updates)
        if not settings:
            raise HTTPException(status_code=404, detail="User not found")
        return settings
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user settings for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/reset", response_model=UserSettingsResponse)
async def reset_user_settings(current_user = Depends(get_current_user)):
    """
    Reset user settings to defaults for the current user.
    
    Returns:
        UserSettingsResponse: Reset user settings
    """
    try:
        # Reset by updating with all default values
        default_updates = UserSettingsUpdate(
            default_agent_type="local",
            default_local_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            default_online_model="gpt-4",
            default_online_provider="openai",
            default_file_archives=[],
            enable_rag_by_default=True,
            default_max_tokens=16,
            default_temperature=1.0,
            default_top_p=1.0,
            default_frequency_penalty=0.0,
            default_presence_penalty=0.0,
            backup_providers=[],
            ui_preferences={}
        )
        
        settings = UserSettingsService.update_user_settings_by_id(current_user.user_id, default_updates)
        if not settings:
            # Create default settings if user doesn't exist
            settings = UserSettingsService.get_or_create_user_settings_by_id(current_user.user_id)
        
        return settings
    except Exception as e:
        logger.error(f"Error resetting user settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/agent-config/preview")
async def preview_agent_config(
    agent_type: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    runner_type: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """
    Preview agent configuration with optional overrides.
    
    Args:
        agent_type: Agent type override
        model: Model override
        endpoint: Endpoint override
        runner_type: Runner type override
        
    Returns:
        Dict: Preview of agent configuration that would be used
    """
    try:
        # Get user settings
        settings = UserSettingsService.get_or_create_user_settings_by_id(current_user.user_id)
        
        # Create overrides
        overrides = AgentConfigRequest(
            agent_type=agent_type,
            model=model,
            endpoint=endpoint,
            runner_type=runner_type
        )
        
        # Generate config preview
        from app.models.user_settings import AgentFactoryConfig
        factory_config = AgentFactoryConfig.from_user_settings(settings, overrides)
        
        return {
            "agent_type": factory_config.agent_type,
            "model": factory_config.model,
            "endpoint": factory_config.endpoint,
            "runner_type": factory_config.runner_type,
            "backup_providers": [provider.dict() for provider in factory_config.backup_providers],
            "generation_config": factory_config.generation_config
        }
    except Exception as e:
        logger.error(f"Error previewing agent config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
