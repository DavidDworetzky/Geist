"""
Service layer for user settings management.
"""
from typing import Optional, Dict, Any
import logging
from app.models.database.user_settings import (
    get_user_settings,
    create_default_user_settings,
    update_user_settings,
    get_or_create_user_settings,
    UserSettingsModel
)
from app.models.database.geist_user import get_default_user
from app.models.user_settings import (
    UserSettingsResponse,
    UserSettingsUpdate,
    AgentConfigRequest,
    AgentFactoryConfig
)
from agents.factory import AgentFactory
from agents.agent_context import AgentContext

logger = logging.getLogger(__name__)

class UserSettingsService:
    """Service for managing user settings and agent configuration."""
    
    @staticmethod
    def get_user_settings_by_id(user_id: int) -> Optional[UserSettingsResponse]:
        """
        Get user settings by user ID.
        
        Args:
            user_id: User ID
            
        Returns:
            UserSettingsResponse if found, None otherwise
        """
        settings_model = get_user_settings(user_id)
        if settings_model:
            return UserSettingsResponse(
                user_settings_id=settings_model.user_settings_id,
                user_id=settings_model.user_id,
                default_agent_type=settings_model.default_agent_type,
                default_local_model=settings_model.default_local_model,
                default_online_model=settings_model.default_online_model,
                default_online_provider=settings_model.default_online_provider,
                default_file_archives=settings_model.default_file_archives,
                enable_rag_by_default=settings_model.enable_rag_by_default,
                default_max_tokens=settings_model.default_max_tokens,
                default_temperature=settings_model.default_temperature,
                default_top_p=settings_model.default_top_p,
                default_frequency_penalty=settings_model.default_frequency_penalty,
                default_presence_penalty=settings_model.default_presence_penalty,
                backup_providers=settings_model.backup_providers,
                ui_preferences=settings_model.ui_preferences,
                create_date=settings_model.create_date,
                update_date=settings_model.update_date
            )
        return None
    
    @staticmethod
    def get_or_create_user_settings_by_id(user_id: int) -> UserSettingsResponse:
        """
        Get user settings by user ID, creating default ones if they don't exist.
        
        Args:
            user_id: User ID
            
        Returns:
            UserSettingsResponse
        """
        settings_model = get_or_create_user_settings(user_id)
        return UserSettingsResponse(
            user_settings_id=settings_model.user_settings_id,
            user_id=settings_model.user_id,
            default_agent_type=settings_model.default_agent_type,
            default_local_model=settings_model.default_local_model,
            default_online_model=settings_model.default_online_model,
            default_online_provider=settings_model.default_online_provider,
            default_file_archives=settings_model.default_file_archives,
            enable_rag_by_default=settings_model.enable_rag_by_default,
            default_max_tokens=settings_model.default_max_tokens,
            default_temperature=settings_model.default_temperature,
            default_top_p=settings_model.default_top_p,
            default_frequency_penalty=settings_model.default_frequency_penalty,
            default_presence_penalty=settings_model.default_presence_penalty,
            backup_providers=settings_model.backup_providers,
            ui_preferences=settings_model.ui_preferences,
            create_date=settings_model.create_date,
            update_date=settings_model.update_date
        )
    
    @staticmethod
    def update_user_settings_by_id(user_id: int, updates: UserSettingsUpdate) -> Optional[UserSettingsResponse]:
        """
        Update user settings.

        Args:
            user_id: User ID
            updates: Settings updates

        Returns:
            Updated UserSettingsResponse if successful, None if user not found
        """
        # Convert Pydantic model to dict, excluding None values
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}

        # Backend validation: auto-infer agent_type based on model/provider changes
        # This acts as a safety net if the frontend doesn't set agent_type correctly
        if 'default_agent_type' not in update_dict or update_dict.get('default_agent_type') is None:
            # If online model or online provider is being set, infer agent_type as 'online'
            if 'default_online_model' in update_dict or 'default_online_provider' in update_dict:
                provider = update_dict.get('default_online_provider', '')
                # Only set to online if provider is not 'offline'
                if provider != 'offline':
                    update_dict['default_agent_type'] = 'online'
                    logger.info(f"Auto-inferred agent_type='online' based on online model/provider update")
            # If local model is being set, infer agent_type as 'local'
            elif 'default_local_model' in update_dict:
                update_dict['default_agent_type'] = 'local'
                logger.info(f"Auto-inferred agent_type='local' based on local model update")

        settings_model = update_user_settings(user_id, update_dict)
        if settings_model:
            return UserSettingsResponse(
                user_settings_id=settings_model.user_settings_id,
                user_id=settings_model.user_id,
                default_agent_type=settings_model.default_agent_type,
                default_local_model=settings_model.default_local_model,
                default_online_model=settings_model.default_online_model,
                default_online_provider=settings_model.default_online_provider,
                default_file_archives=settings_model.default_file_archives,
                enable_rag_by_default=settings_model.enable_rag_by_default,
                default_max_tokens=settings_model.default_max_tokens,
                default_temperature=settings_model.default_temperature,
                default_top_p=settings_model.default_top_p,
                default_frequency_penalty=settings_model.default_frequency_penalty,
                default_presence_penalty=settings_model.default_presence_penalty,
                backup_providers=settings_model.backup_providers,
                ui_preferences=settings_model.ui_preferences,
                create_date=settings_model.create_date,
                update_date=settings_model.update_date
            )
        return None
    
    @staticmethod
    def get_default_user_settings() -> UserSettingsResponse:
        """
        Get default user settings (for the default user).
        
        Returns:
            UserSettingsResponse for default user
        """
        default_user = get_default_user()
        return UserSettingsService.get_or_create_user_settings_by_id(default_user.user_id)
    
    @staticmethod
    def create_agent_from_user_settings(
        user_id: int,
        agent_context: AgentContext,
        overrides: Optional[AgentConfigRequest] = None
    ):
        """
        Create an agent instance based on user settings and optional overrides.
        
        Args:
            user_id: User ID to get settings for
            agent_context: Agent context object
            overrides: Optional configuration overrides
            
        Returns:
            Agent instance
        """
        # Get user settings
        settings = UserSettingsService.get_or_create_user_settings_by_id(user_id)
        
        # Create agent factory config
        factory_config = AgentFactoryConfig.from_user_settings(settings, overrides)
        
        logger.info(f"Creating agent with config: {factory_config}")
        
        # Create agent using factory
        agent = AgentFactory.create_agent(
            agent_type=factory_config.agent_type,
            agent_context=agent_context,
            model=factory_config.model,
            endpoint=factory_config.endpoint,
            api_key=factory_config.api_key,
            runner_type=factory_config.runner_type,
            backup_providers=[provider.dict() for provider in factory_config.backup_providers],
            **factory_config.generation_config
        )
        
        return agent
    
    @staticmethod
    def create_agent_from_default_user(
        agent_context: AgentContext,
        overrides: Optional[AgentConfigRequest] = None
    ):
        """
        Create an agent instance for the default user.
        
        Args:
            agent_context: Agent context object
            overrides: Optional configuration overrides
            
        Returns:
            Agent instance
        """
        default_user = get_default_user()
        return UserSettingsService.create_agent_from_user_settings(
            default_user.user_id,
            agent_context,
            overrides
        )
