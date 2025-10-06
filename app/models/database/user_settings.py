"""
UserSettings database model for storing user preferences and default agent configurations.
"""
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import relationship
from app.models.database.database import Base, SessionLocal
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class UserSettings(Base):
    """
    Database model for user settings and agent preferences.
    """
    __tablename__ = 'user_settings'
    
    user_settings_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('geist_user.user_id'), nullable=False)
    
    # Default agent preferences
    default_agent_type = Column(String, default='local')  # 'local' or 'online'
    default_local_model = Column(String, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    default_online_model = Column(String, default='gpt-4')
    default_online_provider = Column(String, default='openai')  # 'openai', 'anthropic', 'groq', 'grok'
    
    # RAG and file settings
    default_file_archives = Column(JSON, default=list)  # List of file IDs to search by default
    enable_rag_by_default = Column(Boolean, default=True)
    
    # Model generation settings
    default_max_tokens = Column(Integer, default=16)
    default_temperature = Column(Float, default=1.0)
    default_top_p = Column(Float, default=1.0)
    default_frequency_penalty = Column(Float, default=0.0)
    default_presence_penalty = Column(Float, default=0.0)
    
    # Backup provider settings (JSON field)
    backup_providers = Column(JSON, default=list)  # List of backup provider configurations
    
    # UI preferences
    ui_preferences = Column(JSON, default=dict)  # General UI preferences
    
    # Timestamps
    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("GeistUser", backref="settings")

@dataclass
class UserSettingsModel:
    """
    Data model for user settings.
    """
    user_settings_id: int
    user_id: int
    default_agent_type: str
    default_local_model: str
    default_online_model: str
    default_online_provider: str
    default_file_archives: List[int]
    enable_rag_by_default: bool
    default_max_tokens: int
    default_temperature: float
    default_top_p: float
    default_frequency_penalty: float
    default_presence_penalty: float
    backup_providers: List[Dict[str, Any]]
    ui_preferences: Dict[str, Any]
    create_date: datetime.datetime
    update_date: datetime.datetime

def get_user_settings(user_id: int) -> Optional[UserSettingsModel]:
    """
    Get user settings by user ID.
    
    Args:
        user_id: User ID to get settings for
        
    Returns:
        UserSettingsModel if found, None otherwise
    """
    with SessionLocal() as session:
        settings = session.query(UserSettings).filter_by(user_id=user_id).first()
        if settings:
            return UserSettingsModel(
                user_settings_id=settings.user_settings_id,
                user_id=settings.user_id,
                default_agent_type=settings.default_agent_type,
                default_local_model=settings.default_local_model,
                default_online_model=settings.default_online_model,
                default_online_provider=settings.default_online_provider,
                default_file_archives=settings.default_file_archives or [],
                enable_rag_by_default=settings.enable_rag_by_default,
                default_max_tokens=settings.default_max_tokens,
                default_temperature=settings.default_temperature,
                default_top_p=settings.default_top_p,
                default_frequency_penalty=settings.default_frequency_penalty,
                default_presence_penalty=settings.default_presence_penalty,
                backup_providers=settings.backup_providers or [],
                ui_preferences=settings.ui_preferences or {},
                create_date=settings.create_date,
                update_date=settings.update_date
            )
        return None

def create_default_user_settings(user_id: int) -> UserSettingsModel:
    """
    Create default user settings for a user.
    
    Args:
        user_id: User ID to create settings for
        
    Returns:
        Created UserSettingsModel
    """
    with SessionLocal() as session:
        settings = UserSettings(
            user_id=user_id,
            default_agent_type='local',
            default_local_model='meta-llama/Meta-Llama-3.1-8B-Instruct',
            default_online_model='gpt-4',
            default_online_provider='openai',
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
        session.add(settings)
        session.commit()
        session.refresh(settings)
        
        return UserSettingsModel(
            user_settings_id=settings.user_settings_id,
            user_id=settings.user_id,
            default_agent_type=settings.default_agent_type,
            default_local_model=settings.default_local_model,
            default_online_model=settings.default_online_model,
            default_online_provider=settings.default_online_provider,
            default_file_archives=settings.default_file_archives or [],
            enable_rag_by_default=settings.enable_rag_by_default,
            default_max_tokens=settings.default_max_tokens,
            default_temperature=settings.default_temperature,
            default_top_p=settings.default_top_p,
            default_frequency_penalty=settings.default_frequency_penalty,
            default_presence_penalty=settings.default_presence_penalty,
            backup_providers=settings.backup_providers or [],
            ui_preferences=settings.ui_preferences or {},
            create_date=settings.create_date,
            update_date=settings.update_date
        )

def update_user_settings(user_id: int, updates: Dict[str, Any]) -> Optional[UserSettingsModel]:
    """
    Update user settings.
    
    Args:
        user_id: User ID to update settings for
        updates: Dictionary of field updates
        
    Returns:
        Updated UserSettingsModel if successful, None if user not found
    """
    with SessionLocal() as session:
        settings = session.query(UserSettings).filter_by(user_id=user_id).first()
        if not settings:
            return None
        
        # Update fields that are provided
        for field, value in updates.items():
            if hasattr(settings, field):
                setattr(settings, field, value)
        
        settings.update_date = datetime.datetime.utcnow()
        session.commit()
        session.refresh(settings)
        
        return UserSettingsModel(
            user_settings_id=settings.user_settings_id,
            user_id=settings.user_id,
            default_agent_type=settings.default_agent_type,
            default_local_model=settings.default_local_model,
            default_online_model=settings.default_online_model,
            default_online_provider=settings.default_online_provider,
            default_file_archives=settings.default_file_archives or [],
            enable_rag_by_default=settings.enable_rag_by_default,
            default_max_tokens=settings.default_max_tokens,
            default_temperature=settings.default_temperature,
            default_top_p=settings.default_top_p,
            default_frequency_penalty=settings.default_frequency_penalty,
            default_presence_penalty=settings.default_presence_penalty,
            backup_providers=settings.backup_providers or [],
            ui_preferences=settings.ui_preferences or {},
            create_date=settings.create_date,
            update_date=settings.update_date
        )

def get_or_create_user_settings(user_id: int) -> UserSettingsModel:
    """
    Get user settings, creating default ones if they don't exist.
    
    Args:
        user_id: User ID to get or create settings for
        
    Returns:
        UserSettingsModel
    """
    settings = get_user_settings(user_id)
    if not settings:
        settings = create_default_user_settings(user_id)
    return settings
