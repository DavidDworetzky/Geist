"""
DTO models for user settings API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserSettingsBase(BaseModel):
    """Base user settings model."""
    default_agent_type: str = Field(default="local", description="Default agent type (local or online)")
    default_local_model: str = Field(default="meta-llama/Meta-Llama-3.1-8B-Instruct", description="Default local model")
    default_online_model: str = Field(default="gpt-4", description="Default online model")
    default_online_provider: str = Field(default="openai", description="Default online provider")
    default_file_archives: List[int] = Field(default=[], description="Default file archives for RAG")
    enable_rag_by_default: bool = Field(default=True, description="Enable RAG by default")
    default_max_tokens: int = Field(default=4096, description="Default max tokens")
    default_temperature: float = Field(default=1.0, description="Default temperature")
    default_top_p: float = Field(default=1.0, description="Default top_p")
    default_frequency_penalty: float = Field(default=0.0, description="Default frequency penalty")
    default_presence_penalty: float = Field(default=0.0, description="Default presence penalty")
    backup_providers: List[Dict[str, Any]] = Field(default=[], description="Backup provider configurations")
    ui_preferences: Dict[str, Any] = Field(default={}, description="UI preferences")

class UserSettingsCreate(UserSettingsBase):
    """Model for creating user settings."""
    pass

class UserSettingsUpdate(BaseModel):
    """Model for updating user settings (all fields optional)."""
    default_agent_type: Optional[str] = None
    default_local_model: Optional[str] = None
    default_online_model: Optional[str] = None
    default_online_provider: Optional[str] = None
    default_file_archives: Optional[List[int]] = None
    enable_rag_by_default: Optional[bool] = None
    default_max_tokens: Optional[int] = None
    default_temperature: Optional[float] = None
    default_top_p: Optional[float] = None
    default_frequency_penalty: Optional[float] = None
    default_presence_penalty: Optional[float] = None
    backup_providers: Optional[List[Dict[str, Any]]] = None
    ui_preferences: Optional[Dict[str, Any]] = None

class UserSettingsResponse(UserSettingsBase):
    """Model for user settings response."""
    user_settings_id: int
    user_id: int
    create_date: datetime
    update_date: datetime
    
    class Config:
        from_attributes = True

class AgentConfigRequest(BaseModel):
    """Model for agent configuration requests."""
    agent_type: Optional[str] = None  # Override default
    model: Optional[str] = None  # Override default
    endpoint: Optional[str] = None  # For online agents
    runner_type: Optional[str] = None  # For local agents
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

class BackupProviderConfig(BaseModel):
    """Model for backup provider configuration."""
    name: str = Field(description="Provider name")
    base_url: str = Field(description="Provider base URL")
    model: str = Field(description="Model to use with this provider")
    api_key: Optional[str] = Field(default=None, description="API key for this provider")
    priority: int = Field(default=1, description="Priority (1 = highest)")
    
class AgentFactoryConfig(BaseModel):
    """Model for configuring agent factory from user settings."""
    agent_type: str
    model: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    runner_type: Optional[str] = None
    backup_providers: List[BackupProviderConfig] = []
    generation_config: Dict[str, Any] = {}
    
    @classmethod
    def from_user_settings(cls, settings: UserSettingsResponse, overrides: Optional[AgentConfigRequest] = None):
        """Create agent factory config from user settings with optional overrides."""
        overrides = overrides or AgentConfigRequest()
        
        agent_type = overrides.agent_type or settings.default_agent_type
        
        if agent_type == "local":
            model = overrides.model or settings.default_local_model
            runner_type = overrides.runner_type or "mlx_llama"
            endpoint = None
            api_key = None
        else:  # online
            model = overrides.model or settings.default_online_model
            runner_type = None
            # Default endpoint based on provider
            if settings.default_online_provider == "openai":
                endpoint = overrides.endpoint or "https://api.openai.com/v1/chat/completions"
            elif settings.default_online_provider == "anthropic":
                endpoint = overrides.endpoint or "https://api.anthropic.com/v1/messages"
            elif settings.default_online_provider == "groq":
                endpoint = overrides.endpoint or "https://api.groq.com/openai/v1/chat/completions"
            elif settings.default_online_provider == "grok":
                endpoint = overrides.endpoint or "https://api.x.ai/v1/chat/completions"
            else:
                endpoint = overrides.endpoint
            api_key = None  # Will be retrieved from environment
        
        # Convert backup providers
        backup_providers = [
            BackupProviderConfig(**provider) for provider in settings.backup_providers
        ]
        
        # Generation config
        generation_config = {
            "max_tokens": overrides.max_tokens or settings.default_max_tokens,
            "temperature": overrides.temperature or settings.default_temperature,
            "top_p": overrides.top_p or settings.default_top_p,
            "frequency_penalty": overrides.frequency_penalty or settings.default_frequency_penalty,
            "presence_penalty": overrides.presence_penalty or settings.default_presence_penalty,
        }
        
        return cls(
            agent_type=agent_type,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            runner_type=runner_type,
            backup_providers=backup_providers,
            generation_config=generation_config
        )
