"""
DTO models for user settings API.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UserSettingsBase(BaseModel):
    """Base user settings model."""
    default_agent_type: str = Field(default="local", description="Default agent type (local or online)")
    default_local_model: str = Field(default="meta-llama/Meta-Llama-3.1-8B-Instruct", description="Default local model")
    default_online_model: str = Field(default="gpt-4", description="Default online model")
    default_online_provider: str = Field(default="openai", description="Default online provider")
    default_file_archives: list[int] = Field(default=[], description="Default file archives for RAG")
    enable_rag_by_default: bool = Field(default=True, description="Enable RAG by default")
    default_max_tokens: int = Field(default=4096, description="Default max tokens")
    default_temperature: float = Field(default=1.0, description="Default temperature")
    default_top_p: float = Field(default=1.0, description="Default top_p")
    default_frequency_penalty: float = Field(default=0.0, description="Default frequency penalty")
    default_presence_penalty: float = Field(default=0.0, description="Default presence penalty")
    backup_providers: list[dict[str, Any]] = Field(default=[], description="Backup provider configurations")
    ui_preferences: dict[str, Any] = Field(default={}, description="UI preferences")

class UserSettingsCreate(UserSettingsBase):
    """Model for creating user settings."""
    pass

class UserSettingsUpdate(BaseModel):
    """Model for updating user settings (all fields optional)."""
    default_agent_type: str | None = None
    default_local_model: str | None = None
    default_online_model: str | None = None
    default_online_provider: str | None = None
    default_file_archives: list[int] | None = None
    enable_rag_by_default: bool | None = None
    default_max_tokens: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_frequency_penalty: float | None = None
    default_presence_penalty: float | None = None
    backup_providers: list[dict[str, Any]] | None = None
    ui_preferences: dict[str, Any] | None = None

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
    agent_type: str | None = None  # Override default
    model: str | None = None  # Override default
    endpoint: str | None = None  # For online agents
    runner_type: str | None = None  # For local agents
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

class BackupProviderConfig(BaseModel):
    """Model for backup provider configuration."""
    name: str = Field(description="Provider name")
    base_url: str = Field(description="Provider base URL")
    model: str = Field(description="Model to use with this provider")
    api_key: str | None = Field(default=None, description="API key for this provider")
    priority: int = Field(default=1, description="Priority (1 = highest)")

class AgentFactoryConfig(BaseModel):
    """Model for configuring agent factory from user settings."""
    agent_type: str
    model: str
    endpoint: str | None = None
    api_key: str | None = None
    runner_type: str | None = None
    backup_providers: list[BackupProviderConfig] = []
    generation_config: dict[str, Any] = {}

    @classmethod
    def from_user_settings(cls, settings: UserSettingsResponse, overrides: AgentConfigRequest | None = None):
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
