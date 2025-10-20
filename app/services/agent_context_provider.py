import logging
from typing import Dict

from app.models.database.database import Session
from app.models.database.agent_preset import AgentPreset
from agents.agent_settings import AgentSettings
from agents.agent_context import AgentContext
from app.environment import LoadEnvironmentDictionary

logger = logging.getLogger(__name__)


def _get_envs() -> Dict[str, str]:
    """
    Load environment variables dictionary via `LoadEnvironmentDictionary`.

    Returns:
        Dict[str, str]: Environment key-value pairs
    """
    return LoadEnvironmentDictionary()


def get_default_agent_context() -> AgentContext:
    """
    Build the default `AgentContext` from the database preset named "Default Preset".

    Returns:
        AgentContext: Populated context using the default preset.

    Raises:
        ValueError: If the default preset is not found.
    """
    session = Session
    try:
        default_preset = session.query(AgentPreset).filter(AgentPreset.name == "Default Preset").first()
        logger.info(f"Default agent preset: {default_preset}")

        if not default_preset:
            raise ValueError("Default Context preset not found in the database.")

        agent_settings = AgentSettings(
            name=default_preset.name,
            version=default_preset.version,
            description=default_preset.description,
            max_tokens=default_preset.max_tokens,
            n=default_preset.n,
            temperature=default_preset.temperature,
            top_p=default_preset.top_p,
            frequency_penalty=default_preset.frequency_penalty,
            presence_penalty=default_preset.presence_penalty,
            interactive_only=default_preset.interactive_only,
            include_world_processing=default_preset.process_world,
        )

        context = AgentContext(settings=agent_settings, envs=_get_envs())
        return context
    finally:
        session.close()


