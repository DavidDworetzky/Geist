import logging

from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from app.environment import load_environment_dictionary
from app.models.database.agent_preset import AgentPreset
from app.models.database.database import SessionLocal


logger = logging.getLogger(__name__)


def _get_envs() -> dict[str, str]:
    """
    Load environment variables dictionary via `load_environment_dictionary`.

    Returns:
        Dict[str, str]: Environment key-value pairs
    """
    return load_environment_dictionary()


def settings_from_preset(preset: AgentPreset) -> AgentSettings:
    """Build AgentSettings from a preset row, falling back to defaults for NULL columns."""
    def pick(value, default):
        return default if value is None else value

    return AgentSettings(
        name=pick(preset.name, "default"),
        version=pick(preset.version, "1.0"),
        description=pick(preset.description, ""),
        max_tokens=pick(preset.max_tokens, 16),
        n=pick(preset.n, 1),
        temperature=pick(preset.temperature, 1.0),
        top_p=pick(preset.top_p, 1.0),
        frequency_penalty=pick(preset.frequency_penalty, 0.0),
        presence_penalty=pick(preset.presence_penalty, 0.0),
        interactive_only=pick(preset.interactive_only, False),
        include_world_processing=pick(preset.process_world, False),
    )


def get_default_agent_context() -> AgentContext:
    """
    Build the default `AgentContext` from the database preset named "Default Preset".

    Returns:
        AgentContext: Populated context using the default preset.

    Raises:
        ValueError: If the default preset is not found.
    """
    with SessionLocal() as session:
        default_preset = session.query(AgentPreset).filter(AgentPreset.name == "Default Preset").first()
        logger.info(f"Default agent preset: {default_preset}")

        if not default_preset:
            raise ValueError("Default Context preset not found in the database.")

        agent_settings = settings_from_preset(default_preset)

        context = AgentContext(settings=agent_settings, envs=_get_envs())
        return context


