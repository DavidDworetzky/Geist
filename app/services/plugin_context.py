"""Expose plugin skills and MCP servers to chat.

Skills use progressive disclosure: the system prompt carries only each
skill's name and description, and the model pulls a skill's full markdown
body through the ``skills.load`` tool when the description matches the task.
Plugin MCP servers mount through the same tool source machinery as
operator-configured servers.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from agents.models.tool_calling import ToolContext, ToolDefinition, ToolExecutionOutput
from app.services.mcp_tool_source import McpToolSource, get_mcp_manager
from app.services.plugin_loader import PluginRegistry, get_plugin_registry
from app.services.tool_registry import ToolRegistry


logger = logging.getLogger(__name__)

_MAX_PROMPT_SKILLS = 50
_SKILLS_CONTEXT_HEADER = (
    "## Available skills\n"
    "The following installed skills provide task-specific instructions. When a "
    "skill's description matches the current request, call the skills.load tool "
    "with its exact name to read the full instructions before proceeding."
)


def build_plugin_skills_context(registry: PluginRegistry | None = None) -> str:
    """System-prompt block listing installed skills; empty when there are none."""
    registry = registry or get_plugin_registry()
    try:
        skills = registry.skills()
    except Exception:
        logger.exception("Could not load plugin skills; continuing without them")
        return ""
    invocable = [skill for skill in skills if not skill.disable_model_invocation]
    if not invocable:
        return ""
    if len(invocable) > _MAX_PROMPT_SKILLS:
        logger.warning(
            "%d plugin skills installed; only the first %d are advertised",
            len(invocable),
            _MAX_PROMPT_SKILLS,
        )
        invocable = invocable[:_MAX_PROMPT_SKILLS]
    lines = [f"- {skill.qualified_name}: {skill.description}" for skill in invocable]
    return "\n".join([_SKILLS_CONTEXT_HEADER, *lines])


class SkillLoadArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str = Field(
        min_length=1,
        max_length=130,
        description="Qualified skill name, e.g. my-plugin:my-skill",
    )


def install_plugin_support(
    tool_registry: ToolRegistry,
    plugin_registry: PluginRegistry | None = None,
) -> None:
    """Register the skills.load tool and mount plugin MCP servers.

    Safe to call when no plugins are installed: skills.load reports the empty
    catalog, and the plugin MCP source contributes no tools.
    """
    plugins = plugin_registry or get_plugin_registry()

    def load_skill(context: ToolContext, arguments: SkillLoadArguments) -> ToolExecutionOutput:
        skill = plugins.find_skill(arguments.skill)
        if skill is None:
            available = ", ".join(sorted(entry.qualified_name for entry in plugins.skills()))
            raise RuntimeError(
                f"Unknown skill: {arguments.skill}. Installed skills: {available or '(none)'}"
            )
        return ToolExecutionOutput(
            content=skill.load_body() or "(skill has no body)",
            summary=f"Loaded skill {skill.qualified_name}",
        )

    tool_registry.register(
        ToolDefinition(
            name="skills.load",
            description=(
                "Load the full instructions of an installed plugin skill by its "
                "qualified name. Use when an available skill's description matches "
                "the current task."
            ),
            arguments_model=SkillLoadArguments,
            handler=load_skill,
            max_result_chars=100_000,
            source_adapter="PluginRegistry.find_skill",
        )
    )

    source = McpToolSource(
        get_mcp_manager(),
        servers_loader=plugins.enabled_mcp_server_models,
    )
    # Distinguish this source from the DB-backed "mcp" source in the registry.
    source.name = "plugin-mcp"
    tool_registry.add_source(source)
    global _plugin_mcp_source
    _plugin_mcp_source = source


_plugin_mcp_source: McpToolSource | None = None


def get_plugin_mcp_source() -> McpToolSource | None:
    """The mounted plugin MCP source, for cache invalidation on refresh."""
    return _plugin_mcp_source
