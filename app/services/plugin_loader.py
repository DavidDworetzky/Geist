"""Discovery and parsing for Agent Plugins (agent-plugins.org, v1.0.0 core).

A plugin is a directory under the configured plugin root containing:

- ``.claude-plugin/plugin.json`` (or ``plugin.json`` at the plugin root): the
  manifest. ``name`` is required; unrecognized fields are ignored per the
  standard so manifests written for richer hosts still load.
- ``skills/<skill>/SKILL.md``: markdown skills with YAML frontmatter. Skill
  names are namespaced as ``<plugin>:<skill>``.
- ``.mcp.json`` (or ``mcp.json``): MCP server declarations mounted through the
  same client machinery as operator-configured servers.

Trust model: dropping a plugin into the plugin directory is the install/trust
act for its *skills* (passive markdown, loaded on demand). MCP servers spawn
processes or open network connections, so they additionally require the plugin
to be named in ``GEIST_ENABLED_PLUGINS`` — mirroring how DB-configured MCP
servers stay disabled until an operator enables them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.models.database.mcp_server import McpServerModel
from app.runtime_config import default_data_dir


logger = logging.getLogger(__name__)

_PLUGIN_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MANIFEST_MAX_BYTES = 64 * 1024
_SKILL_FILE_MAX_BYTES = 1024 * 1024
_SKILL_BODY_MAX_CHARS = 100_000
_SKILL_DESCRIPTION_MAX_CHARS = 1000
_MAX_PLUGINS = 100
_MAX_SKILLS_PER_PLUGIN = 100
_MAX_MCP_SERVERS_PER_PLUGIN = 100
_PLUGIN_ROOT_VARIABLES = ("${CLAUDE_PLUGIN_ROOT}", "${GEIST_PLUGIN_ROOT}")
_HTTP_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def default_plugin_dir() -> Path:
    configured = os.getenv("GEIST_PLUGIN_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return default_data_dir() / "plugins"


def enabled_plugin_names() -> set[str]:
    """Plugins whose MCP servers may be mounted, from GEIST_ENABLED_PLUGINS."""
    return {
        name.strip() for name in os.getenv("GEIST_ENABLED_PLUGINS", "").split(",") if name.strip()
    }


def parse_frontmatter(text: str) -> tuple[dict[str, str | bool], str]:
    """Split markdown into (frontmatter fields, body).

    Deliberately tolerant of the simple ``key: value`` YAML subset that skill
    frontmatter uses in practice — including unquoted values containing
    colons, which strict YAML rejects. Indented follow-on lines continue the
    previous value.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    fields: dict[str, str | bool] = {}
    last_key: str | None = None
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :]).strip()
            return fields, body
        if not line.strip():
            continue
        if line[0] in (" ", "\t") and last_key is not None:
            previous = fields[last_key]
            if isinstance(previous, str):
                fields[last_key] = f"{previous} {line.strip()}".strip()
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip():
            continue
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "'\"":
            cleaned = cleaned[1:-1]
        if cleaned in (">", ">-", "|", "|-"):
            cleaned = ""
        normalized_key = key.strip().lower()
        if cleaned.lower() in ("true", "false"):
            fields[normalized_key] = cleaned.lower() == "true"
        else:
            fields[normalized_key] = cleaned
        last_key = normalized_key
    # No closing delimiter: treat the document as body-only.
    return {}, text


@dataclass(frozen=True)
class PluginSkill:
    plugin_name: str
    name: str
    description: str
    plugin_root: Path
    path: Path
    disable_model_invocation: bool = False

    @property
    def qualified_name(self) -> str:
        return f"{self.plugin_name}:{self.name}"

    def load_body(self) -> str:
        text = _read_text_inside(
            self.plugin_root,
            self.path,
            maximum_bytes=_SKILL_FILE_MAX_BYTES,
        )
        if text is None:
            raise RuntimeError(f"Skill file is unavailable or unsafe: {self.qualified_name}")
        _, body = parse_frontmatter(text)
        if len(body) > _SKILL_BODY_MAX_CHARS:
            body = f"{body[:_SKILL_BODY_MAX_CHARS]}\n[skill truncated]"
        return body


@dataclass(frozen=True)
class PluginMcpServer:
    plugin_name: str
    name: str
    transport: str  # "stdio" | "http"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.plugin_name}.{self.name}"


@dataclass(frozen=True)
class AgentPlugin:
    name: str
    root: Path
    version: str | None = None
    description: str | None = None
    display_name: str | None = None
    skills: tuple[PluginSkill, ...] = ()
    mcp_servers: tuple[PluginMcpServer, ...] = ()


def _substitute_plugin_root(value: str, root: Path) -> str:
    for variable in _PLUGIN_ROOT_VARIABLES:
        value = value.replace(variable, str(root))
    return value


def _resolve_inside(root: Path, relative: str) -> Path | None:
    """Resolve a manifest-relative path, refusing escapes from the plugin."""
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _read_text_inside(root: Path, path: Path, *, maximum_bytes: int) -> str | None:
    """Read a bounded regular file only when its resolved path stays in root."""
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
        if not resolved.is_file() or resolved.stat().st_size > maximum_bytes:
            return None
        with resolved.open("rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except (OSError, ValueError):
        return None
    if len(payload) > maximum_bytes:
        return None
    return payload.decode("utf-8", errors="replace")


def _read_json(path: Path) -> dict | None:
    try:
        if path.stat().st_size > _MANIFEST_MAX_BYTES:
            logger.warning("Plugin file too large, skipping: %s", path)
            return None
        with path.open("rb") as handle:
            payload = handle.read(_MANIFEST_MAX_BYTES + 1)
        if len(payload) > _MANIFEST_MAX_BYTES:
            logger.warning("Plugin file too large, skipping: %s", path)
            return None
        loaded = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        logger.warning("Could not read plugin file %s: %s", path, error)
        return None
    if not isinstance(loaded, dict):
        logger.warning("Plugin file %s is not a JSON object", path)
        return None
    return loaded


def _load_manifest(root: Path) -> dict | None:
    for relative in (".claude-plugin/plugin.json", "plugin.json"):
        candidate = _resolve_inside(root, relative)
        if candidate is not None and candidate.is_file():
            return _read_json(candidate)
    return None


def _string_or_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, str)]
    return []


def _discover_skills(plugin_name: str, root: Path, manifest: dict) -> list[PluginSkill]:
    skill_dirs: list[Path] = []
    default_dir = _resolve_inside(root, "skills")
    if default_dir is not None and default_dir.is_dir():
        skill_dirs.append(default_dir)
    for entry in _string_or_list(manifest.get("skills")):
        resolved = _resolve_inside(root, entry)
        if resolved is None or not resolved.is_dir():
            logger.warning("Plugin %s skills path invalid, skipping: %s", plugin_name, entry)
            continue
        if resolved not in skill_dirs:
            skill_dirs.append(resolved)

    skills: list[PluginSkill] = []
    seen: set[str] = set()
    for skill_dir in skill_dirs:
        for skill_file in sorted(skill_dir.glob("*/SKILL.md")):
            if len(skills) >= _MAX_SKILLS_PER_PLUGIN:
                logger.warning(
                    "Plugin %s has more than %d skills; ignoring the rest",
                    plugin_name,
                    _MAX_SKILLS_PER_PLUGIN,
                )
                return skills
            text = _read_text_inside(root, skill_file, maximum_bytes=_SKILL_FILE_MAX_BYTES)
            if text is None:
                logger.warning("Plugin skill file is unavailable or unsafe, skipping: %s", skill_file)
                continue
            fields, _ = parse_frontmatter(text)
            raw_name = fields.get("name")
            name = raw_name if isinstance(raw_name, str) and raw_name else skill_file.parent.name
            description = fields.get("description")
            if not _SKILL_NAME_PATTERN.match(name):
                logger.warning("Plugin %s skill has invalid name %r, skipping", plugin_name, name)
                continue
            if not isinstance(description, str):
                logger.warning("Plugin %s skill %s has no description, skipping", plugin_name, name)
                continue
            description = " ".join(description.split())[:_SKILL_DESCRIPTION_MAX_CHARS]
            if not description:
                logger.warning("Plugin %s skill %s has no description, skipping", plugin_name, name)
                continue
            if name in seen:
                logger.warning("Plugin %s has duplicate skill %s, skipping", plugin_name, name)
                continue
            seen.add(name)
            skills.append(
                PluginSkill(
                    plugin_name=plugin_name,
                    name=name,
                    description=description,
                    plugin_root=root.resolve(),
                    path=skill_file.resolve(),
                    disable_model_invocation=fields.get("disable-model-invocation") is True,
                )
            )
    return skills


def _parse_mcp_server(
    plugin_name: str, server_name: str, spec: object, root: Path
) -> PluginMcpServer | None:
    if not isinstance(spec, dict):
        return None
    if not _SKILL_NAME_PATTERN.match(server_name):
        logger.warning(
            "Plugin %s MCP server has invalid name %r, skipping", plugin_name, server_name
        )
        return None
    url = spec.get("url")
    if isinstance(url, str) and url:
        expanded_url = _substitute_plugin_root(url, root)
        if not expanded_url.lower().startswith(("http://", "https://")):
            logger.warning(
                "Plugin %s MCP server %s has non-http url, skipping", plugin_name, server_name
            )
            return None
        raw_headers = spec.get("headers")
        if raw_headers is not None and not isinstance(raw_headers, dict):
            logger.warning(
                "Plugin %s MCP server %s has invalid headers, skipping",
                plugin_name,
                server_name,
            )
            return None
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(key) > 256
            or _HTTP_HEADER_NAME_PATTERN.fullmatch(key) is None
            or len(value) > 8192
            or "\r" in value
            or "\n" in value
            for key, value in (raw_headers or {}).items()
        ):
            logger.warning(
                "Plugin %s MCP server %s has invalid headers, skipping",
                plugin_name,
                server_name,
            )
            return None
        headers = {
            key: _substitute_plugin_root(value, root) for key, value in (raw_headers or {}).items()
        }
        if (
            len(raw_headers or {}) > 64
            or len(expanded_url) > 2048
            or any(len(value) > 8192 for value in headers.values())
        ):
            logger.warning("Plugin %s MCP server %s exceeds HTTP limits, skipping", plugin_name, server_name)
            return None
        return PluginMcpServer(
            plugin_name=plugin_name,
            name=server_name,
            transport="http",
            url=expanded_url,
            headers=headers,
        )
    command = spec.get("command")
    if not isinstance(command, str) or not command or len(command) > 1024 or "\x00" in command:
        logger.warning(
            "Plugin %s MCP server %s has neither command nor url, skipping",
            plugin_name,
            server_name,
        )
        return None
    raw_args = spec.get("args")
    if raw_args is not None and (
        not isinstance(raw_args, list) or not all(isinstance(arg, str) for arg in raw_args)
    ):
        logger.warning(
            "Plugin %s MCP server %s has invalid args, skipping",
            plugin_name,
            server_name,
        )
        return None
    raw_env = spec.get("env")
    if raw_env is not None and not isinstance(raw_env, dict):
        logger.warning(
            "Plugin %s MCP server %s has invalid env, skipping",
            plugin_name,
            server_name,
        )
        return None
    if len(raw_args or []) > 64 or any(len(arg) > 4096 or "\x00" in arg for arg in (raw_args or [])):
        logger.warning("Plugin %s MCP server %s exceeds argument limits, skipping", plugin_name, server_name)
        return None
    if len(raw_env or {}) > 64 or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key
        or len(key) > 256
        or "=" in key
        or "\x00" in key
        or len(value) > 8192
        or "\x00" in value
        for key, value in (raw_env or {}).items()
    ):
        logger.warning("Plugin %s MCP server %s has invalid environment, skipping", plugin_name, server_name)
        return None
    expanded_command = _substitute_plugin_root(command, root)
    args = tuple(_substitute_plugin_root(arg, root) for arg in (raw_args or []))
    env = {
        key: _substitute_plugin_root(value, root) for key, value in (raw_env or {}).items()
    }
    if (
        len(expanded_command) > 1024
        or any(len(argument) > 4096 for argument in args)
        or any(len(value) > 8192 for value in env.values())
    ):
        logger.warning("Plugin %s MCP server %s exceeds expanded value limits, skipping", plugin_name, server_name)
        return None
    return PluginMcpServer(
        plugin_name=plugin_name,
        name=server_name,
        transport="stdio",
        command=expanded_command,
        args=args,
        env=env,
    )


def _discover_mcp_servers(plugin_name: str, root: Path, manifest: dict) -> list[PluginMcpServer]:
    declared = manifest.get("mcpServers")
    servers_spec: dict | None = None
    if isinstance(declared, dict):
        servers_spec = declared
    elif isinstance(declared, str):
        resolved = _resolve_inside(root, declared)
        if resolved is None or not resolved.is_file():
            logger.warning("Plugin %s mcpServers path invalid, skipping: %s", plugin_name, declared)
        else:
            servers_spec = _read_json(resolved)
    if servers_spec is None:
        for candidate in (root / ".mcp.json", root / "mcp.json"):
            if candidate.is_file():
                servers_spec = _read_json(candidate)
                break
    if servers_spec is None:
        return []
    inner = servers_spec.get("mcpServers")
    if isinstance(inner, dict):
        servers_spec = inner
    servers: list[PluginMcpServer] = []
    for index, (server_name, spec) in enumerate(servers_spec.items()):
        if index >= _MAX_MCP_SERVERS_PER_PLUGIN:
            logger.warning(
                "Plugin %s has more than %d MCP servers; ignoring the rest",
                plugin_name,
                _MAX_MCP_SERVERS_PER_PLUGIN,
            )
            break
        server = _parse_mcp_server(plugin_name, str(server_name), spec, root)
        if server is not None:
            servers.append(server)
    return servers


def load_plugin(root: Path) -> AgentPlugin | None:
    """Load one plugin directory, returning None when it is not a valid plugin."""
    manifest = _load_manifest(root)
    if manifest is None:
        logger.warning("Directory %s has no plugin.json manifest, skipping", root)
        return None
    name = manifest.get("name")
    if not isinstance(name, str) or not _PLUGIN_NAME_PATTERN.match(name):
        logger.warning("Plugin at %s has a missing or invalid name, skipping", root)
        return None
    version = manifest.get("version")
    description = manifest.get("description")
    display_name = manifest.get("displayName")
    return AgentPlugin(
        name=name,
        root=root,
        version=version if isinstance(version, str) else None,
        description=description if isinstance(description, str) else None,
        display_name=display_name if isinstance(display_name, str) else None,
        skills=tuple(_discover_skills(name, root, manifest)),
        mcp_servers=tuple(_discover_mcp_servers(name, root, manifest)),
    )


def discover_plugins(plugin_dir: Path) -> list[AgentPlugin]:
    if not plugin_dir.is_dir():
        return []
    plugins: list[AgentPlugin] = []
    seen: set[str] = set()
    for entry in sorted(plugin_dir.iterdir()):
        if entry.is_symlink() or not entry.is_dir() or entry.name.startswith("."):
            continue
        if len(plugins) >= _MAX_PLUGINS:
            logger.warning(
                "More than %d plugins in %s; ignoring the rest", _MAX_PLUGINS, plugin_dir
            )
            break
        plugin = load_plugin(entry)
        if plugin is None:
            continue
        if plugin.name in seen:
            logger.warning("Duplicate plugin name %s at %s, skipping", plugin.name, entry)
            continue
        seen.add(plugin.name)
        plugins.append(plugin)
    return plugins


def _synthetic_server_id(qualified_name: str) -> int:
    """Stable negative id so plugin servers never collide with DB-backed rows."""
    digest = hashlib.sha256(qualified_name.encode("utf-8")).digest()
    return -int.from_bytes(digest, "big") - 1


class PluginRegistry:
    """Discovered plugins with thread-safe refresh, scanned lazily on first use."""

    def __init__(self, plugin_dir: Path | None = None):
        self._configured_dir = plugin_dir
        self._lock = threading.Lock()
        self._plugins: list[AgentPlugin] | None = None

    @property
    def plugin_dir(self) -> Path:
        return self._configured_dir or default_plugin_dir()

    def refresh(self) -> None:
        with self._lock:
            self._plugins = discover_plugins(self.plugin_dir)

    def plugins(self) -> list[AgentPlugin]:
        with self._lock:
            if self._plugins is None:
                self._plugins = discover_plugins(self.plugin_dir)
            return list(self._plugins)

    def skills(self) -> list[PluginSkill]:
        return [skill for plugin in self.plugins() for skill in plugin.skills]

    def find_skill(self, qualified_name: str) -> PluginSkill | None:
        for skill in self.skills():
            if skill.qualified_name == qualified_name:
                return skill
        return None

    def enabled_mcp_server_models(self, _user_id: int | None = None) -> list[McpServerModel]:
        """Plugin MCP servers, as models the shared MCP tool source can mount.

        Only plugins named in GEIST_ENABLED_PLUGINS contribute servers; the
        synthetic negative ids keep the client manager's connection cache
        separate from operator-configured (DB) servers.
        """
        enabled = enabled_plugin_names()
        epoch = datetime.datetime(1970, 1, 1)
        models: list[McpServerModel] = []
        for plugin in self.plugins():
            if plugin.name not in enabled:
                continue
            for server in plugin.mcp_servers:
                models.append(
                    McpServerModel(
                        mcp_server_id=_synthetic_server_id(server.qualified_name),
                        user_id=0,
                        name=server.qualified_name,
                        transport=server.transport,
                        command=server.command,
                        args=list(server.args),
                        env=dict(server.env),
                        url=server.url,
                        headers=dict(server.headers),
                        enabled=True,
                        timeout_seconds=30.0,
                        create_date=epoch,
                        update_date=epoch,
                    )
                )
        return models


_registry: PluginRegistry | None = None
_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = PluginRegistry()
        return _registry
