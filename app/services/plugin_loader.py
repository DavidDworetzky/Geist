"""Agent Plugins 1.0.0 discovery, validation, and runtime configuration.

Geist implements the portable core at fixed package locations: root
``plugin.json``, ``skills/*/SKILL.md``, and optional root ``mcp.json``.
Independent component failures stay isolated, while a bad manifest rejects the
whole package as required by the standard.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import json
import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from app.models.database.mcp_server import McpServerModel
from app.runtime_config import default_data_dir


logger = logging.getLogger(__name__)

PLUGIN_SCHEMA_ID = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
MCP_SCHEMA_ID = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
_PLUGIN_NAME_PATTERN = re.compile(r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_MANIFEST_MAX_BYTES = 64 * 1024
_SKILL_FILE_MAX_BYTES = 1024 * 1024
_SKILL_BODY_MAX_CHARS = 100_000
_SKILL_DESCRIPTION_MAX_CHARS = 1024
_SKILL_COMPATIBILITY_MAX_CHARS = 500
_MAX_PLUGINS = 100
_MAX_SKILLS_PER_PLUGIN = 100
_MAX_MCP_SERVERS_PER_PLUGIN = 100
_HTTP_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_MANIFEST_FIELDS = {
    "$schema",
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "extensions",
}
_MANIFEST_STRING_FIELDS = {
    "version",
    "description",
    "homepage",
    "repository",
    "license",
}
_SKILL_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}
_RESERVED_PLUGIN_ENVIRONMENT = {"PLUGIN_ROOT", "PLUGIN_DATA"}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def default_plugin_dir() -> Path:
    configured = os.getenv("GEIST_PLUGIN_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return default_data_dir() / "plugins"


def default_plugin_data_dir() -> Path:
    configured = os.getenv("GEIST_PLUGIN_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return default_data_dir() / "plugin-data"


def enabled_plugin_names() -> set[str]:
    """Plugins whose MCP servers may be mounted, from GEIST_ENABLED_PLUGINS."""
    return {
        name.strip() for name in os.getenv("GEIST_ENABLED_PLUGINS", "").split(",") if name.strip()
    }


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse the required YAML frontmatter and Markdown body from ``SKILL.md``."""
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        raise ValueError("SKILL.md frontmatter is not closed") from None
    try:
        fields = yaml.load("\n".join(lines[1:closing_index]), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"SKILL.md frontmatter is invalid YAML: {error}") from error
    if not isinstance(fields, dict) or not all(isinstance(key, str) for key in fields):
        raise ValueError("SKILL.md frontmatter must be a string-keyed mapping")
    return fields, "\n".join(lines[closing_index + 1 :]).strip()


@dataclass(frozen=True)
class PluginSkill:
    plugin_name: str
    name: str
    description: str
    plugin_root: Path
    path: Path

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
        fields, body = parse_frontmatter(text)
        metadata = _valid_skill_metadata(fields, self.path.parent)
        if metadata is None or metadata[0] != self.name:
            raise RuntimeError(f"Skill metadata changed or is invalid: {self.qualified_name}")
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
    cwd: str | None = None
    plugin_root: str | None = None
    plugin_data_dir: str | None = None
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
    skills: tuple[PluginSkill, ...] = ()
    mcp_servers: tuple[PluginMcpServer, ...] = ()


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
    candidate = _resolve_inside(root, "plugin.json")
    if candidate is None or not candidate.is_file():
        return None
    manifest = _read_json(candidate)
    if manifest is None:
        return None

    for field_name in sorted(set(manifest) - _MANIFEST_FIELDS):
        logger.warning("Plugin %s has unknown manifest field %s; ignoring it", root, field_name)
        manifest.pop(field_name)

    if manifest.get("$schema") != PLUGIN_SCHEMA_ID:
        logger.warning("Plugin %s has a missing or unsupported $schema", root)
        return None
    name = manifest.get("name")
    if not isinstance(name, str) or len(name) > 64 or _PLUGIN_NAME_PATTERN.fullmatch(name) is None:
        logger.warning("Plugin %s has an invalid name", root)
        return None
    for field_name in _MANIFEST_STRING_FIELDS:
        if field_name in manifest and not isinstance(manifest[field_name], str):
            logger.warning("Plugin %s has invalid manifest field %s", root, field_name)
            return None
    author = manifest.get("author")
    if author is not None and (
        not isinstance(author, dict)
        or set(author) - {"name", "email", "url"}
        or not all(isinstance(value, str) for value in author.values())
    ):
        logger.warning("Plugin %s has an invalid author", root)
        return None
    keywords = manifest.get("keywords")
    if keywords is not None and (
        not isinstance(keywords, list) or not all(isinstance(value, str) for value in keywords)
    ):
        logger.warning("Plugin %s has invalid keywords", root)
        return None
    if "extensions" in manifest and not isinstance(manifest["extensions"], dict):
        logger.warning("Plugin %s has a non-object extensions field; ignoring it", root)
        manifest.pop("extensions")
    return manifest


def _valid_skill_metadata(fields: dict[str, object], skill_dir: Path) -> tuple[str, str] | None:
    unknown_fields = set(fields) - _SKILL_FIELDS
    if unknown_fields:
        logger.warning(
            "Plugin skill %s has unsupported frontmatter fields: %s",
            skill_dir,
            ", ".join(sorted(unknown_fields)),
        )
        return None
    raw_name = fields.get("name")
    raw_description = fields.get("description")
    if not isinstance(raw_name, str) or not isinstance(raw_description, str):
        return None
    name = unicodedata.normalize("NFKC", raw_name.strip())
    directory_name = unicodedata.normalize("NFKC", skill_dir.name)
    if (
        not name
        or len(name) > 64
        or name != name.lower()
        or name.startswith("-")
        or name.endswith("-")
        or "--" in name
        or not all(character.isalnum() or character == "-" for character in name)
        or name != directory_name
    ):
        return None
    description = raw_description.strip()
    if not description or len(description) > _SKILL_DESCRIPTION_MAX_CHARS:
        return None
    license_value = fields.get("license")
    if license_value is not None and not isinstance(license_value, str):
        return None
    compatibility = fields.get("compatibility")
    if compatibility is not None and (
        not isinstance(compatibility, str)
        or not compatibility
        or len(compatibility) > _SKILL_COMPATIBILITY_MAX_CHARS
    ):
        return None
    metadata = fields.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
        )
    ):
        return None
    allowed_tools = fields.get("allowed-tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        return None
    return name, " ".join(description.split())


def _discover_skills(plugin_name: str, root: Path) -> list[PluginSkill]:
    skill_dir = _resolve_inside(root, "skills")
    if skill_dir is None:
        logger.warning("Plugin %s skills directory escapes the plugin root", plugin_name)
        return []
    if not skill_dir.exists():
        return []
    if not skill_dir.is_dir():
        logger.warning("Plugin %s skills path is not a directory", plugin_name)
        return []

    skills: list[PluginSkill] = []
    seen: set[str] = set()
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
        try:
            fields, _ = parse_frontmatter(text)
        except ValueError as error:
            logger.warning("Plugin skill %s is invalid: %s", skill_file, error)
            continue
        metadata = _valid_skill_metadata(fields, skill_file.parent)
        if metadata is None:
            logger.warning("Plugin skill %s has invalid Agent Skills metadata", skill_file)
            continue
        name, description = metadata
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
            )
        )
    return skills


def _parse_mcp_server(
    plugin_name: str,
    server_name: str,
    spec: object,
    root: Path,
    plugin_data_dir: Path,
) -> PluginMcpServer | None:
    if not isinstance(spec, dict):
        return None
    transport = spec.get("type")
    if transport == "sse":
        logger.warning(
            "Plugin %s MCP server %s uses unsupported legacy SSE; skipping",
            plugin_name,
            server_name,
        )
        return None
    if transport == "streamable-http":
        if set(spec) - {"type", "url", "headers"}:
            return None
        url = spec.get("url")
        raw_headers = spec.get("headers", {})
        if (
            not isinstance(url, str)
            or not url
            or len(url) > 2048
            or not _valid_remote_url(url)
            or not isinstance(raw_headers, dict)
            or len(raw_headers) > 64
            or not _valid_http_headers(raw_headers)
        ):
            return None
        return PluginMcpServer(
            plugin_name=plugin_name,
            name=server_name,
            transport="http",
            url=url,
            headers=dict(raw_headers),
        )
    if transport != "stdio" or set(spec) - {"type", "command", "args", "env", "cwd"}:
        return None
    command = spec.get("command")
    if not isinstance(command, str) or not command or len(command) > 1024 or "\x00" in command:
        return None
    if command.startswith("./"):
        resolved_command = _resolve_under(root, command[2:])
        if resolved_command is None:
            return None
        command = str(resolved_command)
    elif "/" in command or "\\" in command:
        return None
    raw_args = spec.get("args", [])
    raw_env = spec.get("env", {})
    if (
        not isinstance(raw_args, list)
        or not all(isinstance(argument, str) for argument in raw_args)
        or not isinstance(raw_env, dict)
    ):
        return None
    if len(raw_args or []) > 64 or any(
        len(arg) > 4096 or "\x00" in arg for arg in (raw_args or [])
    ):
        return None
    if len(raw_env or {}) > 64 or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key
        or len(key) > 256
        or "=" in key
        or "\x00" in key
        or key in _RESERVED_PLUGIN_ENVIRONMENT
        or len(value) > 8192
        or "\x00" in value
        for key, value in (raw_env or {}).items()
    ):
        return None
    root_path = root.resolve()
    data_path = plugin_data_dir.resolve()
    args = tuple(_expand_plugin_variables(argument, root_path, data_path) for argument in raw_args)
    env = {
        key: _expand_plugin_variables(value, root_path, data_path) for key, value in raw_env.items()
    }
    cwd = _resolve_plugin_cwd(spec.get("cwd"), root_path, data_path)
    if cwd is None:
        return None
    if any(len(argument) > 4096 for argument in args) or any(
        len(value) > 8192 for value in env.values()
    ):
        return None
    return PluginMcpServer(
        plugin_name=plugin_name,
        name=server_name,
        transport="stdio",
        command=command,
        args=args,
        env=env,
        cwd=str(cwd),
        plugin_root=str(root_path),
        plugin_data_dir=str(data_path),
    )


def _resolve_under(base: Path, relative: str) -> Path | None:
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def _expand_plugin_variables(value: str, root: Path, data_dir: Path) -> str:
    values = {"PLUGIN_ROOT": str(root), "PLUGIN_DATA": str(data_dir)}
    return re.sub(
        r"\$\{(PLUGIN_ROOT|PLUGIN_DATA)\}",
        lambda match: values[match.group(1)],
        value,
    )


def _resolve_plugin_cwd(value: object, root: Path, data_dir: Path) -> Path | None:
    if value is None:
        return root
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    if value.startswith("./"):
        expanded = _expand_plugin_variables(value, root, data_dir)
        return _resolve_under(root, expanded[2:])
    for placeholder, base in (("${PLUGIN_ROOT}", root), ("${PLUGIN_DATA}", data_dir)):
        if value == placeholder:
            return base
        prefix = f"{placeholder}/"
        if value.startswith(prefix):
            expanded = _expand_plugin_variables(value, root, data_dir)
            try:
                expanded_relative = Path(expanded).relative_to(base)
            except ValueError:
                return None
            return _resolve_under(base, str(expanded_relative))
    return None


def _is_loopback_host(hostname: str) -> bool:
    try:
        return ipaddress.ip_address(hostname.split("%", 1)[0]).is_loopback
    except ValueError:
        return hostname.rstrip(".").lower() == "localhost"


def _valid_http_headers(headers: dict[object, object]) -> bool:
    normalized_names: set[str] = set()
    for name, value in headers.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or len(name) > 256
            or _HTTP_HEADER_NAME_PATTERN.fullmatch(name) is None
            or len(value) > 8192
            or any(
                ord(character) == 0x7F or (ord(character) < 0x20 and character != "\t")
                for character in value
            )
        ):
            return False
        normalized_name = name.lower()
        if normalized_name in normalized_names:
            return False
        normalized_names.add(normalized_name)
    return True


def _valid_remote_url(url: str) -> bool:
    if any(character.isspace() or ord(character) < 0x20 for character in url):
        return False
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        return False
    return parsed.scheme == "https" or _is_loopback_host(parsed.hostname)


def _discover_mcp_servers(plugin_name: str, root: Path) -> list[PluginMcpServer]:
    candidate = _resolve_inside(root, "mcp.json")
    if candidate is None:
        logger.warning("Plugin %s mcp.json escapes the plugin root", plugin_name)
        return []
    if not candidate.exists():
        return []
    if not candidate.is_file():
        logger.warning("Plugin %s mcp.json is not a regular file", plugin_name)
        return []
    document = _read_json(candidate)
    if (
        document is None
        or set(document) != {"$schema", "mcpServers"}
        or document.get("$schema") != MCP_SCHEMA_ID
        or not isinstance(document.get("mcpServers"), dict)
    ):
        logger.warning("Plugin %s has invalid Agent Plugins mcp.json", plugin_name)
        return []
    servers_spec = document["mcpServers"]
    assert isinstance(servers_spec, dict)
    plugin_data_dir = default_plugin_data_dir() / plugin_name
    servers: list[PluginMcpServer] = []
    for index, (server_name, spec) in enumerate(servers_spec.items()):
        if index >= _MAX_MCP_SERVERS_PER_PLUGIN:
            logger.warning(
                "Plugin %s has more than %d MCP servers; ignoring the rest",
                plugin_name,
                _MAX_MCP_SERVERS_PER_PLUGIN,
            )
            break
        server = _parse_mcp_server(
            plugin_name,
            str(server_name),
            spec,
            root,
            plugin_data_dir,
        )
        if server is not None:
            servers.append(server)
        else:
            logger.warning("Plugin %s MCP server %s is invalid; skipping", plugin_name, server_name)
    return servers


def load_plugin(root: Path) -> AgentPlugin | None:
    """Load one plugin directory, returning None when it is not a valid plugin."""
    manifest = _load_manifest(root)
    if manifest is None:
        logger.warning("Directory %s has no plugin.json manifest, skipping", root)
        return None
    name = manifest["name"]
    assert isinstance(name, str)
    version = manifest.get("version")
    description = manifest.get("description")
    return AgentPlugin(
        name=name,
        root=root,
        version=version if isinstance(version, str) else None,
        description=description if isinstance(description, str) else None,
        skills=tuple(_discover_skills(name, root)),
        mcp_servers=tuple(_discover_mcp_servers(name, root)),
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

    def enabled_mcp_server_models(self, workspace_id: int | None = None) -> list[McpServerModel]:
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
                        workspace_id=workspace_id or 0,
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
                        cwd=server.cwd,
                        plugin_root=server.plugin_root,
                        plugin_data_dir=server.plugin_data_dir,
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
