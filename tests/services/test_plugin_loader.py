"""Tests for agent plugin discovery and manifest/skill/MCP parsing."""

import json

import pytest

from app.services.plugin_loader import (
    MCP_SCHEMA_ID,
    PLUGIN_SCHEMA_ID,
    PluginRegistry,
    discover_plugins,
    load_plugin,
    parse_frontmatter,
)


def _write_plugin(
    root,
    name="demo-plugin",
    manifest_extra=None,
    manifest_location="plugin.json",
):
    plugin_root = root / name
    manifest_path = plugin_root / manifest_location
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "$schema": PLUGIN_SCHEMA_ID,
        "name": name,
        "version": "1.0.0",
        "description": "A demo plugin",
    }
    manifest.update(manifest_extra or {})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return plugin_root


def _write_mcp(plugin_root, servers, *, schema=MCP_SCHEMA_ID, extra=None):
    document = {"$schema": schema, "mcpServers": servers}
    document.update(extra or {})
    (plugin_root / "mcp.json").write_text(json.dumps(document), encoding="utf-8")


def _write_skill(plugin_root, skill_name="demo-skill", frontmatter=None, body="Do the thing."):
    skill_dir = plugin_root / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = frontmatter or f"name: {skill_name}\ndescription: Demo skill description"
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return skill_dir


class TestParseFrontmatter:
    def test_parses_simple_fields_and_body(self):
        fields, body = parse_frontmatter(
            "---\nname: my-skill\ndescription: Reviews code\n---\n\nInstructions here."
        )
        assert fields == {"name": "my-skill", "description": "Reviews code"}
        assert body == "Instructions here."

    def test_rejects_invalid_yaml(self):
        with pytest.raises(ValueError, match="invalid YAML"):
            parse_frontmatter(
                "---\ndescription: Use when: pushing, publishing, or verifying\n---\nBody"
            )

    def test_strips_matching_quotes(self):
        fields, _ = parse_frontmatter('---\ndescription: "Quoted value"\n---\nBody')
        assert fields["description"] == "Quoted value"

    def test_parses_nested_metadata(self):
        fields, _ = parse_frontmatter(
            "---\nmetadata:\n  author: example\n  version: '1'\n---\nBody"
        )
        assert fields["metadata"] == {"author": "example", "version": "1"}

    def test_rejects_duplicate_frontmatter_keys(self):
        with pytest.raises(ValueError, match="duplicate key"):
            parse_frontmatter("---\nname: first\nname: second\ndescription: Duplicate\n---\nBody")

    def test_continuation_lines_join_previous_value(self):
        fields, _ = parse_frontmatter(
            "---\ndescription: >-\n  First part\n  second part\n---\nBody"
        )
        assert fields["description"] == "First part second part"

    def test_document_without_frontmatter_is_rejected(self):
        with pytest.raises(ValueError, match="must start"):
            parse_frontmatter("Just a body\nwith lines")

    def test_unterminated_frontmatter_is_rejected(self):
        with pytest.raises(ValueError, match="not closed"):
            parse_frontmatter("---\nname: broken\nno closing delimiter")


class TestLoadPlugin:
    def test_loads_manifest_and_skills(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root)
        plugin = load_plugin(plugin_root)
        assert plugin is not None
        assert plugin.name == "demo-plugin"
        assert plugin.version == "1.0.0"
        assert len(plugin.skills) == 1
        skill = plugin.skills[0]
        assert skill.qualified_name == "demo-plugin:demo-skill"
        assert skill.description == "Demo skill description"
        assert skill.load_body() == "Do the thing."

    def test_legacy_manifest_cannot_replace_root_manifest(self, tmp_path):
        plugin_root = _write_plugin(tmp_path, manifest_location=".claude-plugin/plugin.json")
        assert load_plugin(plugin_root) is None

    def test_missing_manifest_is_skipped(self, tmp_path):
        plugin_root = tmp_path / "no-manifest"
        (plugin_root / "skills").mkdir(parents=True)
        assert load_plugin(plugin_root) is None

    def test_invalid_name_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path, name="bad")
        manifest_path = plugin_root / "plugin.json"
        manifest_path.write_text(
            json.dumps({"$schema": PLUGIN_SCHEMA_ID, "name": "Bad Name!"}),
            encoding="utf-8",
        )
        assert load_plugin(plugin_root) is None

    @pytest.mark.parametrize("name", ["acme.tools", "a", "my-plugin"])
    def test_accepts_standard_plugin_names(self, tmp_path, name):
        assert load_plugin(_write_plugin(tmp_path, name=name)) is not None

    @pytest.mark.parametrize(
        "name",
        ["bad_name", "-bad", "bad-", "bad--name", "bad..name", "a" * 65],
    )
    def test_rejects_nonstandard_plugin_names(self, tmp_path, name):
        assert load_plugin(_write_plugin(tmp_path, name=name)) is None

    def test_requires_supported_schema(self, tmp_path):
        assert (
            load_plugin(_write_plugin(tmp_path, manifest_extra={"$schema": "unsupported"})) is None
        )

    def test_missing_schema_is_rejected(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        (plugin_root / "plugin.json").write_text(
            json.dumps({"name": "demo-plugin"}), encoding="utf-8"
        )
        assert load_plugin(plugin_root) is None

    def test_rejects_invalid_manifest_metadata_type(self, tmp_path):
        assert load_plugin(_write_plugin(tmp_path, manifest_extra={"version": 1})) is None

    def test_non_object_extensions_are_reported_and_ignored(self, tmp_path, caplog):
        plugin = load_plugin(_write_plugin(tmp_path, manifest_extra={"extensions": "bad"}))
        assert plugin is not None
        assert "non-object extensions" in caplog.text

    def test_manifest_symlink_escape_is_rejected(self, tmp_path):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text(
            json.dumps({"$schema": PLUGIN_SCHEMA_ID, "name": "demo-plugin"}),
            encoding="utf-8",
        )
        (plugin_root / "plugin.json").symlink_to(outside)
        assert load_plugin(plugin_root) is None

    def test_unrecognized_manifest_fields_are_ignored(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={"hooks": "./hooks.json", "lspServers": {}, "future": [1, 2]},
        )
        plugin = load_plugin(plugin_root)
        assert plugin is not None and plugin.name == "demo-plugin"

    def test_skill_name_is_required(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root, "dir-named", frontmatter="description: From directory")
        plugin = load_plugin(plugin_root)
        assert plugin.skills == ()

    def test_skill_without_description_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root, frontmatter="name: nodesc")
        plugin = load_plugin(plugin_root)
        assert plugin.skills == ()

    def test_skill_name_must_match_parent_directory(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(
            plugin_root, "directory-name", frontmatter="name: other\ndescription: Mismatch"
        )
        assert load_plugin(plugin_root).skills == ()

    def test_invalid_skill_does_not_disable_valid_sibling(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root, "valid")
        _write_skill(plugin_root, "invalid", frontmatter="description: Missing name")
        assert [skill.name for skill in load_plugin(plugin_root).skills] == ["valid"]

    def test_unknown_skill_frontmatter_field_is_rejected(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(
            plugin_root,
            frontmatter=("name: hidden\ndescription: Manual only\ndisable-model-invocation: true"),
        )
        plugin = load_plugin(plugin_root)
        assert plugin.skills == ()

    def test_manifest_extra_skills_dirs_are_ignored(self, tmp_path):
        plugin_root = _write_plugin(tmp_path, manifest_extra={"skills": ["./more-skills"]})
        _write_skill(plugin_root)
        extra_dir = plugin_root / "more-skills" / "extra"
        extra_dir.mkdir(parents=True)
        (extra_dir / "SKILL.md").write_text(
            "---\ndescription: Extra skill\n---\nExtra body", encoding="utf-8"
        )
        plugin = load_plugin(plugin_root)
        assert {skill.name for skill in plugin.skills} == {"demo-skill"}

    def test_skills_path_escaping_plugin_root_is_rejected(self, tmp_path):
        outside = tmp_path / "outside" / "escaped"
        outside.mkdir(parents=True)
        (outside / "SKILL.md").write_text("---\ndescription: Escaped\n---\nBody", encoding="utf-8")
        plugin_root = _write_plugin(tmp_path, manifest_extra={"skills": ["../outside"]})
        plugin = load_plugin(plugin_root)
        assert plugin.skills == ()

    def test_skill_symlink_escaping_plugin_root_is_rejected(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        outside = tmp_path / "outside.md"
        outside.write_text("---\ndescription: Escaped\n---\nBody", encoding="utf-8")
        skill_dir = plugin_root / "skills" / "escaped"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").symlink_to(outside)

        plugin = load_plugin(plugin_root)

        assert plugin.skills == ()

    def test_oversized_skill_file_is_rejected(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        skill_dir = plugin_root / "skills" / "large"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("x" * (1024 * 1024 + 1), encoding="utf-8")

        assert load_plugin(plugin_root).skills == ()

    def test_skills_wrong_filesystem_kind_does_not_disable_mcp(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        (plugin_root / "skills").write_text("not a directory", encoding="utf-8")
        _write_mcp(plugin_root, {"valid": {"type": "stdio", "command": "server"}})
        plugin = load_plugin(plugin_root)
        assert plugin.skills == ()
        assert [server.name for server in plugin.mcp_servers] == ["valid"]


class TestMcpDiscovery:
    def test_dot_mcp_json_is_not_a_portable_component(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        (plugin_root / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"database": {"command": "server"}}}),
            encoding="utf-8",
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_parses_standard_stdio_server_and_plugin_variables(self, tmp_path, monkeypatch):
        plugin_root = _write_plugin(tmp_path)
        data_root = tmp_path / "plugin-data"
        monkeypatch.setenv("GEIST_PLUGIN_DATA_DIR", str(data_root))
        _write_mcp(
            plugin_root,
            {
                "database": {
                    "type": "stdio",
                    "command": "./bin/server",
                    "args": ["--config", "${PLUGIN_ROOT}/config.json"],
                    "env": {"DATA": "${PLUGIN_DATA}/state"},
                    "cwd": "${PLUGIN_DATA}/work",
                }
            },
        )
        server = load_plugin(plugin_root).mcp_servers[0]
        assert server.transport == "stdio"
        assert server.command == f"{plugin_root}/bin/server"
        assert server.args == ("--config", f"{plugin_root}/config.json")
        assert server.env == {"DATA": f"{data_root}/demo-plugin/state"}
        assert server.cwd == f"{data_root}/demo-plugin/work"
        assert server.plugin_root == str(plugin_root)
        assert server.plugin_data_dir == f"{data_root}/demo-plugin"
        assert server.qualified_name == "demo-plugin.database"

    def test_cwd_expands_every_standard_placeholder_occurrence(self, tmp_path, monkeypatch):
        plugin_root = _write_plugin(tmp_path)
        data_root = tmp_path / "plugin-data"
        monkeypatch.setenv("GEIST_PLUGIN_DATA_DIR", str(data_root))
        _write_mcp(
            plugin_root,
            {
                "database": {
                    "type": "stdio",
                    "command": "server",
                    "cwd": "${PLUGIN_DATA}/nested-${PLUGIN_DATA}",
                }
            },
        )

        server = load_plugin(plugin_root).mcp_servers[0]

        assert "${PLUGIN_DATA}" not in server.cwd
        assert server.cwd == f"{data_root}/demo-plugin/nested-{data_root}/demo-plugin"

    def test_parses_http_server(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {
                "api": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"X": "1"},
                }
            },
        )
        plugin = load_plugin(plugin_root)
        server = plugin.mcp_servers[0]
        assert server.transport == "http"
        assert server.url == "https://example.com/mcp"
        assert server.headers == {"X": "1"}

    def test_manifest_inline_mcp_servers_are_ignored(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={"mcpServers": {"inline": {"type": "stdio", "command": "run-server"}}},
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_manifest_mcp_path_reference_is_ignored(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path, manifest_extra={"mcpServers": "./config/servers.json"}
        )
        config_dir = plugin_root / "config"
        config_dir.mkdir()
        (config_dir / "servers.json").write_text(
            json.dumps({"mcpServers": {"filed": {"command": "srv"}}}), encoding="utf-8"
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    @pytest.mark.parametrize("document", [{}, {"$schema": MCP_SCHEMA_ID}, {"mcpServers": {}}])
    def test_invalid_top_level_mcp_document_disables_component(self, tmp_path, document):
        plugin_root = _write_plugin(tmp_path)
        (plugin_root / "mcp.json").write_text(json.dumps(document), encoding="utf-8")
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_unknown_top_level_mcp_field_disables_component(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(plugin_root, {}, extra={"future": True})
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_unsupported_mcp_schema_disables_component_only(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root)
        _write_mcp(
            plugin_root,
            {"valid": {"type": "stdio", "command": "server"}},
            schema="unsupported",
        )
        plugin = load_plugin(plugin_root)
        assert [skill.name for skill in plugin.skills] == ["demo-skill"]
        assert plugin.mcp_servers == ()

    def test_server_without_command_or_url_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(plugin_root, {"broken": {"type": "stdio", "args": ["x"]}})
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_non_http_url_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {"bad": {"type": "streamable-http", "url": "file:///etc/passwd"}},
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_non_loopback_http_url_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {"bad": {"type": "streamable-http", "url": "http://example.com/mcp"}},
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    @pytest.mark.parametrize(
        "url",
        [
            "https://user@example.com/mcp",
            "https://example.com/mcp#fragment",
            "https://exa mple.com/mcp",
            "https://example.com:bad/mcp",
            "//example.com/mcp",
        ],
    )
    def test_invalid_remote_url_forms_are_skipped(self, tmp_path, url):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(plugin_root, {"bad": {"type": "streamable-http", "url": url}})
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_loopback_http_url_is_allowed(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {"local": {"type": "streamable-http", "url": "http://127.0.0.1:8080/mcp"}},
        )
        assert load_plugin(plugin_root).mcp_servers[0].name == "local"

    def test_http_server_with_non_object_headers_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {
                "bad": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": [],
                }
            },
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_stdio_server_with_invalid_args_or_env_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {
                "bad-args": {"type": "stdio", "command": "server", "args": "--unsafe"},
                "bad-env": {"type": "stdio", "command": "server", "env": ["TOKEN=value"]},
            },
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_http_header_injection_is_rejected(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {
                "bad": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "ok\r\nX-Injected: yes"},
                }
            },
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_literal_header_values_are_preserved_without_expansion(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"X-Plugin-Root": "${PLUGIN_ROOT}"},
                }
            },
        )
        assert load_plugin(plugin_root).mcp_servers[0].headers == {
            "X-Plugin-Root": "${PLUGIN_ROOT}"
        }

    def test_case_insensitive_duplicate_headers_are_rejected(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {
                "bad": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"X-Tenant": "one", "x-tenant": "two"},
                }
            },
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_stdio_environment_requires_string_values(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {"bad": {"type": "stdio", "command": "server", "env": {"X": 1}}},
        )
        assert load_plugin(plugin_root).mcp_servers == ()

    @pytest.mark.parametrize(
        "server",
        [
            {"type": "future", "command": "server"},
            {"type": "stdio", "command": "server", "unknown": True},
            {"type": "stdio", "command": "../outside/server"},
            {"type": "stdio", "command": "server", "env": {"PLUGIN_ROOT": "override"}},
            {"type": "stdio", "command": "server", "cwd": "${PLUGIN_DATA}/../escape"},
        ],
    )
    def test_invalid_closed_stdio_variants_are_skipped(self, tmp_path, server):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(plugin_root, {"bad": server})
        assert load_plugin(plugin_root).mcp_servers == ()

    def test_unsupported_sse_entry_does_not_disable_valid_sibling(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_mcp(
            plugin_root,
            {
                "legacy": {"type": "sse", "url": "https://example.com/sse"},
                "valid": {"type": "stdio", "command": "server"},
            },
        )
        assert [server.name for server in load_plugin(plugin_root).mcp_servers] == ["valid"]


class TestPluginRegistry:
    def test_discovers_multiple_plugins_and_skips_invalid(self, tmp_path):
        _write_skill(_write_plugin(tmp_path, "alpha"))
        _write_skill(_write_plugin(tmp_path, "beta"))
        (tmp_path / "not-a-plugin").mkdir()
        (tmp_path / ".hidden").mkdir()
        plugins = discover_plugins(tmp_path)
        assert [plugin.name for plugin in plugins] == ["alpha", "beta"]

    def test_missing_plugin_dir_is_empty(self, tmp_path):
        assert discover_plugins(tmp_path / "missing") == []

    def test_symlinked_plugin_root_is_skipped(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        _write_plugin(outside, "linked")
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "linked").symlink_to(outside / "linked", target_is_directory=True)

        assert discover_plugins(plugin_dir) == []

    def test_registry_find_skill(self, tmp_path):
        _write_skill(_write_plugin(tmp_path, "alpha"))
        registry = PluginRegistry(tmp_path)
        skill = registry.find_skill("alpha:demo-skill")
        assert skill is not None
        assert registry.find_skill("alpha:unknown") is None

    def test_registry_refresh_picks_up_new_plugins(self, tmp_path):
        registry = PluginRegistry(tmp_path)
        assert registry.plugins() == []
        _write_skill(_write_plugin(tmp_path, "late"))
        assert registry.plugins() == []  # cached scan
        registry.refresh()
        assert [plugin.name for plugin in registry.plugins()] == ["late"]

    def test_enabled_mcp_server_models_respect_env(self, tmp_path, monkeypatch):
        alpha = _write_plugin(tmp_path, "alpha")
        beta = _write_plugin(tmp_path, "beta")
        _write_mcp(alpha, {"one": {"type": "stdio", "command": "srv"}})
        _write_mcp(beta, {"two": {"type": "stdio", "command": "srv"}})
        registry = PluginRegistry(tmp_path)

        monkeypatch.delenv("GEIST_ENABLED_PLUGINS", raising=False)
        assert registry.enabled_mcp_server_models() == []

        monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "alpha")
        models = registry.enabled_mcp_server_models()
        assert [model.name for model in models] == ["alpha.one"]
        assert models[0].mcp_server_id < 0
        assert models[0].enabled is True
        assert models[0].cwd == str(alpha)
        assert models[0].plugin_root == str(alpha)

        monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "alpha, beta")
        assert {model.name for model in registry.enabled_mcp_server_models()} == {
            "alpha.one",
            "beta.two",
        }

    def test_synthetic_server_ids_are_stable(self, tmp_path, monkeypatch):
        plugin_root = _write_plugin(tmp_path, "alpha")
        _write_mcp(plugin_root, {"one": {"type": "stdio", "command": "srv"}})
        monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "alpha")
        registry = PluginRegistry(tmp_path)
        first = registry.enabled_mcp_server_models()[0].mcp_server_id
        second = registry.enabled_mcp_server_models()[0].mcp_server_id
        assert first == second
