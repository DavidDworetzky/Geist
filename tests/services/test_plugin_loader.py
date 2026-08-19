"""Tests for agent plugin discovery and manifest/skill/MCP parsing."""

import json

from app.services.plugin_loader import (
    PluginRegistry,
    discover_plugins,
    load_plugin,
    parse_frontmatter,
)


def _write_plugin(
    root,
    name="demo-plugin",
    manifest_extra=None,
    manifest_location=".claude-plugin/plugin.json",
):
    plugin_root = root / name
    manifest_path = plugin_root / manifest_location
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "version": "1.0.0", "description": "A demo plugin"}
    manifest.update(manifest_extra or {})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return plugin_root


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

    def test_tolerates_unquoted_colons_in_values(self):
        fields, _ = parse_frontmatter(
            "---\ndescription: Use when: pushing, publishing, or verifying\n---\nBody"
        )
        assert fields["description"] == "Use when: pushing, publishing, or verifying"

    def test_strips_matching_quotes(self):
        fields, _ = parse_frontmatter('---\ndescription: "Quoted value"\n---\nBody')
        assert fields["description"] == "Quoted value"

    def test_parses_booleans(self):
        fields, _ = parse_frontmatter("---\ndisable-model-invocation: true\n---\nBody")
        assert fields["disable-model-invocation"] is True

    def test_continuation_lines_join_previous_value(self):
        fields, _ = parse_frontmatter(
            "---\ndescription: >-\n  First part\n  second part\n---\nBody"
        )
        assert fields["description"] == "First part second part"

    def test_document_without_frontmatter_is_all_body(self):
        fields, body = parse_frontmatter("Just a body\nwith lines")
        assert fields == {}
        assert body == "Just a body\nwith lines"

    def test_unterminated_frontmatter_is_all_body(self):
        text = "---\nname: broken\nno closing delimiter"
        fields, body = parse_frontmatter(text)
        assert fields == {}
        assert body == text


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

    def test_accepts_root_level_manifest(self, tmp_path):
        plugin_root = _write_plugin(tmp_path, manifest_location="plugin.json")
        assert load_plugin(plugin_root) is not None

    def test_missing_manifest_is_skipped(self, tmp_path):
        plugin_root = tmp_path / "no-manifest"
        (plugin_root / "skills").mkdir(parents=True)
        assert load_plugin(plugin_root) is None

    def test_invalid_name_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path, name="bad")
        manifest_path = plugin_root / ".claude-plugin" / "plugin.json"
        manifest_path.write_text(json.dumps({"name": "Bad Name!"}), encoding="utf-8")
        assert load_plugin(plugin_root) is None

    def test_unrecognized_manifest_fields_are_ignored(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={"hooks": "./hooks.json", "lspServers": {}, "future": [1, 2]},
        )
        plugin = load_plugin(plugin_root)
        assert plugin is not None and plugin.name == "demo-plugin"

    def test_skill_name_defaults_to_directory(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root, "dir-named", frontmatter="description: From directory")
        plugin = load_plugin(plugin_root)
        assert plugin.skills[0].name == "dir-named"

    def test_skill_without_description_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root, frontmatter="name: nodesc")
        plugin = load_plugin(plugin_root)
        assert plugin.skills == ()

    def test_disable_model_invocation_flag(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(
            plugin_root,
            frontmatter=("name: hidden\ndescription: Manual only\ndisable-model-invocation: true"),
        )
        plugin = load_plugin(plugin_root)
        assert plugin.skills[0].disable_model_invocation is True

    def test_manifest_extra_skills_dirs_add_to_default(self, tmp_path):
        plugin_root = _write_plugin(tmp_path, manifest_extra={"skills": ["./more-skills"]})
        _write_skill(plugin_root)
        extra_dir = plugin_root / "more-skills" / "extra"
        extra_dir.mkdir(parents=True)
        (extra_dir / "SKILL.md").write_text(
            "---\ndescription: Extra skill\n---\nExtra body", encoding="utf-8"
        )
        plugin = load_plugin(plugin_root)
        assert {skill.name for skill in plugin.skills} == {"demo-skill", "extra"}

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


class TestMcpDiscovery:
    def test_parses_dot_mcp_json_with_wrapper(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        (plugin_root / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "database": {
                            "command": "${CLAUDE_PLUGIN_ROOT}/bin/server",
                            "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],
                            "env": {"DATA": "${CLAUDE_PLUGIN_ROOT}/data"},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        plugin = load_plugin(plugin_root)
        assert len(plugin.mcp_servers) == 1
        server = plugin.mcp_servers[0]
        assert server.transport == "stdio"
        assert server.command == f"{plugin_root}/bin/server"
        assert server.args == ("--config", f"{plugin_root}/config.json")
        assert server.env == {"DATA": f"{plugin_root}/data"}
        assert server.qualified_name == "demo-plugin.database"

    def test_parses_http_server(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        (plugin_root / "mcp.json").write_text(
            json.dumps(
                {"mcpServers": {"api": {"url": "https://example.com/mcp", "headers": {"X": "1"}}}}
            ),
            encoding="utf-8",
        )
        plugin = load_plugin(plugin_root)
        server = plugin.mcp_servers[0]
        assert server.transport == "http"
        assert server.url == "https://example.com/mcp"
        assert server.headers == {"X": "1"}

    def test_manifest_inline_mcp_servers(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={"mcpServers": {"inline": {"command": "run-server"}}},
        )
        plugin = load_plugin(plugin_root)
        assert plugin.mcp_servers[0].name == "inline"

    def test_manifest_mcp_path_reference(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path, manifest_extra={"mcpServers": "./config/servers.json"}
        )
        config_dir = plugin_root / "config"
        config_dir.mkdir()
        (config_dir / "servers.json").write_text(
            json.dumps({"mcpServers": {"filed": {"command": "srv"}}}), encoding="utf-8"
        )
        plugin = load_plugin(plugin_root)
        assert plugin.mcp_servers[0].name == "filed"

    def test_server_without_command_or_url_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path, manifest_extra={"mcpServers": {"broken": {"args": ["x"]}}}
        )
        plugin = load_plugin(plugin_root)
        assert plugin.mcp_servers == ()

    def test_non_http_url_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path, manifest_extra={"mcpServers": {"bad": {"url": "file:///etc/passwd"}}}
        )
        plugin = load_plugin(plugin_root)
        assert plugin.mcp_servers == ()

    def test_http_server_with_non_object_headers_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={
                "mcpServers": {"bad": {"url": "https://example.com/mcp", "headers": []}}
            },
        )

        plugin = load_plugin(plugin_root)

        assert plugin.mcp_servers == ()

    def test_stdio_server_with_invalid_args_or_env_is_skipped(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={
                "mcpServers": {
                    "bad-args": {"command": "server", "args": "--unsafe"},
                    "bad-env": {"command": "server", "env": ["TOKEN=value"]},
                }
            },
        )

        plugin = load_plugin(plugin_root)

        assert plugin.mcp_servers == ()

    def test_http_header_injection_is_rejected(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={
                "mcpServers": {
                    "bad": {
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "ok\r\nX-Injected: yes"},
                    }
                }
            },
        )

        assert load_plugin(plugin_root).mcp_servers == ()

    def test_stdio_environment_requires_string_values(self, tmp_path):
        plugin_root = _write_plugin(
            tmp_path,
            manifest_extra={"mcpServers": {"bad": {"command": "server", "env": {"X": 1}}}},
        )

        assert load_plugin(plugin_root).mcp_servers == ()


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
        _write_plugin(tmp_path, "alpha", manifest_extra={"mcpServers": {"one": {"command": "srv"}}})
        _write_plugin(tmp_path, "beta", manifest_extra={"mcpServers": {"two": {"command": "srv"}}})
        registry = PluginRegistry(tmp_path)

        monkeypatch.delenv("GEIST_ENABLED_PLUGINS", raising=False)
        assert registry.enabled_mcp_server_models() == []

        monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "alpha")
        models = registry.enabled_mcp_server_models()
        assert [model.name for model in models] == ["alpha.one"]
        assert models[0].mcp_server_id < 0
        assert models[0].enabled is True

        monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "alpha, beta")
        assert {model.name for model in registry.enabled_mcp_server_models()} == {
            "alpha.one",
            "beta.two",
        }

    def test_synthetic_server_ids_are_stable(self, tmp_path, monkeypatch):
        _write_plugin(tmp_path, "alpha", manifest_extra={"mcpServers": {"one": {"command": "srv"}}})
        monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "alpha")
        registry = PluginRegistry(tmp_path)
        first = registry.enabled_mcp_server_models()[0].mcp_server_id
        second = registry.enabled_mcp_server_models()[0].mcp_server_id
        assert first == second
