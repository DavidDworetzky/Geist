"""Tests for reflection-based tool schema generation."""

import pytest

from adapters.base_adapter import BaseAdapter
from adapters.markdown_file_adapter import MarkdownFileAdapter
from adapters.tool_schema import (
    build_action_schema,
    enumerate_tool_schemas,
    render_tool_prompt,
)


class TypedAdapter(BaseAdapter):
    """Adapter with a variety of parameter annotations for schema tests."""

    def enumerate_actions(self):
        return ["typed_action", "untyped_action"]

    def typed_action(
        self,
        name: str,
        count: int,
        ratio: float,
        enabled: bool,
        tags: list[str],
        config: dict[str, str],
        limit: int | None = None,
        **kwargs,
    ) -> str:
        """Do a typed thing.

        Longer explanation that should not appear in the summary.
        """
        return name

    def untyped_action(self, anything, other=None):
        return anything


class EmptyAdapter(BaseAdapter):
    def enumerate_actions(self):
        return []

    def public_helper(self):
        return "must not be exposed"


class BrokenAdapter(BaseAdapter):
    def enumerate_actions(self):
        raise RuntimeError("declaration failed")

    def public_helper(self):
        return "must not be exposed"


def test_build_action_schema_maps_types_and_required():
    schema = build_action_schema(TypedAdapter, "typed_action")

    assert schema.adapter == "TypedAdapter"
    assert schema.action == "typed_action"
    assert schema.description == "Do a typed thing."

    properties = schema.parameters["properties"]
    assert properties["name"] == {"type": "string"}
    assert properties["count"] == {"type": "integer"}
    assert properties["ratio"] == {"type": "number"}
    assert properties["enabled"] == {"type": "boolean"}
    assert properties["tags"] == {"type": "array", "items": {"type": "string"}}
    assert properties["config"] == {"type": "object"}
    # Optional[int] unwraps to integer and is not required
    assert properties["limit"] == {"type": "integer"}
    # **kwargs and self are excluded
    assert "kwargs" not in properties
    assert "self" not in properties

    assert schema.parameters["required"] == ["name", "count", "ratio", "enabled", "tags", "config"]


def test_build_action_schema_tolerates_untyped_parameters():
    schema = build_action_schema(TypedAdapter, "untyped_action")

    assert schema.parameters["properties"]["anything"] == {}
    assert schema.parameters["required"] == ["anything"]
    assert "other" not in schema.parameters["required"]


def test_build_action_schema_rejects_missing_action():
    with pytest.raises(ValueError):
        build_action_schema(TypedAdapter, "does_not_exist")


def test_enumerate_tool_schemas_uses_enumerate_actions(tmp_path):
    adapter = MarkdownFileAdapter(file_root=str(tmp_path))
    schemas = enumerate_tool_schemas(adapter)

    names = {schema.action for schema in schemas}
    assert names == {"read_file", "write_file", "get_files"}

    read_file = next(schema for schema in schemas if schema.action == "read_file")
    assert read_file.adapter == "MarkdownFileAdapter"
    assert read_file.parameters["required"] == ["filename"]
    assert read_file.description.startswith("Read content from a markdown file")


@pytest.mark.parametrize("adapter", [EmptyAdapter(), BrokenAdapter()])
def test_enumerate_tool_schemas_fails_closed_for_empty_or_broken_declarations(adapter):
    assert enumerate_tool_schemas(adapter) == []


def test_qualified_name_and_openai_tool_shape(tmp_path):
    adapter = MarkdownFileAdapter(file_root=str(tmp_path))
    schema = next(
        schema for schema in enumerate_tool_schemas(adapter) if schema.action == "write_file"
    )

    assert schema.qualified_name == "MarkdownFileAdapter__write_file"

    tool = schema.to_openai_tool()
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "MarkdownFileAdapter__write_file"
    assert tool["function"]["parameters"]["properties"]["content"] == {"type": "string"}


def test_render_tool_prompt_lists_every_tool(tmp_path):
    adapter = MarkdownFileAdapter(file_root=str(tmp_path))
    rendered = render_tool_prompt(enumerate_tool_schemas(adapter))

    assert '"class": "MarkdownFileAdapter"' in rendered
    assert '"function": "read_file"' in rendered
    assert '"function": "write_file"' in rendered
