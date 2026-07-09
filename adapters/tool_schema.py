"""
Reflection-based tool schema generation for adapters.

Builds JSON-schema descriptions of adapter actions from method signatures,
type hints, and docstrings so agents can expose adapter functions to models
as structured tools (native function calling for online providers, schema
grounded prompts for local models).
"""
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, get_type_hints

from adapters.base_adapter import BaseAdapter


_PRIMITIVE_TYPE_MAP = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dict: {"type": "object"},
    list: {"type": "array"},
    type(None): {"type": "null"},
}

# Separator used to namespace an action name with its adapter class for
# providers that require flat tool names (OpenAI-compatible endpoints reject
# dots in function names).
QUALIFIED_NAME_SEPARATOR = "__"


@dataclass
class ToolSchema:
    """JSON-schema description of a single adapter action."""
    adapter: str
    action: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.adapter}{QUALIFIED_NAME_SEPARATOR}{self.action}"

    def to_openai_tool(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible `tools` array entry."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Render in the class/function shape used by prompt-based tool calling."""
        return {
            "class": self.adapter,
            "function": self.action,
            "description": self.description,
            "parameters": self.parameters,
        }


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _PRIMITIVE_TYPE_MAP:
        return dict(_PRIMITIVE_TYPE_MAP[annotation])

    origin = get_origin(annotation)
    if origin is Union:
        non_null = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_null) == 1:
            return _annotation_to_schema(non_null[0])
        fragments = [_annotation_to_schema(arg) for arg in non_null]
        fragments = [fragment for fragment in fragments if fragment]
        return {"anyOf": fragments} if fragments else {}
    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        schema: dict[str, Any] = {"type": "array"}
        if args:
            items = _annotation_to_schema(args[0])
            if items:
                schema["items"] = items
        return schema
    if origin is dict:
        return {"type": "object"}
    return {}


def _docstring_summary(func: Any, fallback: str) -> str:
    """First paragraph of a callable's docstring, collapsed to one line."""
    doc = inspect.getdoc(func)
    if not doc:
        return fallback
    first_paragraph = doc.split("\n\n", 1)[0]
    return " ".join(line.strip() for line in first_paragraph.splitlines()).strip()


def build_action_schema(adapter_cls: type, action_name: str) -> ToolSchema:
    """
    Build a ToolSchema for a single adapter action via signature reflection.

    Parameters without defaults are marked required. Unannotated parameters are
    accepted with an unconstrained schema so legacy adapters remain callable.
    """
    method = getattr(adapter_cls, action_name, None)
    if method is None or not callable(method):
        raise ValueError(f"{adapter_cls.__name__} has no callable action '{action_name}'")

    try:
        hints = get_type_hints(method)
    except Exception:
        hints = getattr(method, "__annotations__", {}) or {}

    signature = inspect.signature(method)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, parameter.annotation)
        properties[name] = _annotation_to_schema(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required

    return ToolSchema(
        adapter=adapter_cls.__name__,
        action=action_name,
        description=_docstring_summary(method, fallback=f"{adapter_cls.__name__}.{action_name}"),
        parameters=parameters,
    )


def _visible_actions(adapter: BaseAdapter | type) -> list[str]:
    """
    Determine which actions an adapter exposes.

    Instances keep the existing `enumerate_actions()` visibility contract;
    classes fall back to reflection over public methods.
    """
    if isinstance(adapter, BaseAdapter):
        try:
            actions = adapter.enumerate_actions()
            if actions:
                return list(actions)
        except Exception:
            pass
        adapter = type(adapter)

    return [
        name
        for name, member in inspect.getmembers(adapter, inspect.isfunction)
        if not name.startswith("_") and name != "enumerate_actions"
    ]


def enumerate_tool_schemas(adapter: BaseAdapter | type) -> list[ToolSchema]:
    """Build ToolSchemas for every visible action of an adapter instance or class."""
    adapter_cls = adapter if inspect.isclass(adapter) else type(adapter)
    schemas = []
    for action in _visible_actions(adapter):
        try:
            schemas.append(build_action_schema(adapter_cls, action))
        except ValueError:
            # enumerate_actions may advertise names that don't resolve; skip them
            continue
    return schemas


def render_tool_prompt(schemas: list[ToolSchema]) -> str:
    """Render tool schemas as a compact JSON listing for prompt-based tool calling."""
    return "\n".join(json.dumps(schema.to_prompt_dict(), separators=(", ", ": ")) for schema in schemas)
