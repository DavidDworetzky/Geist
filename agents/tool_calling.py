"""
Structured tool calling shared by agents.

Provides robust parsing of model-emitted tool calls (JSON embedded in prose,
markdown fences, alias keys), schema validation with type coercion, and a
dispatcher that executes adapter actions and journals results to the agent's
function log. Used natively by OnlineAgent (provider `tools` API) and as the
schema-grounded prompt path for LocalAgent.
"""
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from adapters.tool_schema import (
    QUALIFIED_NAME_SEPARATOR,
    ToolSchema,
    enumerate_tool_schemas,
    render_tool_prompt,
)


logger = logging.getLogger(__name__)

STRUCTURED_TOOL_SYSTEM_PROMPT = (
    "You are an agent that completes tasks by calling tools. "
    "Respond with ONLY a single JSON object of the form "
    '{"class": "AdapterName", "function": "action_name", "parameters": {...}} '
    "using one of the listed tools. Do not include any other text."
)


class ToolCallError(Exception):
    """Raised when a tool call cannot be parsed or validated."""


@dataclass
class ToolCall:
    """A parsed request to invoke one adapter action."""
    adapter: str
    function: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    @classmethod
    def from_qualified_name(cls, name: str, arguments: dict[str, Any], raw: str = "") -> "ToolCall":
        adapter, separator, function = name.partition(QUALIFIED_NAME_SEPARATOR)
        if not separator or not adapter or not function:
            raise ToolCallError(
                f"Tool name '{name}' is not namespaced as Adapter{QUALIFIED_NAME_SEPARATOR}action"
            )
        return cls(adapter=adapter, function=function, arguments=arguments, raw=raw)


@dataclass
class ToolResult:
    """Outcome of dispatching a single tool call."""
    call: ToolCall | None
    success: bool
    result: Any = None
    error: str | None = None

    def to_content(self) -> str:
        """Serialize for feeding back to a model as a tool message."""
        payload = {"success": self.success}
        if self.success:
            try:
                json.dumps(self.result)
                payload["result"] = self.result
            except (TypeError, ValueError):
                payload["result"] = str(self.result)
        else:
            payload["error"] = self.error
        return json.dumps(payload)

    def to_log_entry(self) -> dict[str, Any]:
        entry = {
            "adapter": self.call.adapter if self.call else None,
            "function": self.call.function if self.call else None,
            "arguments": self.call.arguments if self.call else None,
            "success": self.success,
        }
        if self.success:
            entry["result"] = str(self.result)
        else:
            entry["error"] = self.error
        return entry


@dataclass
class ToolCompletion:
    """Result of a tool-augmented completion loop."""
    content: str | None
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0
    used_native_tools: bool = False


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_candidates(text: str) -> list[str]:
    """
    Extract candidate JSON object strings from model output.

    Handles markdown fences and balanced top-level objects embedded in prose.
    """
    candidates: list[str] = []
    for fenced in _JSON_FENCE_RE.findall(text):
        if fenced.startswith("{"):
            candidates.append(fenced)

    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:index + 1])
                start = None

    # Preserve order but drop duplicates (a fenced block is also found by the scan)
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def parse_tool_call(text: str) -> ToolCall:
    """
    Parse a tool call from model output.

    Accepts the canonical {"class", "function", "parameters"} shape as well as
    common aliases ("adapter", "action"/"name", "arguments") and qualified
    names ("Adapter__action"). Raises ToolCallError with an actionable message
    suitable for feeding back to the model on retry.
    """
    if not text or not text.strip():
        raise ToolCallError("Empty completion; expected a JSON tool call object.")

    errors: list[str] = []
    for candidate in extract_json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON ({exc.msg})")
            continue
        if not isinstance(data, dict):
            continue

        adapter = data.get("class") or data.get("adapter")
        function = data.get("function") or data.get("action") or data.get("name")
        arguments = data.get("parameters", data.get("arguments"))

        if isinstance(function, str) and QUALIFIED_NAME_SEPARATOR in function and not adapter:
            adapter, _, function = function.partition(QUALIFIED_NAME_SEPARATOR)

        if not adapter or not function:
            errors.append("JSON object is missing 'class' and/or 'function' keys")
            continue
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            errors.append("'parameters' must be a JSON object of argument name/value pairs")
            continue
        return ToolCall(adapter=str(adapter), function=str(function), arguments=arguments, raw=candidate)

    if errors:
        raise ToolCallError("; ".join(errors))
    raise ToolCallError("No JSON object found in completion; respond with only the tool call JSON.")


_JSON_TYPE_CHECKS = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

_TRUE_STRINGS = {"true", "yes", "1"}
_FALSE_STRINGS = {"false", "no", "0"}


def _coerce_value(value: Any, expected_type: str) -> Any:
    """Coerce a value to the expected JSON type, raising ValueError if impossible."""
    expected = _JSON_TYPE_CHECKS[expected_type]
    if isinstance(value, expected) and not (expected_type in ("integer", "number") and isinstance(value, bool)):
        return float(value) if expected_type == "number" and isinstance(value, int) else value

    if expected_type == "number" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if expected_type == "integer":
            return int(stripped)
        if expected_type == "number":
            return float(stripped)
        if expected_type == "boolean":
            lowered = stripped.lower()
            if lowered in _TRUE_STRINGS:
                return True
            if lowered in _FALSE_STRINGS:
                return False
            raise ValueError(f"cannot interpret '{value}' as boolean")
        if expected_type in ("array", "object"):
            parsed = json.loads(stripped)
            if isinstance(parsed, expected):
                return parsed
            raise ValueError(f"parsed JSON is not of type {expected_type}")
    if expected_type == "string" and isinstance(value, (int, float, bool)):
        return json.dumps(value) if isinstance(value, bool) else str(value)
    raise ValueError(f"expected {expected_type}, got {type(value).__name__}")


def find_schema(call: ToolCall, schemas: list[ToolSchema]) -> ToolSchema | None:
    """Find the schema matching a call, tolerating case mismatches."""
    for schema in schemas:
        if schema.adapter == call.adapter and schema.action == call.function:
            return schema
    for schema in schemas:
        if (schema.adapter.lower() == call.adapter.lower()
                and schema.action.lower() == call.function.lower()):
            return schema
    return None


def validate_tool_call(call: ToolCall, schemas: list[ToolSchema]) -> ToolCall:
    """
    Validate a parsed call against the available tool schemas.

    Returns a normalized copy (canonical adapter/action casing, coerced argument
    types, unknown arguments dropped). Raises ToolCallError describing every
    problem found so the message can be fed back to the model.
    """
    schema = find_schema(call, schemas)
    if schema is None:
        available = ", ".join(s.qualified_name for s in schemas) or "none"
        raise ToolCallError(
            f"Unknown tool '{call.adapter}.{call.function}'. Available tools: {available}"
        )

    properties = schema.parameters.get("properties", {})
    required = schema.parameters.get("required", [])
    errors: list[str] = []
    normalized: dict[str, Any] = {}

    for name, value in call.arguments.items():
        if name not in properties:
            logger.warning(
                "Dropping unknown argument '%s' for tool %s", name, schema.qualified_name
            )
            continue
        expected_type = properties[name].get("type")
        if expected_type in _JSON_TYPE_CHECKS:
            try:
                normalized[name] = _coerce_value(value, expected_type)
            except (ValueError, TypeError, json.JSONDecodeError):
                errors.append(
                    f"Argument '{name}' must be of type {expected_type}, got {value!r}"
                )
        else:
            normalized[name] = value

    for name in required:
        if name not in normalized and not any(error.startswith(f"Argument '{name}'") for error in errors):
            errors.append(f"Missing required argument '{name}'")

    if errors:
        raise ToolCallError("; ".join(errors))

    return ToolCall(
        adapter=schema.adapter,
        function=schema.action,
        arguments=normalized,
        raw=call.raw,
    )


class ToolDispatcher:
    """
    Validates and executes tool calls against initialized adapter instances.

    Execution failures are captured as unsuccessful ToolResults rather than
    raised, so one bad call cannot crash an agent loop. Every dispatch is
    journaled to the provided function log.
    """

    def __init__(self, adapter_wrappers: list[Any], schemas: list[ToolSchema] | None = None,
                 function_log: list[Any] | None = None):
        self._instances = {wrapper.name: wrapper.instance for wrapper in adapter_wrappers}
        if schemas is None:
            schemas = []
            for wrapper in adapter_wrappers:
                schemas.extend(enumerate_tool_schemas(wrapper.instance))
        self.schemas = schemas
        self.function_log = function_log if function_log is not None else []

    def dispatch(self, call: ToolCall) -> ToolResult:
        try:
            call = validate_tool_call(call, self.schemas)
        except ToolCallError as exc:
            result = ToolResult(call=call, success=False, error=str(exc))
            self._log(result)
            return result

        instance = self._instances.get(call.adapter)
        if instance is None:
            result = ToolResult(
                call=call,
                success=False,
                error=f"Adapter '{call.adapter}' is not initialized for this agent",
            )
            self._log(result)
            return result

        try:
            output = getattr(instance, call.function)(**call.arguments)
            result = ToolResult(call=call, success=True, result=output)
        except Exception as exc:
            logger.exception("Tool %s.%s raised", call.adapter, call.function)
            result = ToolResult(
                call=call,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._log(result)
        return result

    def dispatch_text(self, text: str) -> ToolResult:
        """Parse model output and dispatch it, capturing parse failures as results."""
        try:
            call = parse_tool_call(text)
        except ToolCallError as exc:
            result = ToolResult(call=None, success=False, error=str(exc))
            self._log(result)
            return result
        return self.dispatch(call)

    def _log(self, result: ToolResult) -> None:
        try:
            self.function_log.append(json.dumps(result.to_log_entry(), default=str))
        except Exception:
            logger.warning("Failed to append tool result to function log", exc_info=True)


def run_prompt_tool_call(
    complete_fn: Callable[[str, str], str],
    schemas: list[ToolSchema],
    dispatcher: ToolDispatcher,
    task_prompt: str,
    max_attempts: int = 3,
) -> ToolResult:
    """
    Prompt-based tool calling with schema visibility and validation feedback.

    Used by local models (and as the fallback for online providers without
    native tool support): the model sees the reflected tool schemas, its output
    is parsed and validated, and validation errors are fed back on retry.

    Args:
        complete_fn: Callable of (prompt, system_prompt) returning model text.
        schemas: Tool schemas visible to the model.
        dispatcher: Dispatcher used to execute the validated call.
        task_prompt: Description of the task the tool call should accomplish.
        max_attempts: Attempts before giving up (validation feedback in between).
    """
    if not schemas:
        return ToolResult(call=None, success=False, error="No tools are available to this agent")

    base_prompt = (
        f"{task_prompt}\n\nAVAILABLE_TOOLS (JSON schemas):\n{render_tool_prompt(schemas)}\n\n"
        'Respond with ONLY one JSON object: {"class": ..., "function": ..., "parameters": {...}}'
    )
    prompt = base_prompt
    last_error = "no attempts made"

    for _ in range(max_attempts):
        text = complete_fn(prompt, STRUCTURED_TOOL_SYSTEM_PROMPT)
        try:
            call = parse_tool_call(text or "")
            call = validate_tool_call(call, schemas)
        except ToolCallError as exc:
            last_error = str(exc)
            logger.info("Invalid tool call attempt, retrying with feedback: %s", last_error)
            prompt = (
                f"{base_prompt}\n\nYour previous response was invalid: {last_error}\n"
                "Respond again with ONLY the corrected JSON object."
            )
            continue
        return dispatcher.dispatch(call)

    return ToolResult(
        call=None,
        success=False,
        error=f"Failed to produce a valid tool call after {max_attempts} attempts: {last_error}",
    )
