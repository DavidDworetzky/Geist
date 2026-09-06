"""Shared native-tool protocol for tokenizer chat-template runtimes."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from agents.models.tool_calling import ChatMessage, ModelTurn, ToolCall, ToolDefinition


_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_PATTERN = re.compile(r"<function=([A-Za-z0-9_-]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_PATTERN = re.compile(r"<parameter=([^<>\s]+)>(.*?)</parameter>", re.DOTALL)


@dataclass(frozen=True)
class ChatTemplateToolPayload:
    """Provider-safe messages and schemas for a tokenizer chat template."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    provider_to_internal: dict[str, str]


def provider_tool_name(name: str) -> str:
    """Return a deterministic name accepted by common tool chat templates."""

    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        return name
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "tool"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:55]}_{digest}"


def build_tool_payload(
    messages: list[ChatMessage],
    tools: list[ToolDefinition],
) -> ChatTemplateToolPayload:
    """Serialize a Geist conversation for OpenAI-shaped tokenizer templates."""

    internal_to_provider: dict[str, str] = {}
    provider_to_internal: dict[str, str] = {}
    names = [tool.name for tool in tools]
    for message in messages:
        names.extend(call.name for call in message.tool_calls)
        if message.name:
            names.append(message.name)

    for name in dict.fromkeys(names):
        provider_name = provider_tool_name(name)
        if provider_name in provider_to_internal and provider_to_internal[provider_name] != name:
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
            provider_name = f"{provider_name[:51]}_{digest}"
        internal_to_provider[name] = provider_name
        provider_to_internal[provider_name] = name

    serialized_messages: list[dict[str, Any]] = []
    for message in messages:
        value: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            value["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": internal_to_provider[call.name],
                        # Tokenizer templates render JSON themselves. Supplying
                        # a dictionary avoids teaching the model a double-encoded
                        # arguments string from the OpenAI wire format.
                        "arguments": call.arguments,
                    },
                    "id": call.id,
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            value["tool_call_id"] = message.tool_call_id
        if message.name and message.role != "tool":
            value["name"] = internal_to_provider.get(message.name, message.name)
        serialized_messages.append(value)

    return ChatTemplateToolPayload(
        messages=serialized_messages,
        tools=[tool.to_openai(internal_to_provider[tool.name]) for tool in tools],
        provider_to_internal=provider_to_internal,
    )


def tokenizer_supports_tools(
    tokenizer: Any,
    *,
    template_options: dict[str, Any] | None = None,
) -> bool:
    """Probe whether a tokenizer template consumes and renders tool schemas."""

    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return False
    sentinel = "geist_tool_probe_7f3d"
    tool = {
        "type": "function",
        "function": {
            "name": sentinel,
            "description": "Capability probe",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    options = dict(template_options or {})
    options.update(
        {
            "tools": [tool],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    )
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "tool capability probe"}],
            **options,
        )
    except Exception:
        # Third-party Jinja templates raise backend-specific exception types;
        # capability detection must remain fail-closed for all of them.
        return False
    return sentinel in str(rendered)


def parse_tool_response(
    response: str,
    *,
    provider_to_internal: dict[str, str],
    tools: list[dict[str, Any]] | None = None,
) -> ModelTurn:
    """Parse Qwen-style tool markup into a fail-closed structured model turn."""

    raw_response = response.strip()
    matches = list(_TOOL_CALL_PATTERN.finditer(raw_response))
    has_marker = _TOOL_CALL_OPEN in raw_response or _TOOL_CALL_CLOSE in raw_response
    if has_marker and not matches:
        raise ValueError("Model returned incomplete tool-call markup")

    payloads: list[str]
    if matches:
        without_calls = _TOOL_CALL_PATTERN.sub("", raw_response).strip()
        if _TOOL_CALL_OPEN in without_calls or _TOOL_CALL_CLOSE in without_calls:
            raise ValueError("Model returned malformed tool-call markup")
        payloads = [match.group(1) for match in matches]
        text = without_calls
    else:
        payloads = []
        text = raw_response
        if provider_to_internal and raw_response.startswith("{"):
            try:
                candidate = json.loads(raw_response)
            except json.JSONDecodeError:
                candidate = None
            if (
                isinstance(candidate, dict)
                and isinstance(candidate.get("name"), str)
                and candidate["name"] in provider_to_internal
                and ("arguments" in candidate or "parameters" in candidate)
            ):
                payloads = [raw_response]
                text = ""

    if "<function=" in text:
        raise ValueError("Model returned unwrapped function-call markup")

    calls: list[ToolCall] = []
    for payload in payloads:
        if payload.lstrip().startswith("<function="):
            value = _parse_function_call(payload, tools or [])
        else:
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError("Model returned invalid tool-call JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Model returned a non-object tool call")
        provider_name = value.get("name")
        if not isinstance(provider_name, str) or provider_name not in provider_to_internal:
            raise ValueError(f"Model requested unknown tool: {provider_name!r}")
        # This tokenizer path follows the OpenAI-shaped contract, so arguments
        # intentionally wins if a model emits both keys. Parameters remains a
        # compatibility alias for Qwen responses that mirror the tool schema.
        arguments = value.get("arguments", value.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ValueError("Model returned invalid tool arguments") from error
        if not isinstance(arguments, dict):
            raise ValueError("Model returned non-object tool arguments")
        call_id = value.get("id")
        calls.append(
            ToolCall(
                id=(
                    call_id
                    if isinstance(call_id, str) and call_id
                    else f"toolcall_{uuid.uuid4().hex}"
                ),
                name=provider_to_internal[provider_name],
                arguments=arguments,
            )
        )

    return ModelTurn(
        text=text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
    )


def _schema_types(schema: dict[str, Any], root: dict[str, Any]) -> set[str]:
    pending = [schema]
    seen: set[int] = set()
    types: set[str] = set()
    while pending:
        current = pending.pop()
        if not isinstance(current, dict) or id(current) in seen:
            continue
        seen.add(id(current))
        declared = current.get("type", [])
        if isinstance(declared, str):
            types.add(declared)
        elif isinstance(declared, list):
            types.update(value for value in declared if isinstance(value, str))
        for key in ("anyOf", "oneOf", "allOf"):
            variants = current.get(key, [])
            if isinstance(variants, list):
                pending.extend(variants)
        reference = current.get("$ref", "")
        if isinstance(reference, str) and reference.startswith("#/"):
            resolved = root
            for key in reference[2:].split("/"):
                if not isinstance(resolved, dict):
                    break
                resolved = resolved.get(key.replace("~1", "/").replace("~0", "~"), {})
            if isinstance(resolved, dict):
                pending.append(resolved)
    return types


def _parse_function_call(payload: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse Qwen's XML-like function syntax; its tags are not standard XML."""
    function = _FUNCTION_PATTERN.fullmatch(payload.strip())
    if function is None:
        raise ValueError("Model returned malformed function-call markup")
    name, body = function.groups()
    schema = next(
        (
            tool["function"].get("parameters", {})
            for tool in tools
            if tool.get("function", {}).get("name") == name
        ),
        None,
    )
    if not isinstance(schema, dict):
        raise ValueError(f"Model requested unknown tool or missing function schema: {name!r}")
    arguments: dict[str, Any] = {}
    cursor = 0
    while cursor < len(body):
        if body[cursor].isspace():
            cursor += 1
            continue
        if body.startswith("</function>", cursor):
            raise ValueError("Model returned multiple functions in one tool-call wrapper")
        parameter = _PARAMETER_PATTERN.match(body, cursor)
        if parameter is None:
            raise ValueError("Model returned malformed function parameter markup")
        key, text = parameter.groups()
        if key in arguments:
            raise ValueError("Model returned a duplicate function parameter")
        # The template adds one framing newline on each side of the value.
        # Preserve user strings (including whitespace and JSON-looking text).
        text = text.removeprefix("\r\n") if text.startswith("\r\n") else text.removeprefix("\n")
        text = text.removesuffix("\r\n") if text.endswith("\r\n") else text.removesuffix("\n")
        additional = schema.get("additionalProperties", {})
        parameter_schema = schema.get("properties", {}).get(
            key, additional if isinstance(additional, dict) else {}
        )
        types = _schema_types(parameter_schema, schema)
        # Qwen renders None and literal "null" identically for nullable strings.
        # Preserve explicit null semantics, but never strip a non-null string.
        if "null" in types and text == "null":
            arguments[key] = None
        elif "string" in types:
            arguments[key] = text
        else:
            try:
                arguments[key] = json.loads(text)
            except json.JSONDecodeError as error:
                if types:
                    raise ValueError("Model returned invalid JSON for a typed parameter") from error
                arguments[key] = text
        cursor = parameter.end()
    return {"name": name, "arguments": arguments}


class ToolResponseStream:
    """Expose text incrementally; keep tool markup private until validation."""

    def __init__(
        self,
        provider_to_internal: dict[str, str],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.provider_to_internal = provider_to_internal
        self.tools = tools
        self._raw: list[str] = []
        self._emitted: list[str] = []
        self._tool: list[str] | None = None
        self._pending = ""
        self._whitespace: list[str] = []
        self._at_start = True
        self._bare_json = False

    def feed(self, segment: str) -> str:
        self._raw.append(segment)
        if self._bare_json:
            return ""
        remaining = self._pending + segment
        self._pending = ""
        if self._at_start:
            remaining = remaining.lstrip()
            if not remaining:
                return ""
            self._at_start = False
            # The complete-response parser accepts untagged JSON tool calls.
            # Their text/tool interpretation is only authoritative at EOF.
            if self.provider_to_internal and remaining.startswith("{"):
                self._bare_json = True
                return ""

        visible = []
        while remaining:
            if self._tool is not None:
                closing = remaining.find(_TOOL_CALL_CLOSE)
                if closing < 0:
                    body, self._pending = self._split_marker_prefix(remaining, (_TOOL_CALL_CLOSE,))
                    self._tool.append(body)
                    break
                end = closing + len(_TOOL_CALL_CLOSE)
                self._tool.append(remaining[:end])
                # Reject bad markup before releasing any following text. Calls
                # are only returned by finish(), after validating the whole turn.
                parse_tool_response(
                    "".join(self._tool),
                    provider_to_internal=self.provider_to_internal,
                    tools=self.tools,
                )
                self._tool = None
                remaining = remaining[end:]
                continue

            opening = remaining.find(_TOOL_CALL_OPEN)
            closing = remaining.find(_TOOL_CALL_CLOSE)
            function = remaining.find("<function=")
            if function >= 0 and (opening < 0 or function < opening):
                raise ValueError("Model returned unwrapped function-call markup")
            if closing >= 0 and (opening < 0 or closing < opening):
                raise ValueError("Model returned malformed tool-call markup")
            if opening >= 0:
                visible.append(remaining[:opening])
                self._tool = [_TOOL_CALL_OPEN]
                remaining = remaining[opening + len(_TOOL_CALL_OPEN) :]
                continue
            text, self._pending = self._split_marker_prefix(
                remaining, (_TOOL_CALL_OPEN, _TOOL_CALL_CLOSE, "<function=")
            )
            visible.append(text)
            break

        text = "".join(visible)
        # Leading whitespace must never enter the deferred inter-chunk buffer.
        if not self._emitted:
            text = text.lstrip()
        stripped = text.rstrip()
        if not stripped:
            if text:
                self._whitespace.append(text)
            return ""
        delta = "".join(self._whitespace) + stripped
        self._whitespace = [text[len(stripped) :]]
        self._emitted.append(delta)
        return delta

    def finish(self) -> tuple[str, ModelTurn]:
        turn = parse_tool_response(
            "".join(self._raw),
            provider_to_internal=self.provider_to_internal,
            tools=self.tools,
        )
        emitted = "".join(self._emitted)
        if not turn.text.startswith(emitted):
            raise ValueError("Streamed text does not match the parsed model turn")
        return turn.text[len(emitted) :], turn

    @staticmethod
    def _split_marker_prefix(text: str, markers: tuple[str, ...]) -> tuple[str, str]:
        retained = max(
            (
                length
                for marker in markers
                for length in range(1, min(len(text), len(marker) - 1) + 1)
                if text.endswith(marker[:length])
            ),
            default=0,
        )
        return (text[:-retained], text[-retained:]) if retained else (text, "")
