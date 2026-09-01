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
                and "name" in candidate
                and ("arguments" in candidate or "parameters" in candidate)
            ):
                payloads = [raw_response]
                text = ""

    calls: list[ToolCall] = []
    for payload in payloads:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("Model returned invalid tool-call JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Model returned a non-object tool call")
        provider_name = value.get("name")
        if not isinstance(provider_name, str) or provider_name not in provider_to_internal:
            raise ValueError(f"Model requested unknown tool: {provider_name!r}")
        # Some tokenizer-native models mirror the tool schema's ``parameters``
        # key in their call instead of emitting OpenAI's ``arguments`` key.
        # Accept that known variant without weakening tool-name or schema
        # validation downstream.
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
