"""
OnlineAgent implementation for routing to OpenAI-compatible HTTP endpoints.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Iterator
from typing import Any, cast

import httpx

from agents.agent_context import AgentContext
from agents.base_agent import BaseAgent
from agents.exceptions import CompletionRequestError
from agents.model_catalog import PROVIDERS
from agents.models.generic_completion import GenericCompletion
from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolDefinition,
)
from agents.models.tool_calling import (
    ToolCall as ChatToolCall,
)
from agents.tool_calling import (
    ToolCall as AdapterToolCall,
)
from agents.tool_calling import (
    ToolCallError,
    ToolCompletion,
    ToolDispatcher,
    ToolResult,
    run_prompt_tool_call,
)


logger = logging.getLogger(__name__)


class NativeProviderError(CompletionRequestError):
    """A provider failure classified for safe pre-emission retry decisions."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class OnlineAgent(BaseAgent):
    """
    Online agent implementation that routes requests to OpenAI-compatible HTTP endpoints.
    Supports multiple providers including OpenAI, Anthropic, Grok, and Groq.
    """

    supports_native_tool_calling = False

    def __init__(
        self,
        agent_context: AgentContext,
        base_url: str,
        model: str,
        api_key: str | None = None,
        backup_providers: list[dict[str, Any]] | None = None,
        generation_config: dict[str, Any] | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        supports_native_tool_calling: bool | None = None,
        as_subprocess: bool = False,
        **kwargs,
    ):
        """
        Initialize OnlineAgent with HTTP client configuration.

        Args:
            agent_context: Agent context object
            base_url: Base URL for the API endpoint
            model: Model name to use
            api_key: API key for authentication
            backup_providers: List of backup provider configurations
            generation_config: Optional default generation parameters
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            supports_native_tool_calling: Explicit capability override for
                custom OpenAI-compatible endpoints.
            as_subprocess: Whether to run as subprocess
        """
        super().__init__(agent_context, as_subprocess)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or self._get_api_key_from_env()
        self.backup_providers = backup_providers or []
        self.generation_config = generation_config or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.supports_native_tool_calling = (
            supports_native_tool_calling
            if supports_native_tool_calling is not None
            else self._known_native_tool_provider()
        )

        # Initialize HTTP client
        self.client = httpx.Client(timeout=timeout)

        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _known_native_tool_provider(self) -> bool:
        normalized_url = self.base_url.lower()
        return any(
            provider_host in normalized_url
            for provider_host in ("openai.com", "anthropic.com", "groq.com", "api.x.ai")
        )

    def _get_api_key_from_env(self) -> str | None:
        """Get provider-specific API key from environment variables based on the endpoint."""
        for provider in PROVIDERS.values():
            if provider.base_url and provider.base_url.rstrip("/") in self.base_url:
                return os.getenv(provider.api_key_env)
        if "openai.com" in self.base_url:
            return os.getenv("OPENAI_API_KEY")
        elif "anthropic" in self.base_url:
            return os.getenv("ANTHROPIC_API_KEY")
        elif "groq.com" in self.base_url:
            return os.getenv("GROQ_API_KEY")
        elif "x.ai" in self.base_url:  # Grok
            return os.getenv("GROK_API_KEY")
        return None

    def _make_request(
        self, payload: dict[str, Any], use_backup: bool = False, backup_index: int = 0
    ) -> dict[str, Any]:
        """
        Make HTTP request to the API endpoint with retry logic.

        Args:
            payload: Request payload
            use_backup: Whether to use backup provider
            backup_index: Index of backup provider to use

        Returns:
            Response dictionary
        """
        current_url = self.base_url
        current_headers = self.headers.copy()
        current_model = self.model

        # Use backup provider if specified
        if use_backup and backup_index < len(self.backup_providers):
            backup = self.backup_providers[backup_index]
            current_url = backup.get("base_url", self.base_url)
            current_model = backup.get("model", self.model)
            if "api_key" in backup:
                current_headers["Authorization"] = f"Bearer {backup['api_key']}"

            # Update payload with backup model
            payload = payload.copy()
            payload["model"] = current_model

        # Ensure endpoint includes chat/completions
        if not current_url.endswith("/chat/completions"):
            current_url = f"{current_url}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Making request to {current_url} (attempt {attempt + 1})")

                response = self.client.post(current_url, json=payload, headers=current_headers)

                if response.status_code == 200:
                    return cast(dict[str, Any], response.json())
                else:
                    self.logger.warning(
                        f"Request failed with status {response.status_code}: {response.text}"
                    )

                    # Try backup on certain failures
                    if response.status_code >= 500 and not use_backup and self.backup_providers:
                        self.logger.info("Trying backup provider due to server error")
                        return self._make_request(payload, use_backup=True, backup_index=0)

                    if attempt == self.max_retries - 1:
                        raise CompletionRequestError(
                            f"API request failed after {self.max_retries} attempts. "
                            f"Status: {response.status_code}, Response: {response.text}"
                        )

            except httpx.RequestError as e:
                self.logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    # Try backup provider on final failure
                    if not use_backup and self.backup_providers:
                        self.logger.info("Trying backup provider due to request error")
                        return self._make_request(payload, use_backup=True, backup_index=0)
                    raise CompletionRequestError(
                        f"Request failed after {self.max_retries} attempts: {e}"
                    ) from e

        raise CompletionRequestError("Unexpected error in request handling")

    def _build_payload(
        self,
        prompt: str,
        max_tokens: int | None,
        n: int | None,
        temperature: float | None,
        top_p: float | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        stop: str | list[str] | None,
        system_prompt: str | None,
        chat_id: int | None,
    ) -> dict[str, Any]:
        """Build a chat-completions payload from resolved params and hydrated history."""
        params = self._resolve_generation_params(
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        payload = {
            "messages": self._build_messages(prompt, system_prompt=system_prompt, chat_id=chat_id),
            "model": self.model,
            "max_tokens": params.max_tokens,
            "n": params.n,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
        }
        if stop is not None:
            payload["stop"] = stop
        return payload

    def complete_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        n: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        echo: bool = False,
        best_of: int | None = None,
        prompt_tokens: list[int] | None = None,
        response_format: str = "text",
        system_prompt: str | None = None,
        chat_id: int | None = None,
    ) -> GenericCompletion:
        """Complete text using the online API."""
        payload = self._build_payload(
            prompt,
            max_tokens,
            n,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            stop,
            system_prompt,
            chat_id,
        )
        # Make request
        response_data = self._make_request(payload)

        # Convert to GenericCompletion
        completion = GenericCompletion.from_dict(response_data)

        # Add to chat history using the None-safe extraction method
        ai_message = completion.get_assistant_content()
        chat_history = self._agent_context._add_to_chat_history(
            user_message=prompt, ai_message=ai_message, chat_id=chat_id
        )

        completion.chat_id = chat_history.chat_session_id
        return completion

    def stream_complete_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        n: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        echo: bool = False,
        best_of: int | None = None,
        prompt_tokens: list[int] | None = None,
        response_format: str = "text",
        system_prompt: str | None = None,
        chat_id: int | None = None,
    ) -> Iterator[str]:
        """Stream text completion from the online API."""
        payload = self._build_payload(
            prompt,
            max_tokens,
            n,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            stop,
            system_prompt,
            chat_id,
        )
        payload["stream"] = True
        # Ensure endpoint includes chat/completions
        current_url = self.base_url
        if not current_url.endswith("/chat/completions"):
            current_url = f"{current_url}/chat/completions"

        # Stream the response
        full_content = ""
        try:
            with self.client.stream(
                "POST", current_url, json=payload, headers=self.headers, timeout=self.timeout
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise CompletionRequestError(
                        f"Streaming request failed with status {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                    yield content
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse SSE chunk: {data}")
                            continue

            # Add to chat history after streaming completes
            if chat_id is not None:
                self._agent_context._add_to_chat_history(
                    user_message=prompt, ai_message=full_content, chat_id=chat_id
                )
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise

    def _complete_raw(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Single completion returning plain text, without chat-history side effects.
        Used for tool-call loops so intermediate attempts don't pollute history.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens or self._agent_context.settings.max_tokens or 16,
            "temperature": temperature
            if temperature is not None
            else (self._agent_context.settings.temperature or 1.0),
        }
        response = self._make_request(payload)
        choices = response.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    def complete_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_tool_iterations: int = 4,
        chat_id: int | None = None,
    ) -> ToolCompletion:
        """
        Tool-augmented completion using the provider's native function calling.

        Adapter actions are exposed via the OpenAI-compatible `tools` parameter;
        returned tool_calls are validated against the reflected schemas and
        dispatched, with results fed back as `tool` messages until the model
        produces a final text answer or the iteration cap is reached. If the
        endpoint rejects the tools payload, falls back to schema-grounded
        prompt-based tool calling.
        """
        schemas = self._agent_context.get_tool_schemas()
        dispatcher = self._agent_context.get_tool_dispatcher()

        if not schemas:
            plain_content = self._complete_raw(prompt, system_prompt, max_tokens, temperature)
            return ToolCompletion(content=plain_content, tool_results=[], iterations=1)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        tools_payload = [schema.to_openai_tool() for schema in schemas]

        tool_results: list[ToolResult] = []
        content: str | None = None
        iteration = 0

        for iteration in range(1, max_tool_iterations + 1):
            payload = {
                "messages": messages,
                "model": self.model,
                "max_tokens": max_tokens or self._agent_context.settings.max_tokens or 16,
                "temperature": temperature
                if temperature is not None
                else (self._agent_context.settings.temperature or 1.0),
                "tools": tools_payload,
            }
            try:
                response = self._make_request(payload)
            except Exception as e:
                if iteration == 1:
                    self.logger.warning(
                        f"Native tool calling failed ({e}); falling back to prompt-based tool calling"
                    )
                    return self._prompt_based_tool_completion(prompt, chat_id=chat_id)
                raise

            choices = response.get("choices") or []
            if not choices:
                break
            message = choices[0].get("message") or {}
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                message_content = message.get("content")
                content = message_content if isinstance(message_content, str) else None
                break

            messages.append(message)
            for tool_call in tool_calls:
                result = self._dispatch_native_tool_call(tool_call, dispatcher)
                tool_results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": result.to_content(),
                    }
                )

        if chat_id is not None:
            try:
                self._agent_context._add_to_chat_history(
                    user_message=prompt, ai_message=content, chat_id=chat_id
                )
            except Exception as e:
                self.logger.warning(f"Failed to record tool completion in chat history: {e}")

        return ToolCompletion(
            content=content,
            tool_results=tool_results,
            iterations=iteration,
            used_native_tools=True,
        )

    def _dispatch_native_tool_call(
        self, tool_call: dict[str, Any], dispatcher: ToolDispatcher
    ) -> ToolResult:
        """Convert a provider tool_call entry into a validated, dispatched ToolCall."""
        function = tool_call.get("function") or {}
        name = function.get("name") or ""
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
            if not isinstance(arguments, dict):
                raise ToolCallError("tool call arguments must decode to a JSON object")
            call = AdapterToolCall.from_qualified_name(name, arguments, raw=str(raw_arguments))
        except (json.JSONDecodeError, ToolCallError) as e:
            self.logger.warning(f"Malformed native tool call '{name}': {e}")
            return ToolResult(call=None, success=False, error=f"Malformed tool call '{name}': {e}")
        return dispatcher.dispatch(call)

    def _prompt_based_tool_completion(
        self, prompt: str, chat_id: int | None = None
    ) -> ToolCompletion:
        """Schema-grounded prompt fallback when native tool calling is unavailable."""
        result = run_prompt_tool_call(
            complete_fn=lambda task, system: self._complete_raw(task, system),
            schemas=self._agent_context.get_tool_schemas(),
            dispatcher=self._agent_context.get_tool_dispatcher(),
            task_prompt=prompt,
        )
        return ToolCompletion(
            content=result.to_content(),
            tool_results=[result],
            iterations=1,
            used_native_tools=False,
        )

    def _is_anthropic(self) -> bool:
        return "anthropic" in self.base_url.lower()

    def _openai_chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _anthropic_messages_url(self) -> str:
        if self.base_url.endswith("/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    @staticmethod
    def _tool_arguments(value: str) -> dict[str, Any]:
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {"__raw_arguments__": value}
        return parsed if isinstance(parsed, dict) else {"__raw_arguments__": value}

    @staticmethod
    def _provider_tool_name(name: str) -> str:
        """Encode an internal tool name for provider function-name constraints."""
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            return name
        readable = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "tool"
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        return f"{readable[:55]}_{digest}"

    @classmethod
    def _provider_tool_name_maps(
        cls,
        tools: list[ToolDefinition],
        messages: list[ChatMessage] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        internal_to_provider: dict[str, str] = {}
        provider_to_internal: dict[str, str] = {}

        names = [tool.name for tool in tools]
        for message in messages or []:
            names.extend(call.name for call in message.tool_calls)
            if message.name:
                names.append(message.name)

        for name in dict.fromkeys(names):
            provider_name = cls._provider_tool_name(name)
            if (
                provider_name in provider_to_internal
                and provider_to_internal[provider_name] != name
            ):
                digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
                provider_name = f"{provider_name[:51]}_{digest}"
            internal_to_provider[name] = provider_name
            provider_to_internal[provider_name] = name
        return internal_to_provider, provider_to_internal

    def _stream_openai_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        internal_to_provider, provider_to_internal = self._provider_tool_name_maps(tools, messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai(internal_to_provider) for message in messages],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stream": True,
        }
        if config.stop:
            payload["stop"] = config.stop
        if tools:
            payload["tools"] = [tool.to_openai(internal_to_provider[tool.name]) for tool in tools]
            payload["tool_choice"] = "auto"

        text_parts: list[str] = []
        call_parts: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        saw_done = False
        with self.client.stream(
            "POST",
            self._openai_chat_url(),
            json=payload,
            headers=self.headers,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise NativeProviderError(
                    f"Streaming request failed with status {response.status_code}: {response.text}",
                    retryable=response.status_code in {408, 409, 429}
                    or response.status_code >= 500,
                )

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    self.logger.warning("Ignoring invalid provider SSE data")
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield ModelEvent.text_delta(content)

                for fallback_index, tool_delta in enumerate(delta.get("tool_calls") or []):
                    index = tool_delta.get("index", fallback_index)
                    current = call_parts.setdefault(
                        int(index),
                        {"id": "", "name": "", "arguments": ""},
                    )
                    current["id"] += tool_delta.get("id") or ""
                    function = tool_delta.get("function") or {}
                    current["name"] += function.get("name") or ""
                    current["arguments"] += function.get("arguments") or ""

        if not saw_done and finish_reason is None:
            raise NativeProviderError(
                "Provider stream ended before a terminal marker",
                retryable=True,
            )
        if not text_parts and not call_parts and finish_reason is None:
            raise NativeProviderError("Provider returned an empty model turn", retryable=True)

        calls = [
            ChatToolCall(
                id=value["id"] or f"toolcall_{uuid.uuid4().hex}",
                name=provider_to_internal.get(value["name"], value["name"]),
                arguments=self._tool_arguments(value["arguments"]),
            )
            for _index, value in sorted(call_parts.items())
        ]
        yield ModelEvent.turn_complete(
            ModelTurn(
                text="".join(text_parts),
                tool_calls=calls,
                finish_reason=finish_reason,
            )
        )

    @staticmethod
    def _anthropic_payload_messages(
        messages: list[ChatMessage],
        tool_name_map: dict[str, str] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        result: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            if message.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": (tool_name_map or {}).get(call.name, call.name),
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                result.append({"role": "assistant", "content": blocks or ""})
            elif message.role == "tool":
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content or "",
                            }
                        ],
                    }
                )
            else:
                result.append({"role": "user", "content": message.content or ""})
        return ("\n\n".join(system_parts) or None), result

    def _stream_anthropic_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        internal_to_provider, provider_to_internal = self._provider_tool_name_maps(tools, messages)
        system, provider_messages = self._anthropic_payload_messages(messages, internal_to_provider)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": provider_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if system:
            payload["system"] = system
        if config.stop:
            payload["stop_sequences"] = config.stop
        if tools:
            payload["tools"] = [
                tool.to_anthropic(internal_to_provider[tool.name]) for tool in tools
            ]
            payload["tool_choice"] = {"type": "auto"}

        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        response = self.client.post(
            self._anthropic_messages_url(),
            json=payload,
            headers=headers,
        )
        if response.status_code >= 400:
            raise NativeProviderError(
                f"Anthropic request failed with status {response.status_code}: {response.text}",
                retryable=response.status_code in {408, 409, 429} or response.status_code >= 500,
            )
        data = response.json()
        if not data.get("content") and not data.get("stop_reason"):
            raise NativeProviderError("Anthropic returned an empty model turn", retryable=True)
        text_parts: list[str] = []
        calls: list[ChatToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text" and block.get("text"):
                text_parts.append(block["text"])
                yield ModelEvent.text_delta(block["text"])
            elif block.get("type") == "tool_use":
                arguments = block.get("input")
                calls.append(
                    ChatToolCall(
                        id=block.get("id") or f"toolcall_{uuid.uuid4().hex}",
                        name=provider_to_internal.get(
                            block.get("name") or "", block.get("name") or ""
                        ),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
                )
        yield ModelEvent.turn_complete(
            ModelTurn(
                text="".join(text_parts),
                tool_calls=calls,
                finish_reason=data.get("stop_reason"),
            )
        )

    def stream_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        """Normalize provider-native tool calls into Geist's small Python contract."""
        last_error: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            emitted_event = False
            try:
                events = (
                    self._stream_anthropic_model_turn(messages, tools, config)
                    if self._is_anthropic()
                    else self._stream_openai_model_turn(messages, tools, config)
                )
                for event in events:
                    emitted_event = True
                    yield event
                return
            except (httpx.RequestError, NativeProviderError) as error:
                retryable = not isinstance(error, NativeProviderError) or error.retryable
                if emitted_event or not retryable:
                    raise
                last_error = error
                if attempt + 1 < max(1, self.max_retries):
                    self.logger.warning(
                        "Native model turn failed before emitting text; retrying (%s/%s): %s",
                        attempt + 1,
                        self.max_retries,
                        error,
                    )

        for backup in sorted(
            self.backup_providers,
            key=lambda provider: int(provider.get("priority", 1)),
        ):
            backup_url = str(backup.get("base_url") or "").strip()
            backup_model = str(backup.get("model") or self.model).strip()
            if not backup_url:
                continue
            self.logger.warning("Trying native tool turn backup provider at %s", backup_url)
            backup_agent = OnlineAgent(
                agent_context=self._agent_context,
                base_url=backup_url,
                model=backup_model,
                api_key=backup.get("api_key"),
                timeout=self.timeout,
                max_retries=self.max_retries,
                supports_native_tool_calling=backup.get("supports_native_tool_calling"),
            )
            if tools and not backup_agent.supports_native_tool_calling:
                self.logger.warning(
                    "Skipping backup provider without declared native tool capability: %s",
                    backup_url,
                )
                backup_agent.client.close()
                continue
            emitted_event = False
            try:
                for event in backup_agent.stream_model_turn(messages, tools, config):
                    emitted_event = True
                    yield event
                return
            except Exception as error:
                if emitted_event:
                    raise
                last_error = error
                self.logger.warning("Native backup provider failed: %s", error)
            finally:
                backup_agent.client.close()

        if last_error is not None:
            raise last_error
        raise CompletionRequestError("Native model turn failed without a provider response")

    def complete_audio(
        self,
        audio_file,
        max_tokens: int | None = 16,
        n: int | None = 1,
        temperature: float | None = 1.0,
        top_p: float | None = 1,
        frequency_penalty: float | None = 0,
        presence_penalty: float | None = 0,
        stop: str | list[str] | None = None,
        echo: bool = False,
        best_of: int | None = None,
        prompt_tokens: list[int] | None = None,
        response_format: str = "text",
        system_prompt: str | None = None,
        chat_id: int | None = None,
    ):
        """Audio completion for online providers (OpenAI Whisper API)."""
        if "openai.com" not in self.base_url:
            raise NotImplementedError("Audio completion only supported for OpenAI endpoints")

        # Use OpenAI's transcription endpoint
        transcription_url = self.base_url.replace("/chat/completions", "/audio/transcriptions")

        # Transcribe audio
        with open(audio_file, "rb") as f:
            files = {"file": f}
            data = {"model": "whisper-1", "response_format": "text", "temperature": temperature}

            response = self.client.post(
                transcription_url,
                files=files,
                data=data,
                headers={"Authorization": self.headers["Authorization"]},
            )

        if response.status_code != 200:
            raise CompletionRequestError(
                f"Transcription failed: {response.status_code} - {response.text}"
            )

        transcribed_text = response.text

        # Generate text response
        return self.complete_text(
            prompt=transcribed_text,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            echo=echo,
            best_of=best_of,
            prompt_tokens=prompt_tokens,
            response_format=response_format,
            system_prompt=system_prompt,
            chat_id=chat_id,
        )

    def connect_realtime_audio(self):
        """Real-time audio connection (not implemented)."""
        raise NotImplementedError("Real-time audio not yet implemented for OnlineAgent")

    def _cleanup_resources(self):
        """Close the HTTP client."""
        if getattr(self, "client", None):
            with contextlib.suppress(Exception):
                self.client.close()

    def __del__(self):
        """Clean up HTTP client on destruction."""
        self._cleanup_resources()
