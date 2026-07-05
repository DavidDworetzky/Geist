"""
OnlineAgent implementation for routing to OpenAI-compatible HTTP endpoints.
"""
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import httpx

from agents.agent_context import AgentContext
from agents.base_agent import BaseAgent
from agents.exceptions import CompletionRequestError
from agents.models.generic_completion import GenericCompletion

logger = logging.getLogger(__name__)


class OnlineAgent(BaseAgent):
    """
    Online agent implementation that routes requests to OpenAI-compatible HTTP endpoints.
    Supports multiple providers including OpenAI, Anthropic, Grok, and Groq.
    """

    def __init__(
        self,
        agent_context: AgentContext,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        backup_providers: Optional[List[Dict[str, str]]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        as_subprocess: bool = False,
        **kwargs
    ):
        """
        Initialize OnlineAgent with HTTP client configuration.

        Args:
            agent_context: Agent context object
            base_url: Base URL for the API endpoint
            model: Model name to use
            api_key: API key for authentication
            backup_providers: List of backup provider configurations
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            as_subprocess: Whether to run as subprocess
        """
        super().__init__(agent_context, as_subprocess)

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key or self._get_api_key_from_env()
        self.backup_providers = backup_providers or []
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize HTTP client
        self.client = httpx.Client(timeout=timeout)

        # Set up headers
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get provider-specific API key from environment variables based on the endpoint."""
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
        self,
        payload: Dict[str, Any],
        use_backup: bool = False,
        backup_index: int = 0
    ) -> Dict[str, Any]:
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

                response = self.client.post(
                    current_url,
                    json=payload,
                    headers=current_headers
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.warning(
                        f"Request failed with status {response.status_code}: {response.text}"
                    )

                    # Try backup on certain failures
                    if (response.status_code >= 500 and
                        not use_backup and
                        self.backup_providers):
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
                        f"Request failed after {self.max_retries} attempts: {e}") from e

        raise CompletionRequestError("Unexpected error in request handling")

    def _build_payload(self, prompt: str, max_tokens: Optional[int], n: Optional[int],
                       temperature: Optional[float], top_p: Optional[float],
                       frequency_penalty: Optional[float], presence_penalty: Optional[float],
                       stop: Optional[str], system_prompt: Optional[str],
                       chat_id: Optional[int]) -> Dict[str, Any]:
        """Build a chat-completions payload from resolved params and hydrated history."""
        params = self._resolve_generation_params(
            max_tokens=max_tokens, n=n, temperature=temperature, top_p=top_p,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

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
        max_tokens: int = None,
        n: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: Optional[str] = None,
        echo: bool = False,
        best_of: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        response_format: str = "text",
        system_prompt: str = None,
        chat_id: Optional[int] = None
    ) -> GenericCompletion:
        """Complete text using the online API."""
        payload = self._build_payload(
            prompt, max_tokens, n, temperature, top_p,
            frequency_penalty, presence_penalty, stop, system_prompt, chat_id)

        # Make request
        response_data = self._make_request(payload)

        # Convert to GenericCompletion
        completion = GenericCompletion.from_dict(response_data)

        # Add to chat history with None check
        ai_message = None
        if completion.choices and len(completion.choices) > 0:
            ai_message = completion.choices[0].message.content
        chat_history = self._agent_context._add_to_chat_history(
            user_message=prompt,
            ai_message=ai_message,
            chat_id=chat_id
        )

        completion.chat_id = chat_history.chat_session_id
        return completion

    def stream_complete_text(
        self,
        prompt: str,
        max_tokens: int = None,
        n: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: Optional[str] = None,
        echo: bool = False,
        best_of: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        response_format: str = "text",
        system_prompt: str = None,
        chat_id: Optional[int] = None
    ) -> Iterator[str]:
        """Stream text completion from the online API."""
        payload = self._build_payload(
            prompt, max_tokens, n, temperature, top_p,
            frequency_penalty, presence_penalty, stop, system_prompt, chat_id)
        payload["stream"] = True

        # Ensure endpoint includes chat/completions
        current_url = self.base_url
        if not current_url.endswith("/chat/completions"):
            current_url = f"{current_url}/chat/completions"

        # Stream the response
        full_content = ""
        try:
            with self.client.stream(
                "POST",
                current_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
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
                    user_message=prompt,
                    ai_message=full_content,
                    chat_id=chat_id
                )
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise

    def complete_audio(
        self,
        audio_file,
        max_tokens: int = 16,
        n: int = 1,
        stop: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        echo: bool = False,
        best_of: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        response_format: str = "text",
        system_prompt: Optional[str] = None,
        chat_id: Optional[int] = None
    ):
        """Audio completion for online providers (OpenAI Whisper API)."""
        if "openai.com" not in self.base_url:
            raise NotImplementedError("Audio completion only supported for OpenAI endpoints")

        # Use OpenAI's transcription endpoint
        transcription_url = self.base_url.replace("/chat/completions", "/audio/transcriptions")

        # Transcribe audio
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {
                'model': 'whisper-1',
                'response_format': 'text',
                'temperature': temperature
            }

            response = self.client.post(
                transcription_url,
                files=files,
                data=data,
                headers={"Authorization": self.headers["Authorization"]}
            )

        if response.status_code != 200:
            raise CompletionRequestError(
                f"Transcription failed: {response.status_code} - {response.text}")

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
            chat_id=chat_id
        )

    def connect_realtime_audio(self):
        """Real-time audio connection (not implemented)."""
        raise NotImplementedError("Real-time audio not yet implemented for OnlineAgent")

    def _cleanup_resources(self):
        """Close the HTTP client."""
        if getattr(self, 'client', None):
            try:
                self.client.close()
            except Exception:
                pass

    def __del__(self):
        """Clean up HTTP client on destruction."""
        self._cleanup_resources()
