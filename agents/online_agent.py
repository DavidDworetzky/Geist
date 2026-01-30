"""
OnlineAgent implementation for routing to OpenAI-compatible HTTP endpoints.
"""
import os
import json
import logging
import subprocess
import signal
from typing import Optional, Dict, Any, List, Iterator
import httpx
from agents.base_agent import BaseAgent
from agents.agent_context import AgentContext
from agents.models.generic_completion import GenericCompletion
from agents.models.agent_completion import AgentCompletion
from app.models.database.chat_session import get_chat_history
from agents.response_utils import (
    AgentResponseMixin,
    WORLD_TICK_PROMPT,
    TASK_TICK_PROMPT,
    EXECUTION_TICK_PROMPT,
    build_chat_messages,
)

logger = logging.getLogger(__name__)


class OnlineAgent(AgentResponseMixin, BaseAgent):
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
        
        self.logger = logging.getLogger(__name__)
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
        """Get API key from environment variables based on the endpoint."""
        if "openai.com" in self.base_url:
            return os.getenv("OPENAI_API_KEY")
        elif "anthropic" in self.base_url:
            return os.getenv("ANTHROPIC_API_KEY")
        elif "groq.com" in self.base_url:
            return os.getenv("GROQ_API_KEY")
        elif "x.ai" in self.base_url:  # Grok
            return os.getenv("GROK_API_KEY")
        else:
            # Fallback to generic API key
            return os.getenv("API_KEY")
    
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
                        raise Exception(
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
                    raise Exception(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise Exception("Unexpected error in request handling")
    
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
        # Normalize parameters using shared utility
        params = self._normalize_params(max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty)

        # Build messages with chat history hydration
        history = None
        if chat_id is not None:
            try:
                history = get_chat_history(chat_id)
            except Exception as e:
                self.logger.warning(f"Failed to hydrate chat history for chat_id={chat_id}: {e}")

        messages = build_chat_messages(prompt, system_prompt, history)

        # Build payload
        payload = {
            "messages": messages,
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

        # Make request
        response_data = self._make_request(payload)

        # Convert to GenericCompletion
        completion = GenericCompletion.from_dict(response_data)

        # Add to chat history using the common extraction method
        ai_message = completion.get_assistant_content()
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
        # Normalize parameters using shared utility
        params = self._normalize_params(max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty)

        # Build messages with chat history hydration
        history = None
        if chat_id is not None:
            try:
                history = get_chat_history(chat_id)
            except Exception as e:
                self.logger.warning(f"Failed to hydrate chat history for chat_id={chat_id}: {e}")

        messages = build_chat_messages(prompt, system_prompt, history)

        # Build payload with streaming enabled
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": params.max_tokens,
            "n": params.n,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "stream": True
        }

        if stop is not None:
            payload["stop"] = stop
        
        # Ensure endpoint includes chat/completions
        current_url = self.base_url
        if not current_url.endswith("/chat/completions"):
            current_url = f"{current_url}/chat/completions"
        
        # Stream the response
        full_content = ""
        try:
            with httpx.stream(
                "POST",
                current_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            ) as response:
                if response.status_code != 200:
                    raise Exception(
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
            raise Exception(f"Transcription failed: {response.status_code} - {response.text}")
        
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
    
    def initialize(self, task_prompt: str = None):
        """Initialize the agent with optional task."""
        if task_prompt:
            self._agent_context.task_context.append(task_prompt)
        
        if self.as_subprocess:
            self.logger.info("Initializing agent with subprocess.")
            process = subprocess.Popen(['python3', '-u', 'tick.py'], stdout=subprocess.PIPE)
            self._agent_context.subprocess_id = process.pid
        else:
            self.logger.info("Initializing agent without subprocess.")
            self._agent_context.subprocess_id = None
    
    def phase_out(self):
        """Phase out the agent and clean up resources."""
        self._agent_context._save()
        if hasattr(self, 'client'):
            self.client.close()
        self.terminate_subprocess()
    
    def phase_in(self):
        """Phase in the agent and restore state."""
        self.initialize()
    
    def terminate_subprocess(self):
        """Terminate any running subprocess."""
        subprocess_id = self._agent_context.subprocess_id
        if subprocess_id:
            os.kill(subprocess_id, signal.SIGTERM)
            self._agent_context.subprocess_id = None
    
    def tick(self):
        """Execute one agent tick."""
        self.logger.info("OnlineAgent Tick.")
        if self._agent_context.settings.include_world_processing:
            self.tick_world()
        self.tick_tasks()
        self.tick_execution()
    
    def tick_world(self):
        """Advance world state reasoning."""
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=False)
        input_prompt = WORLD_TICK_PROMPT + context_string
        result = self.complete_text(prompt=input_prompt)
        split_result = self._transform_completions(result)
        self._agent_context.world_context = split_result
        return split_result
    
    def tick_tasks(self):
        """Advance task context reasoning."""
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=TASK_TICK_PROMPT + context_string)
        result = self._transform_completions(result)
        split_result = result[0].split("\\n") if result else []
        self._agent_context.task_context = split_result
        return split_result
    
    def tick_execution(self):
        """Advance execution context reasoning."""
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=EXECUTION_TICK_PROMPT + context_string)
        result = self._transform_completions(result)
        split_result = result[0].split("\\n") if result else []
        self._agent_context.execution_context = split_result
        return split_result
    
