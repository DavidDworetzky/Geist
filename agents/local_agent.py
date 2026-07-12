"""
LocalAgent implementation for running local inference models.
"""
import logging
from typing import Any

from agents.agent_context import AgentContext
from agents.architectures import get_runner
from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.architectures.registry import ensure_runners_registered
from agents.base_agent import BaseAgent
from agents.models.llama_completion import LlamaCompletion
from agents.tool_calling import ToolCompletion, run_prompt_tool_call


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an agent looking to complete tasks for individuals. You will be given context about the world, the task and functions you can call. Take the most direct and thorough way of satisfying these constraints."


class LocalAgent(BaseAgent):
    """
    Local agent implementation that uses pluggable runners for inference.
    """

    def __init__(
        self,
        agent_context: AgentContext,
        model_id: str,
        runner_type: str = "mlx_llama",
        device_config: dict[str, Any] | None = None,
        as_subprocess: bool = False
    ):
        """
        Initialize LocalAgent with specified runner.

        Args:
            agent_context: Agent context object
            model_id: Model identifier to load
            runner_type: Type of runner to use (default: "mlx_llama")
            device_config: Optional device configuration
            as_subprocess: Whether to run as subprocess
        """
        super().__init__(agent_context, as_subprocess)

        self.model_id = model_id
        self.runner_type = runner_type
        self.device_config = device_config or {}
        self.runner: BaseRunner | None = None

        # Initialize the runner
        self._initialize_runner()

    def _initialize_runner(self) -> None:
        """Initialize the specified runner."""
        # Ensure runners are registered before trying to use them
        ensure_runners_registered()

        runner_class = get_runner(self.runner_type)
        if not runner_class:
            raise ValueError(f"Unknown runner type: {self.runner_type}")

        self.logger.info(f"Initializing {self.runner_type} runner with model: {self.model_id}")
        runner = runner_class()
        runner.load(self.model_id, self.device_config)
        self.runner = runner

    def _create_generation_config(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None
    ) -> GenerationConfig:
        """Create generation config with defaults from agent context."""
        params = self._resolve_generation_params(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        return GenerationConfig(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            stop=stop
        )

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
        chat_id: int | None = None
    ):
        """Complete text using the local runner."""
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        if not system_prompt:
            system_prompt = SYSTEM_PROMPT

        # Create generation config
        gen_config = self._create_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )

        # Generate completion using runner
        completion_result = self.runner.complete(
            system_prompt=system_prompt,
            user_prompt=prompt,
            generation_config=gen_config
        )

        # Convert to LlamaCompletion format for compatibility
        completion = LlamaCompletion.from_dict(completion_result)

        # Add to chat history
        ai_message = completion.get_assistant_content()
        chat_history = self._agent_context._add_to_chat_history(
            user_message=prompt,
            ai_message=ai_message,
            chat_id=chat_id
        )

        completion.chat_id = chat_history.chat_session_id
        return completion

    def _complete_raw(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int = None, temperature: float = None) -> str:
        """
        Single completion returning plain text, without chat-history side effects.
        Used for tool-call loops so intermediate attempts don't pollute history.
        """
        if not self.runner:
            raise RuntimeError("Runner not initialized")
        gen_config = self._create_generation_config(max_tokens=max_tokens, temperature=temperature)
        completion_result = self.runner.complete(
            system_prompt=system_prompt or SYSTEM_PROMPT,
            user_prompt=prompt,
            generation_config=gen_config
        )
        completion = LlamaCompletion.from_dict(completion_result)
        return next((msg.content for msg in completion.messages if msg.role == 'assistant'), "")

    def complete_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = None,
        temperature: float = None,
        max_tool_iterations: int = 3,
        chat_id: int | None = None,
    ) -> ToolCompletion:
        """
        Tool-augmented completion for local models via schema-grounded prompting.

        Local models keep the reflection-based function visibility: adapter
        actions are reflected into JSON schemas and rendered into the prompt.
        Robustness comes from tolerant JSON extraction, schema validation with
        type coercion, and validation-error feedback on retries.
        """
        result = run_prompt_tool_call(
            complete_fn=lambda task, system: self._complete_raw(
                task, system_prompt or system, max_tokens=max_tokens, temperature=temperature
            ),
            schemas=self._agent_context.get_tool_schemas(),
            dispatcher=self._agent_context.get_tool_dispatcher(),
            task_prompt=prompt,
            max_attempts=max_tool_iterations,
        )
        if chat_id is not None:
            try:
                self._agent_context._add_to_chat_history(
                    user_message=prompt, ai_message=result.to_content(), chat_id=chat_id
                )
            except Exception as e:
                self.logger.warning(f"Failed to record tool completion in chat history: {e}")
        return ToolCompletion(
            content=result.to_content(),
            tool_results=[result],
            iterations=1,
            used_native_tools=False,
        )

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
        chat_id: int | None = None
    ):
        """Stream text completion - delegates to complete_text for now."""
        return self.complete_text(
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            echo=echo,
            best_of=best_of,
            prompt_tokens=prompt_tokens,
            response_format=response_format,
            system_prompt=system_prompt,
            chat_id=chat_id
        )

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
        chat_id: int | None = None
    ):
        """Audio completion not implemented for LocalAgent."""
        self.logger.warning("Audio completion not implemented for LocalAgent")
        raise NotImplementedError("Audio completion not supported for LocalAgent")

    def connect_realtime_audio(self):
        """Real-time audio not implemented for LocalAgent."""
        self.logger.warning("Real-time audio not implemented for LocalAgent")
        raise NotImplementedError("Real-time audio not supported for LocalAgent")

    def _cleanup_resources(self):
        """Release the model runner."""
        if self.runner:
            self.runner.cleanup()
