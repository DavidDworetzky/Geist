"""
LocalAgent implementation for running local inference models.
"""
import logging
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent
from agents.agent_context import AgentContext
from agents.architectures import get_runner
from agents.architectures.base_runner import GenerationConfig
from agents.architectures.registry import ensure_runners_registered
from agents.models.agent_completion import AgentCompletion
from agents.models.llama_completion import LlamaCompletion
from agents.tool_calling import (
    ToolCallError,
    ToolCompletion,
    parse_tool_call,
    run_prompt_tool_call,
    validate_tool_call,
)
from adapters.tool_schema import render_tool_prompt
import subprocess
import os
import signal
import psutil
from agents.response_utils import (
    AgentResponseMixin,
    WORLD_TICK_PROMPT,
    TASK_TICK_PROMPT,
    EXECUTION_TICK_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class LocalAgent(AgentResponseMixin, BaseAgent):
    """
    Local agent implementation that uses pluggable runners for inference.
    """
    
    def __init__(
        self,
        agent_context: AgentContext,
        model_id: str,
        runner_type: str = "mlx_llama",
        device_config: Optional[Dict[str, Any]] = None,
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
        
        self.logger = logging.getLogger(__name__)
        self.model_id = model_id
        self.runner_type = runner_type
        self.device_config = device_config or {}
        self.runner = None
        
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
        self.runner = runner_class()
        self.runner.load(self.model_id, self.device_config)
    
    def _create_generation_config(
        self,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None
    ) -> GenerationConfig:
        """Create generation config with defaults from agent context."""
        return GenerationConfig(
            max_tokens=max_tokens or self._agent_context.settings.max_tokens or 16,
            temperature=temperature or self._agent_context.settings.temperature or 1.0,
            top_p=top_p or self._agent_context.settings.top_p or 1.0,
            frequency_penalty=frequency_penalty or self._agent_context.settings.frequency_penalty or 0.0,
            presence_penalty=presence_penalty or self._agent_context.settings.presence_penalty or 0.0,
            stop=stop
        )

    def complete_text(
        self,
        prompt: str,
        max_tokens: int = None,
        n: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        echo: bool = False,
        best_of: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        response_format: str = "text",
        system_prompt: str = None,
        chat_id: Optional[int] = None
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
        
        # Add to chat history using the common extraction method
        ai_message = completion.get_assistant_content()
        chat_history = self._agent_context._add_to_chat_history(
            user_message=prompt,
            ai_message=ai_message,
            chat_id=chat_id
        )

        completion.chat_id = chat_history.chat_session_id
        return completion

    def _complete_raw(self, prompt: str, system_prompt: Optional[str] = None,
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
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        max_tool_iterations: int = 3,
        chat_id: Optional[int] = None,
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
        max_tokens: int = None,
        n: int = None,
        stop: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        echo: bool = False,
        best_of: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        response_format: str = "text",
        system_prompt: Optional[str] = None,
        chat_id: Optional[int] = None
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
        """Audio completion not implemented for LocalAgent."""
        self.logger.warning("Audio completion not implemented for LocalAgent")
        raise NotImplementedError("Audio completion not supported for LocalAgent")
    
    def connect_realtime_audio(self):
        """Real-time audio not implemented for LocalAgent."""
        self.logger.warning("Real-time audio not implemented for LocalAgent")
        raise NotImplementedError("Real-time audio not supported for LocalAgent")
    
    def initialize(self, task_prompt: str = None):
        """Initialize the agent with optional task."""
        if task_prompt:
            self._agent_context.task_context.append(task_prompt)

        if self.as_subprocess:
            self.logger.info("Initializing agent with subprocess.")
            try:
                process = subprocess.Popen(
                    ['python3', '-u', 'tick.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self._agent_context.subprocess_id = process.pid
            except Exception as e:
                self.logger.error(f"Failed to start subprocess: {e}")
                self._agent_context.subprocess_id = None
        else:
            self.logger.info("Initializing agent without subprocess.")
            self._agent_context.subprocess_id = None
    
    def phase_out(self):
        """Phase out the agent and clean up resources."""
        self.save_state_snapshot(reason="phase_out")
        self._agent_context._save()
        if self.runner:
            self.runner.cleanup()
        self.terminate_subprocess()

    def phase_in(self):
        """Phase in the agent and restore state from the latest snapshot."""
        self.initialize()
        self.restore_state_snapshot()
    
    def terminate_subprocess(self):
        """Terminate any running subprocess."""
        subprocess_id = self._agent_context.subprocess_id
        if subprocess_id:
            try:
                os.kill(subprocess_id, signal.SIGTERM)
            except ProcessLookupError:
                self.logger.warning(f"Process {subprocess_id} not found, may have already terminated")
            except OSError as e:
                self.logger.error(f"Error terminating subprocess {subprocess_id}: {e}")
            finally:
                self._agent_context.subprocess_id = None
    
    def tick(self):
        """Execute one agent tick and snapshot the resulting state."""
        self.logger.info("LocalAgent Tick.")
        if self._agent_context.settings.include_world_processing:
            self.tick_world()
        self.tick_tasks()
        self.tick_execution()
        self.save_state_snapshot(reason="tick")
    
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
        """Advance execution context reasoning with full tool visibility."""
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=EXECUTION_TICK_PROMPT + context_string + self._tool_visibility_prompt())
        result = self._transform_completions(result)
        split_result = result[0].split("\\n") if result else []
        self._agent_context.execution_context = split_result
        return split_result

    def _tool_visibility_prompt(self) -> str:
        """Rendered schemas of the available tools, for prompt-based execution."""
        schemas = self._agent_context.get_tool_schemas()
        if not schemas:
            return ""
        return "\nAVAILABLE_TOOLS (JSON schemas):\n" + render_tool_prompt(schemas)
    
    def _is_valid_function_json(self, function_json: str) -> bool:
        """Validate that text contains a parseable, schema-valid tool call.

        Overrides the AgentResponseMixin version: structured tool calling
        validates against adapter schemas instead of bare JSON shape.
        """
        try:
            call = parse_tool_call(function_json)
            validate_tool_call(call, self._agent_context.get_tool_schemas())
            return True
        except ToolCallError:
            return False

    def _take_json_and_call_function(self, function_json: str):
        """Parse, validate, and execute a tool call from model output.

        Overrides the AgentResponseMixin version: dispatch goes through the
        schema-validating tool dispatcher rather than raw reflection.
        """
        result = self._agent_context.get_tool_dispatcher().dispatch_text(function_json)
        if not result.success:
            raise Exception(f"Tool call failed: {result.error}")
        return result.result
