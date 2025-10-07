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
import json
import subprocess
import os
import signal
import psutil

logger = logging.getLogger(__name__)

# Constants for agent prompting
WORLD_TICK_PROMPT = """You are a world class executive. Your plans are direct, and detailed only if necessary. 
Given what you know about the world today, and the main task that you need to complete, consider if there are any additional facts that you should add to the list of things you consider. 
Do not add anything that doesn't need to be added, consolidate anything that is worth consolidating with simpler statements."""

TASK_TICK_PROMPT = "You are a focused individual. Given the main task that you wish to complete, and current working subtasks, create a specific list of actionable tasks that will complete your problem. Delimit these as plain english separated by the | character. Do not use function calls yet - only plain english."

FUNCTION_CALL_JSON = """
{
    "class" : "class_name",
    "function": "function_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}
"""

EXECUTION_TICK_PROMPT = f"You are given a list of tasks and list of function calls that you can make. Given the state of the world, and classes available to you - formulate a function call that will help you complete your task. You should formulate the function call as {FUNCTION_CALL_JSON}. Only call functions that are listed in our adapter list."

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
        
        # Add to chat history
        ai_message = next((gen.content for gen in completion.messages if gen.role == 'assistant'), None)
        chat_history = self._agent_context._add_to_chat_history(
            user_message=prompt,
            ai_message=ai_message,
            chat_id=chat_id
        )
        
        completion.chat_id = chat_history.chat_session_id
        return completion
    
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
            process = subprocess.Popen(['python3', '-u', 'tick.py'], stdout=subprocess.PIPE)
            self._agent_context.subprocess_id = process.pid
        else:
            self.logger.info("Initializing agent without subprocess.")
            self._agent_context.subprocess_id = None
    
    def phase_out(self):
        """Phase out the agent and clean up resources."""
        self._agent_context._save()
        if self.runner:
            self.runner.cleanup()
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
        self.logger.info("LocalAgent Tick.")
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
    
    def _aggregated_context(self, world_context: bool, task_context: bool, execution_context: bool) -> str:
        """Get aggregated context string."""
        context_string = ""
        if world_context:
            context_string += "WORLD_CONTEXT:" + "\\n".join(self._agent_context.world_context)
        if task_context:
            context_string += "TASK_CONTEXT:" + "\\n".join(self._agent_context.task_context)
        if execution_context:
            context_string += "EXECUTION_CONTEXT:" + "\\n".join(self._agent_context.execution_context)
        return context_string
    
    def _transform_completions(self, completion):
        """Transform completion to list of content strings."""
        try:
            if hasattr(completion, 'messages'):
                # LlamaCompletion format
                return [msg.content for msg in completion.messages if msg.role == 'assistant']
            elif isinstance(completion, dict) and 'choices' in completion:
                # GPT-style format
                return [choice['message']['content'] for choice in completion['choices']]
            else:
                # Fallback
                return [str(completion)]
        except Exception as e:
            self.logger.error(f"Failed to transform completion: {completion}, exception: {e}")
            raise Exception(f"Completion failed to destructure: {completion}")
    
    def _is_valid_function_json(self, function_json: str) -> bool:
        """Validate function call JSON format."""
        try:
            function_json = function_json.replace('\\n', '')
            parsed_json = json.loads(function_json)
            required_keys = ["function", "parameters", "class"]
            if all(key in parsed_json for key in required_keys):
                if isinstance(parsed_json["parameters"], dict):
                    return True
            return False
        except json.JSONDecodeError:
            return False
    
    def _take_json_and_call_function(self, function_json: str):
        """Execute function call from JSON specification."""
        if not self._is_valid_function_json(function_json):
            raise Exception(f"Invalid function call json: {function_json}")
        
        json_data = json.loads(function_json)
        class_name = json_data["class"]
        adapter_class = next(
            (wrapper for wrapper in self._agent_context.initialized_classes if wrapper.name == class_name),
            None
        )
        
        if not adapter_class:
            raise Exception(f"No adapter class matching {class_name}")
        
        adapter_class = adapter_class.instance
        function_to_call = getattr(adapter_class, json_data["function"])
        parameters = json_data["parameters"]
        
        return function_to_call(**parameters)
