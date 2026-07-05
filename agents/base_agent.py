import json
import logging
import os
import signal
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import psutil

from agents.agent_context import AgentContext
from agents.exceptions import AgentError, CompletionFormatError, FunctionCallError
from app.models.database.chat_session import get_chat_history


# Shared prompts for the OODA tick loop.
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

MAX_FUNCTION_JSON_RETRIES = 3


@dataclass
class GenerationParams:
    """Resolved generation parameters after merging caller values with agent settings."""
    max_tokens: int
    n: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float


class BaseAgent(ABC):
    """
    Base class for all agents.

    Subclasses implement the completion methods (`complete_text`,
    `stream_complete_text`, `complete_audio`, `connect_realtime_audio`).
    The OODA tick loop, adapter function dispatch, chat-history hydration,
    and subprocess lifecycle are shared here.
    """

    def __init__(self, agent_context: AgentContext, as_subprocess: bool = False):
        self._agent_context = agent_context
        self.as_subprocess = as_subprocess
        self.logger = logging.getLogger(type(self).__module__)

    # ------------------------------------------------------------------
    # Completion interface (provider-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def complete_text(self, prompt: str, max_tokens: int | None = None, n: int | None = None,
                      temperature: float | None = None, top_p: float | None = None,
                      frequency_penalty: float | None = None, presence_penalty: float | None = None,
                      stop: str | list[str] | None = None, echo: bool = False, best_of: int | None = None,
                      prompt_tokens: list[int] | None = None, response_format: str = "text",
                      system_prompt: str | None = None, chat_id: int | None = None):
        """Generate a completion for the given prompt."""

    @abstractmethod
    def stream_complete_text(self, prompt: str, max_tokens: int | None = None, n: int | None = None,
                             temperature: float | None = None, top_p: float | None = None,
                             frequency_penalty: float | None = None, presence_penalty: float | None = None,
                             stop: str | list[str] | None = None, echo: bool = False, best_of: int | None = None,
                             prompt_tokens: list[int] | None = None, response_format: str = "text",
                             system_prompt: str | None = None, chat_id: int | None = None) -> Iterator[str]:
        """Stream a completion for the given prompt, yielding text chunks."""

    @abstractmethod
    def complete_audio(self, audio_file, max_tokens: int | None = None, n: int | None = None,
                       temperature: float | None = None, top_p: float | None = None,
                       frequency_penalty: float | None = None, presence_penalty: float | None = None,
                       stop: str | list[str] | None = None, echo: bool = False, best_of: int | None = None,
                       prompt_tokens: list[int] | None = None, response_format: str = "text",
                       system_prompt: str | None = None, chat_id: int | None = None):
        """Generate a completion from an audio input."""

    @abstractmethod
    def connect_realtime_audio(self):
        """Connect to a realtime audio stream."""

    # ------------------------------------------------------------------
    # Generation parameter resolution
    # ------------------------------------------------------------------

    def _resolve_generation_params(self, max_tokens: int | None = None, n: int | None = None,
                                   temperature: float | None = None, top_p: float | None = None,
                                   frequency_penalty: float | None = None,
                                   presence_penalty: float | None = None) -> GenerationParams:
        """
        Merge explicit caller values with agent settings and hard defaults.

        Explicit values always win; `is None` checks keep falsy-but-valid
        values (temperature=0, penalties=0) intact.
        """
        settings = self._agent_context.settings

        def pick(value, setting, default):
            if value is not None:
                return value
            if setting is not None:
                return setting
            return default

        return GenerationParams(
            max_tokens=pick(max_tokens, settings.max_tokens, 16),
            n=pick(n, settings.n, 1),
            temperature=pick(temperature, settings.temperature, 1.0),
            top_p=pick(top_p, settings.top_p, 1.0),
            frequency_penalty=pick(frequency_penalty, settings.frequency_penalty, 0.0),
            presence_penalty=pick(presence_penalty, settings.presence_penalty, 0.0),
        )

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str, system_prompt: str | None = None,
                        chat_id: int | None = None) -> list[dict]:
        """Build a chat message list, hydrating prior turns from the stored session."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_id is not None:
            try:
                history = get_chat_history(chat_id)
                for pair in history.chat_history:
                    user_msg = pair.get("user")
                    ai_msg = pair.get("ai")
                    if user_msg is not None:
                        messages.append({"role": "user", "content": user_msg})
                    if ai_msg is not None:
                        messages.append({"role": "assistant", "content": ai_msg})
            except Exception as e:
                self.logger.warning(f"Failed to hydrate chat history for chat_id={chat_id}: {e}")
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def initialize(self, task_prompt: str | None = None):
        """Initialize the agent, optionally pushing a task and starting the tick subprocess."""
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
        """Persist agent state, release provider resources, and stop any subprocess."""
        self._agent_context._save()
        self._cleanup_resources()
        self.terminate_subprocess()

    def phase_in(self):
        """Restart the agent and rehydrate its state."""
        self.initialize()

    def _cleanup_resources(self):  # noqa: B027 - optional hook, subclasses override as needed
        """Release provider-specific resources (HTTP clients, model runners). Override as needed."""

    def terminate_subprocess(self):
        """Terminate the agent's tick subprocess if one is running."""
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

    def is_subprocess_running(self) -> bool:
        subprocess_id = self._agent_context.subprocess_id
        if not subprocess_id:
            return False
        try:
            return psutil.Process(subprocess_id).is_running()
        except psutil.NoSuchProcess:
            return False

    def state(self):
        logging.info("getting agent state.")
        return {
            "world_context": self._agent_context.world_context,
            "task_context": self._agent_context.task_context,
            "execution_context": self._agent_context.execution_context
        }

    # ------------------------------------------------------------------
    # OODA tick loop
    # ------------------------------------------------------------------

    def tick(self):
        """Run one observe/orient/decide/act cycle: world reasoning, task expansion, execution."""
        self.logger.info("Agent tick.")
        if self._agent_context.settings.include_world_processing:
            self.tick_world()
        self.tick_tasks()
        self.tick_execution()

    def tick_world(self):
        """Advance world-state reasoning and replace the world context."""
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=False)
        result = self.complete_text(prompt=WORLD_TICK_PROMPT + context_string)
        split_result = self._transform_completions(result)
        self._agent_context.world_context = split_result
        return split_result

    def tick_tasks(self):
        """Pop the next task and expand it into '|'-delimited execution subtasks."""
        if not self._agent_context.task_context:
            raise AgentError("No tasks available in task context for execution.")
        task_to_execute = self._agent_context.task_context.pop(0)
        prompt = f"executing task: {task_to_execute}" + self._aggregated_context(
            world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=TASK_TICK_PROMPT + prompt)
        completions = self._transform_completions(result)
        subtasks = [task.strip() for completion in completions for task in completion.split('|')]
        self._agent_context.execution_context = subtasks
        return subtasks

    def tick_execution(self):
        """Resolve each pending execution subtask into an adapter function call and run it."""
        results = []
        for task in self._agent_context.execution_context:
            context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
            function_json = self._request_function_json(task, context_string)
            if function_json is None:
                continue
            results.append(self._take_json_and_call_function(function_json))

        self._agent_context.execution_context = []
        return results

    def _request_function_json(self, task: str, context_string: str) -> str | None:
        """Ask the model for a valid function-call JSON, retrying on malformed output."""
        prompt = f"task: {task}" + EXECUTION_TICK_PROMPT + context_string
        candidate = None
        for attempt in range(MAX_FUNCTION_JSON_RETRIES + 1):
            completions = self._transform_completions(self.complete_text(prompt=prompt))
            if not completions:
                self.logger.error(f"Empty completion on attempt {attempt + 1} for task: {task}")
                continue
            candidate = completions[0]
            if self._is_valid_function_json(candidate):
                return candidate
        if candidate is None:
            self.logger.error(f"No completion produced for task: {task}")
            return None
        raise FunctionCallError(
            f"Exceeded retries for valid function call JSON; last result: {candidate}")

    # ------------------------------------------------------------------
    # Context aggregation and completion parsing
    # ------------------------------------------------------------------

    def _aggregated_context(self, world_context: bool, task_context: bool, execution_context: bool) -> str:
        """Aggregate the requested context sections into a single prompt string."""
        context_string = ""
        if world_context and self._agent_context.settings.include_world_processing:
            context_string += "WORLD_CONTEXT:" + "\n".join(self._agent_context.world_context)
        if task_context:
            context_string += "TASK_CONTEXT:" + "\n".join(self._agent_context.task_context)
        if execution_context:
            context_string += "EXECUTION_CONTEXT:" + "\n".join(self._agent_context.execution_context)
        return context_string

    def _transform_completions(self, completion) -> list[str]:
        """Extract assistant message contents from any supported completion shape."""
        try:
            if hasattr(completion, 'choices') and completion.choices:
                return [choice.message.content for choice in completion.choices]
            if hasattr(completion, 'messages'):
                return [msg.content for msg in completion.messages if msg.role == 'assistant']
            if isinstance(completion, dict) and 'choices' in completion:
                return [choice['message']['content'] for choice in completion['choices']]
            return [str(completion)]
        except Exception as e:
            self.logger.error(f"Failed to transform completion: {completion}, exception: {e}")
            raise CompletionFormatError(
                f"Completion failed to destructure: {completion}. "
                "Is your LLM protocol returning the correct format?") from e

    # ------------------------------------------------------------------
    # Adapter function dispatch
    # ------------------------------------------------------------------

    def _is_valid_function_json(self, function_json: str) -> bool:
        """Validate the model-produced function call JSON structure."""
        try:
            parsed_json = json.loads(function_json.replace('\n', ''))
        except json.JSONDecodeError:
            return False
        required_keys = ["function", "parameters", "class"]
        return (all(key in parsed_json for key in required_keys)
                and isinstance(parsed_json["parameters"], dict))

    def _take_json_and_call_function(self, function_json: str):
        """Dispatch a validated function-call JSON to the matching adapter instance."""
        if not self._is_valid_function_json(function_json):
            raise FunctionCallError(f"Invalid function call json: {function_json}")

        json_data = json.loads(function_json.replace('\n', ''))
        class_name = json_data["class"]
        function_name = json_data["function"]

        wrapper = next(
            (wrapper for wrapper in self._agent_context.initialized_classes if wrapper.name == class_name),
            None
        )
        if not wrapper:
            raise FunctionCallError(f"No adapter class matching {class_name}")

        # Only public adapter methods are callable from model output.
        if function_name.startswith("_"):
            raise FunctionCallError(f"Refusing to call non-public adapter method: {function_name}")
        function_to_call = getattr(wrapper.instance, function_name, None)
        if not callable(function_to_call):
            raise FunctionCallError(f"Adapter {class_name} has no callable function {function_name}")

        return function_to_call(**json_data["parameters"])
