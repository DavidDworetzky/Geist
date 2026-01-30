"""
Shared response handling utilities for agents.

This module consolidates common response handling, transformation, and utility
methods that were previously duplicated across multiple agent implementations.
"""
import json
import logging
from typing import List, Optional, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Shared prompt constants
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


def get_assistant_message(completion: Any) -> Optional[str]:
    """
    Extract assistant message from any completion type.

    Handles:
    - GPT4Completion / GenericCompletion (choices[0].message.content)
    - LlamaCompletion (messages where role='assistant')
    - Raw dicts with 'choices' key

    Args:
        completion: Completion object from any supported format

    Returns:
        The assistant message content, or None if not found
    """
    # Handle LlamaCompletion style (has messages attribute)
    if hasattr(completion, 'messages'):
        for msg in completion.messages:
            if hasattr(msg, 'role') and msg.role == 'assistant':
                return msg.content
        return None

    # Handle GPT4Completion / GenericCompletion style (has choices attribute)
    if hasattr(completion, 'choices') and completion.choices:
        first_choice = completion.choices[0]
        if hasattr(first_choice, 'message'):
            return first_choice.message.content

    # Handle raw dict responses
    if isinstance(completion, dict):
        if 'choices' in completion and completion['choices']:
            return completion['choices'][0].get('message', {}).get('content')
        if 'messages' in completion:
            for msg in completion['messages']:
                if msg.get('role') == 'assistant':
                    return msg.get('content')

    return None


def transform_completions(completion: Any) -> List[str]:
    """
    Transform completion to list of content strings.

    Unified transformation that handles all completion formats.

    Args:
        completion: Completion object from any supported format

    Returns:
        List of message content strings
    """
    try:
        # Handle LlamaCompletion style
        if hasattr(completion, 'messages'):
            return [msg.content for msg in completion.messages if msg.role == 'assistant']

        # Handle GPT4Completion / GenericCompletion style
        if hasattr(completion, 'choices') and completion.choices:
            return [choice.message.content for choice in completion.choices]

        # Handle raw dict with choices
        if isinstance(completion, dict) and 'choices' in completion:
            return [choice['message']['content'] for choice in completion['choices']]

        # Fallback
        return [str(completion)]

    except Exception as e:
        logger.error(f"Failed to transform completion: {completion}, exception: {e}")
        raise ValueError(f"Completion failed to destructure: {completion}. Format interop failure.")


def is_valid_function_json(function_json: str) -> bool:
    """
    Validate function call JSON format.

    Args:
        function_json: JSON string representing a function call

    Returns:
        True if the JSON is valid and contains required keys
    """
    try:
        function_json = function_json.replace('\n', '')
        parsed_json = json.loads(function_json)
        required_keys = ["function", "parameters", "class"]
        if all(key in parsed_json for key in required_keys):
            if isinstance(parsed_json["parameters"], dict):
                return True
        return False
    except json.JSONDecodeError:
        return False


def execute_function_call(function_json: str, initialized_classes: List[Any]) -> Any:
    """
    Execute function call from JSON specification.

    Args:
        function_json: JSON string containing function call spec
        initialized_classes: List of wrapper objects with name and instance attributes

    Returns:
        Result of the function call

    Raises:
        ValueError: If JSON is invalid or adapter class not found
    """
    if not is_valid_function_json(function_json):
        raise ValueError(f"Invalid function call json: {function_json}")

    json_data = json.loads(function_json)
    class_name = json_data["class"]

    adapter_wrapper = next(
        (wrapper for wrapper in initialized_classes if wrapper.name == class_name),
        None
    )

    if not adapter_wrapper:
        raise ValueError(f"No adapter class matching {class_name}")

    adapter_instance = adapter_wrapper.instance
    function_to_call = getattr(adapter_instance, json_data["function"])
    parameters = json_data["parameters"]

    return function_to_call(**parameters)


def build_aggregated_context(
    agent_context: Any,
    world_context: bool,
    task_context: bool,
    execution_context: bool
) -> str:
    """
    Build aggregated context string from agent context.

    Args:
        agent_context: Agent context object
        world_context: Include world context
        task_context: Include task context
        execution_context: Include execution context

    Returns:
        Aggregated context string
    """
    context_string = ""

    if world_context and getattr(agent_context.settings, 'include_world_processing', False):
        context_string += "WORLD_CONTEXT:" + "\n".join(agent_context.world_context)

    if task_context:
        context_string += "TASK_CONTEXT:" + "\n".join(agent_context.task_context)

    if execution_context:
        context_string += "EXECUTION_CONTEXT:" + "\n".join(agent_context.execution_context)

    return context_string


@dataclass
class GenerationParams:
    """Normalized generation parameters with defaults."""
    max_tokens: int = 16
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


def normalize_generation_params(
    agent_settings: Any,
    max_tokens: int = None,
    n: int = None,
    temperature: float = None,
    top_p: float = None,
    frequency_penalty: float = None,
    presence_penalty: float = None
) -> GenerationParams:
    """
    Normalize generation parameters with defaults from agent settings.

    Args:
        agent_settings: Agent settings object
        max_tokens: Override for max_tokens
        n: Override for n
        temperature: Override for temperature
        top_p: Override for top_p
        frequency_penalty: Override for frequency_penalty
        presence_penalty: Override for presence_penalty

    Returns:
        GenerationParams with normalized values
    """
    return GenerationParams(
        max_tokens=max_tokens or getattr(agent_settings, 'max_tokens', None) or 16,
        n=n or getattr(agent_settings, 'n', None) or 1,
        temperature=temperature or getattr(agent_settings, 'temperature', None) or 1.0,
        top_p=top_p or getattr(agent_settings, 'top_p', None) or 1.0,
        frequency_penalty=frequency_penalty or getattr(agent_settings, 'frequency_penalty', None) or 0.0,
        presence_penalty=presence_penalty or getattr(agent_settings, 'presence_penalty', None) or 0.0
    )


def build_chat_messages(
    prompt: str,
    system_prompt: Optional[str],
    chat_history: Optional[Any]
) -> List[dict]:
    """
    Build chat messages list with optional system prompt and history.

    Args:
        prompt: Current user prompt
        system_prompt: Optional system prompt
        chat_history: Optional chat history object with chat_history attribute

    Returns:
        List of message dicts suitable for chat completion APIs
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if chat_history:
        for pair in chat_history.chat_history:
            user_msg = pair.get("user")
            ai_msg = pair.get("ai")
            if user_msg is not None:
                messages.append({"role": "user", "content": user_msg})
            if ai_msg is not None:
                messages.append({"role": "assistant", "content": ai_msg})

    messages.append({"role": "user", "content": prompt})
    return messages


class AgentResponseMixin:
    """
    Mixin class providing shared response handling methods for agents.

    Agents should inherit from this mixin to get common utility methods
    for response transformation, context aggregation, and function calling.

    Requirements:
        - Agent must have _agent_context attribute with settings and initialized_classes
    """

    def _transform_completions(self, completion: Any) -> List[str]:
        """Transform completion to list of content strings."""
        return transform_completions(completion)

    def _aggregated_context(
        self,
        world_context: bool,
        task_context: bool,
        execution_context: bool
    ) -> str:
        """Get aggregated context string."""
        return build_aggregated_context(
            self._agent_context,
            world_context,
            task_context,
            execution_context
        )

    def _is_valid_function_json(self, function_json: str) -> bool:
        """Validate function call JSON format."""
        return is_valid_function_json(function_json)

    def _take_json_and_call_function(self, function_json: str) -> Any:
        """Execute function call from JSON specification."""
        return execute_function_call(function_json, self._agent_context.initialized_classes)

    def _normalize_params(
        self,
        max_tokens: int = None,
        n: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None
    ) -> GenerationParams:
        """Normalize generation parameters with defaults from settings."""
        return normalize_generation_params(
            self._agent_context.settings,
            max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty
        )

    def _get_assistant_message(self, completion: Any) -> Optional[str]:
        """Extract assistant message from completion."""
        return get_assistant_message(completion)
