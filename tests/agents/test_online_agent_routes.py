"""
Route-level tests for the OnlineAgent-backed completion and tick endpoints.

These cover the behavior formerly exercised through GPT4Agent: text completion
via /agent/complete_text, and the OODA tick loop (task expansion + adapter
function dispatch) via /agent/initialize_task_and_tick.
"""
from unittest.mock import patch

from agents.base_agent import EXECUTION_TICK_PROMPT, TASK_TICK_PROMPT, WORLD_TICK_PROMPT
from app.main import AgentType, agent_cache


HAIKU_COMPLETION = {
  "id": "chatcmpl-AHZzoFcxDG62aTvWx1jS0a2VGkitH",
  "object": "chat.completion",
  "created": 1728753308,
  "model": "gpt-4-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Silent orb of night,\nGlowing in soft silver light,\nGuiding",
        "refusal": None
      },
      "logprobs": None,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 16,
    "total_tokens": 30,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  },
  "system_fingerprint": None
}

#by convention, if EXISTS is set for a key, existence is checked but not equivalence
agent_completion = {
    "chat_id" : "EXISTS",
    "message" : ["Silent orb of night,\nGlowing in soft silver light,\nGuiding"],
    "id" : "EXISTS",
}

def is_function_prompt(prompt: str) -> bool:
    return 'Only call functions that are listed in our adapter list.' in prompt

def is_task_prompt(prompt: str) -> bool:
    return 'actionable tasks' in prompt

def completions_generator(prompt: str) -> dict:
    '''
    Trivial mock completions function that outputs a modified version of the prompt for the output.
    '''
    is_function_prompt_flag = is_function_prompt(prompt)
    is_task_prompt_flag = is_task_prompt(prompt)
    base_completion = {
        "choices":
        [
            {"message": {'content': f"{prompt}:completion1"}},
            {"message": {'content': f"{prompt}:completion2"}}
        ]
    }
    task_completion = {
        "choices":
        [
            {"message" : {'content': "log the beginning of a haiku"}},
            {"message" : {'content': "log the end of a haiku"}}
        ]
    }
    function_completion = {
        "choices":
        [
            {"message": {'content': """{
    \"class\" : \"LogAdapter\",
    \"function\": \"log\",
    \"parameters\": {
        \"output\": \"logging a haiku!\"
    }
}"""}}
        ]
    }
    if is_function_prompt_flag:
        return function_completion

    if is_task_prompt_flag:
        return task_completion

    return base_completion


def expected_world_context(task: str) -> list[str]:
    """World context produced by one world tick over an empty world with a single task."""
    world_prompt = WORLD_TICK_PROMPT + "WORLD_CONTEXT:" + f"TASK_CONTEXT:{task}"
    return [f"{world_prompt}:completion1", f"{world_prompt}:completion2"]


def test_assert_prompt_invariants():
    assert not is_function_prompt(WORLD_TICK_PROMPT) and not is_task_prompt(WORLD_TICK_PROMPT)
    assert is_task_prompt(TASK_TICK_PROMPT) and not is_function_prompt(TASK_TICK_PROMPT)
    assert not is_task_prompt(EXECUTION_TICK_PROMPT) and is_function_prompt(EXECUTION_TICK_PROMPT)


@patch('agents.online_agent.OnlineAgent._make_request')
@patch('adapters.log_adapter.LogAdapter.log', autospec=True)
def test_completion(log, make_request, online_agent, client):
    make_request.return_value = HAIKU_COMPLETION
    log.side_effect = lambda self, output: print(output)
    agent_cache[AgentType.GPT4AGENT] = online_agent
    try:
        # prepare the request payload for the completion
        payload = {
            "prompt": "write a haiku about the moon",
            "max_tokens": 1024,
            "n": 1,
            "stop": [
                "string"
            ],
            "temperature": 1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "echo": False,
            "best_of": 0,
            "prompt_tokens": [
                0
            ],
            "response_format": "text",
            "agent_type": "GPT4AGENT"
        }

        # send the request
        response = client.post("agent/complete_text", json=payload)

        # assert the response
        assert response.status_code == 200
        response_payload = response.json()
        for key in agent_completion:
            if agent_completion[key] == "EXISTS":
                assert response_payload[key] is not None
            else:
                assert response_payload[key] == agent_completion[key]
    finally:
        agent_cache[AgentType.GPT4AGENT] = None


@patch('agents.online_agent.OnlineAgent.complete_text')
@patch('adapters.log_adapter.LogAdapter.log', autospec=True)
def test_tick_with_prompt_and_world_processing(log, complete_text, online_agent, client):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    log.side_effect = lambda self, output: print(output)
    agent_cache[AgentType.GPT4AGENT] = online_agent
    try:
        payload = {
            "prompt": "Write a haiku",
            "agent_type": "GPT4AGENT"
        }

        response = client.post("agent/initialize_task_and_tick", json=payload)

        assert response.status_code == 200
        response_payload = response.json()

        # World processing is enabled on this agent, so the world tick replaces
        # the world context; the task is popped and expanded, and both resulting
        # subtasks are executed as LogAdapter.log function calls.
        assert response_payload == {
            'world_context': expected_world_context("Write a haiku"),
            'task_context': [],
            'execution_context': []
        }
        assert log.call_count == 2
    finally:
        agent_cache[AgentType.GPT4AGENT] = None


@patch('agents.online_agent.OnlineAgent.complete_text')
@patch('adapters.log_adapter.LogAdapter.log', autospec=True)
def test_tick_with_prompt_without_world_processing(log, complete_text, process_world_variation_online_agents, client):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    log.side_effect = lambda self, output: print(output)
    no_world_agent = next(
        agent for agent in process_world_variation_online_agents
        if not agent._agent_context.settings.include_world_processing)
    agent_cache[AgentType.GPT4AGENT] = no_world_agent
    try:
        payload = {
            "prompt": "Write a haiku",
            "agent_type": "GPT4AGENT"
        }

        response = client.post("agent/initialize_task_and_tick", json=payload)

        assert response.status_code == 200
        # Without world processing the world tick is skipped entirely, the task
        # is consumed, and execution drains the generated subtasks.
        assert response.json() == {
            'world_context': [],
            'task_context': [],
            'execution_context': []
        }
        assert log.call_count == 2
    finally:
        agent_cache[AgentType.GPT4AGENT] = None
