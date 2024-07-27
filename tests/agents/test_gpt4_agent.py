from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from agents.gpt4_agent import WORLD_TICK_PROMPT, TASK_TICK_PROMPT, EXECUTION_TICK_PROMPT
from app.main import app
import pytest

def is_function_prompt(prompt: str) -> bool:
    return 'Only call functions that are listed in our adapter list.' in prompt

def is_task_prompt(prompt: str) -> bool:
    return 'actionable tasks' in prompt

def completions_generator(prompt: str) -> str:
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
            {"message": {'content': f"""{{
    \"class\" : \"LogAdapter\",
    \"function\": \"log\",
    \"parameters\": {{
        \"output\": \"logging a haiku!\"
    }}
}}"""}}
        ]
    }
    if is_function_prompt_flag:
        return function_completion
    
    if is_task_prompt_flag:
        return task_completion
    
    return base_completion

def get_mock_context(include_world_processing: bool = False) -> dict[str, str]:
    context = MagicMock()
    state = {
        'world_context' : '',
        'task_context' : '',
        'execution_context' : '',
        'ticks' : 0,
        'include_world_processing': include_world_processing
    }
    context.world_context.return_value = state['world_context']
    context.task_context.return_value = state['task_context']
    context.execution_context.return_value = state['execution_context']
    return context

mock_context_variations = [
    get_mock_context(include_world_processing=False), get_mock_context(include_world_processing=True)
]

def get_mock_gpt4_agent(mock_context: any) -> MagicMock:
    mock_agent = MagicMock()
    mock_agent.complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    mock_agent._agent_context = mock_context
    return mock_agent

mock_gpt4_agent_variations = [get_mock_gpt4_agent(context) for context in mock_context_variations]

def test_assert_prompt_invariants():
    assert not is_function_prompt(WORLD_TICK_PROMPT) and not is_task_prompt(WORLD_TICK_PROMPT)
    assert is_task_prompt(TASK_TICK_PROMPT) and not is_function_prompt(TASK_TICK_PROMPT)
    assert not is_task_prompt(EXECUTION_TICK_PROMPT) and is_function_prompt(EXECUTION_TICK_PROMPT)

@patch('app.main.GPT4Agent')
@patch('agents.gpt4_agent.GPT4Agent.complete_text')
@patch('adapters.log_adapter.LogAdapter.log')
def test_tick_with_prompt(log, complete_text, mock_gpt4_agent, gpt4agent, client):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    mock_gpt4_agent.return_value = gpt4agent
    log.side_effect = lambda output: print(output)
    
    # Prepare the request payload
    payload = {
        "prompt": "Write a haiku"
    }

    # Send a POST request to the /complete_text endpoint
    response = client.post("agent/initialize_task_and_tick", json=payload)

    # Assert the response status code and content
    assert response.status_code == 200
    response_payload = response.json()

    print(response_payload)
    assert response_payload == {
    'world_context': [],
    'task_context': [],
    'execution_context': []
    }
    assert log.call_count == 2

@patch('app.main.get_gpt4_client')
@patch('agents.gpt4_agent.GPT4Agent')
@patch('agents.gpt4_agent.GPT4Agent.complete_text')
@patch('adapters.log_adapter.LogAdapter.log')
def test_tick_with_prompt_and_world_processing_parameters(log, complete_text, mock_gpt4_agent, mock_gpt4_agent2, gpt4agent, client, process_world_variation_gpt4agents):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    #initialization patching / injection
    mock_gpt4_agent.return_value = gpt4agent
    mock_gpt4_agent2.return_value = gpt4agent
    log.side_effect = lambda output: print(output)
    print(process_world_variation_gpt4agents)
    
    # Prepare the request payload
    payload = {
        "prompt": "Write a haiku"
    }

    # Send a POST request to the /complete_text endpoint
    response = client.post("agent/initialize_task_and_tick", json=payload)

    # Assert the response status code and content
    assert response.status_code == 200
    response_payload = response.json()

    print(response_payload)
    assert response_payload == {
    'world_context': [],
    'task_context': [],
    'execution_context': []
    }
    assert log.call_count == 2




