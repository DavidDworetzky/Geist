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


@patch('app.main.LlamaAgent')
@patch('agents.llama_agent.LLamaAgent.complete_text')
@patch('adapters.log_adapter.LogAdapter.log')
def llama_test_tick_with_prompt(log, complete_text, mock_llama_agent, llama_agent, client):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    mock_llama_agent.return_value = llama_agent
    log.side_effect = lambda output: print(output)
    
    # Prepare the request payload
    payload = {
        "prompt": "Write a haiku",
        "agent_type" : "LLAMAAGENT"
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