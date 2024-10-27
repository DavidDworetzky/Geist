from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from agents.llama_agent import WORLD_TICK_PROMPT, TASK_TICK_PROMPT, EXECUTION_TICK_PROMPT
from app.main import app
from agents.llama_agent import LlamaAgent
from agents.models.llama_completion import LlamaCompletion, Message
import pytest

HAIKU_COMPLETION = "Silent orb of night,\nGlowing in soft silver light,\nGuiding"

agent_completion = {
    "message": ["Silent orb of night,\nGlowing in soft silver light,\nGuiding"],
    "id": "chatcmpl-AHZzoFcxDG62aTvWx1jS0a2VGkitH"
}       

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
@patch('agents.llama_agent.LlamaAgent.complete_text')
@patch('adapters.log_adapter.LogAdapter.log')
def test_llama_tick_with_prompt(log, complete_text, mock_llama_agent, llama_agent, client):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    mock_llama_agent.return_value = llama_agent
    log.side_effect = lambda output: print(output)
    
    # Prepare the request payload
    payload = {
        "prompt": "Write a haiku",
        "agent_type" : "LLAMA"
    }

    # Send a POST request to the /complete_text endpoint
    response = client.post("agent/initialize_task_and_tick", json=payload)

    # Assert the response status code and content
    assert response.status_code == 200
    response_payload = response.json()

    print(response_payload)
    assert response_payload == {
        'world_context': [
            'You are a world class executive. Your plans are plans are direct, and detailed only if necessary. \n'
            'Given what you know about the world today, and the main task that you need to complete, consider if there are any '
            'additional facts that you should add to the list of things you consider. \n'
            "Do not add anything that doesn't need to be added, consolidate anything that is worth consolidating with "
            'simpler statements.WORLD_CONTEXT:TASK_CONTEXT:Write a haiku:completion1',
            'You are a world class executive. Your plans are plans are direct, and detailed only if necessary. \n'
            'Given what you know about the world today, and the main task that you need to complete, consider if there are any '
            'additional facts that you should add to the list of things you consider. \n'
            "Do not add anything that doesn't need to be added, consolidate anything that is worth consolidating with "
            'simpler statements.WORLD_CONTEXT:TASK_CONTEXT:Write a haiku:completion2'
        ],
        'task_context': [],
        'execution_context': []
    }
    assert log.call_count == 2

@patch('app.main.LlamaAgent')
@patch('agents.llama_agent.LlamaAgent._complete_llama_sequence')
@patch('adapters.log_adapter.LogAdapter.log')
def test_completion(log, complete_text, mock_llama_agent, llama_agent, client):
    llama_completion = LlamaCompletion(
        messages=[Message(role="assistant", content=HAIKU_COMPLETION)]
    )
    complete_text.return_value = llama_completion
    mock_llama_agent.return_value = llama_agent
    log.side_effect = lambda output: print(output)

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
        "agent_type": "LLAMA"
    }

    # send the request
    response = client.post("agent/complete_text", json=payload)

    # assert the response
    assert response.status_code == 200
    response_payload = response.json()
    assert response_payload["message"] == agent_completion["message"]
