from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

def completions_generator(prompt: str) -> str:
    '''
    Trivial mock completions function that outputs a modified version of the prompt for the output. 
    '''
    is_function_prompt = 'function' in prompt
    is_task_prompt = 'actionable tasks' in prompt
    base_completion = {
        "completions": 
        [
            {"text": f"{prompt}:completion1"},
            {"text": f"{prompt}:completion2"}
        ]
    }
    task_completion = {
        "completions":
        [
            {"text" : "log the beginning of a haiku"},
            {"text" : "log the end of a haiku"}
        ]
    }
    function_completion = {
        "completions":
        [
            {"text": f"""{{
    \"class\" : \"LogAdapter\",
    \"function\": \"log\",
    \"parameters\": {{
        \"output\": \"logging a haiku!\"
    }}
}}"""}
        ]
    }
    if is_function_prompt:
        return function_completion
    
    if is_task_prompt:
        return task_completion
    
    return base_completion

def get_mock_context() -> dict[str, str]:
    context = MagicMock()
    state = {
        'world_context' : '',
        'task_context' : '',
        'execution_context' : '',
        'ticks' : 0
    }
    context.world_context.return_value = state['world_context']
    context.task_context.return_value = state['task_context']
    context.execution_context.return_value = state['execution_context']
    return context

def get_mock_gpt4_agent() -> MagicMock:
    mock_agent = MagicMock()
    mock_agent.complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    mock_agent._agent_context = get_mock_context()
    return mock_agent

@patch('app.main.GPT4Agent')
@patch('agents.gpt4_agent.GPT4Agent.complete_text')
def test_tick(complete_text, mock_gpt4_agent, gpt4agent, client):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    mock_gpt4_agent.return_value = gpt4agent
    # Prepare the request payload
    payload = {
        "prompt": "Write a haiku"
    }

    # Send a POST request to the /complete_text endpoint
    response = client.post("/initialize_task_and_tick", json=payload)

    # Assert the response status code and content
    assert response.status_code == 200
    response_payload = response.json()
    print(response_payload)
    assert response_payload == {'world_context': ["You are a deep and thorough thinker. \nGiven what you know about the world today, and the main task that you need to complete, consider if there are any additional important facts that you should add to the list of your knowledge. \nDo not add anything that doesn't need to be added, consolidate anything that is worth consolidating with simpler facts.WORLD_CONTEXT:TASK_CONTEXT:Write a haiku:completion1", "You are a deep and thorough thinker. \nGiven what you know about the world today, and the main task that you need to complete, consider if there are any additional important facts that you should add to the list of your knowledge. \nDo not add anything that doesn't need to be added, consolidate anything that is worth consolidating with simpler facts.WORLD_CONTEXT:TASK_CONTEXT:Write a haiku:completion2"], 'task_context': [], 'execution_context': []}

