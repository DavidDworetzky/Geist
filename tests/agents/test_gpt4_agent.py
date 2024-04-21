from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

def completions_generator(input: str) -> str:
    '''
    Trivial mock completions function that outputs a modified version of the prompt for the output. 
    '''
    base_completion = {
        "completions": 
        [
            {"text": f"{input}:completion1"},
            {"text": f"{input}:completion2"}
        ]
    }
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
    mock_agent.complete_text.side_effect = completions_generator
    mock_agent._agent_context = get_mock_context()
    mock_agent.tick.side_effect = tick_world
    return mock_agent

def tick_world(context, agent, world_prompt, task_prompt, execution_prompt):
    context.world_context += agent.complete_text(context.world_context, world_prompt)
    context.task_context += agent.complete_text(context.task_context, task_prompt)
    context.execution_context += agent.complete_text(context.execution_context, execution_prompt)
    context.tick += 1
    return context

@patch('app.main.GPT4Agent')
def test_tick(client):
    mock_agent = get_mock_gpt4_agent()
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
    assert response_payload == {
        "task_context" : "",
        "world_context" : "",
        "execution_context" : ""
    }

    # Verify that the GPT4Agent's complete_text method was called with the correct arguments
    mock_agent.complete_text.assert_called_once_with(
        prompt="Write a haiku",
        max_tokens=50,
        n=2,
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        echo=False,
        best_of=1,
        prompt_tokens=None,
        response_format="json"
    )