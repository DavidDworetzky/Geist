from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

@patch('app.main.GPT4Agent')
def test_complete_text_endpoint(mock_gpt4_agent_class, client):
    # Create a mock GPT4Agent instance
    mock_agent = MagicMock()
    mock_agent.complete_text.return_value = {
        "completions": [
            {"text": "Sample completion 1"},
            {"text": "Sample completion 2"}
        ]
    }
    mock_gpt4_agent_class.return_value = mock_agent

    # Prepare the request payload
    payload = {
        "prompt": "Write a haiku",
        "max_tokens": 50,
        "n": 2,
        "stop": None,
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "echo": False,
        "best_of": 1,
        "prompt_tokens": None,
        "response_format": "json"
    }

    # Send a POST request to the /complete_text endpoint
    response = client.post("agent/complete_text", json=payload)

    # Assert the response status code and content
    assert response.status_code == 200
    response_payload = response.json()
    print(response_payload)
    assert response_payload == {
        "completions": [
            {"text": "Sample completion 1"},
            {"text": "Sample completion 2"}
        ]
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