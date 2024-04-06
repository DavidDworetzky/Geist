from fastapi.testclient import TestClient
import app, get_gpt4_client
from unittest.mock import patch

client = TestClient(app)

@patch('main.get_gpt4_client')
def test_complete_text_endpoint(mock_get_gpt4_client):
    # Create a mock GPT4Agent instance
    mock_agent = mock_get_gpt4_client.return_value
    mock_agent.complete_text.return_value = {
        "completions": [
            {"text": "Sample completion 1"},
            {"text": "Sample completion 2"}
        ]
    }

    # Prepare the request payload
    payload = {
        "prompt": "Test Prompt",
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
    response = client.post("/complete_text", json=payload)

    # Assert the response status code and content
    assert response.status_code == 200
    assert response.json() == {
        "completions": [
            {"text": "Sample completion 1"},
            {"text": "Sample completion 2"}
        ]
    }

    # Verify that the GPT4Agent's complete_text method was called with the correct arguments
    mock_agent.complete_text.assert_called_once_with(
        prompt="Test prompt",
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