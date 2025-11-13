from unittest.mock import patch
from fastapi.testclient import TestClient
import copy
from app.main import agent_cache, AgentType

ASSISTANT_REPLY_1 = "First reply from assistant"

MOCK_COMPLETION_RESPONSE = {
    "id": "mock-id",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4-0613",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ASSISTANT_REPLY_1,
                "refusal": None
            },
            "logprobs": None,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "system_fingerprint": None
}


@patch('app.main.GPT4Agent')
@patch('agents.gpt4_agent.GPT4Agent._complete_text')
def test_server_side_history_hydration_in_completion(mock__complete_text, mock_gpt4_ctor, gpt4agent, client):
    """
    Verify that when calling /agent/complete_text/{session_id} twice,
    the second call's payload includes the prior user/assistant turns.
    """
    # Clear the agent cache to ensure we use the newly mocked agent
    agent_cache[AgentType.GPT4AGENT] = None

    captured_payloads = []

    def capture_payload(url, payload, headers):
        captured_payloads.append(payload)
        return copy.deepcopy(MOCK_COMPLETION_RESPONSE)

    mock__complete_text.side_effect = capture_payload
    # Set the mock constructor to return the real agent fixture with mocked _complete_text
    mock_gpt4_ctor.return_value = gpt4agent

    session_id = 4321

    # First turn
    r1 = client.post(f"/agent/complete_text/{session_id}", json={
        "prompt": "First message",
        "agent_type": "GPT4AGENT"
    })
    assert r1.status_code == 200

    # Second turn
    r2 = client.post(f"/agent/complete_text/{session_id}", json={
        "prompt": "Second message",
        "agent_type": "GPT4AGENT"
    })
    assert r2.status_code == 200

    # Ensure we captured both payloads sent to the provider
    assert len(captured_payloads) == 2, f"Expected 2 payloads, got {len(captured_payloads)}"

    # First call should include only system (optional) + current user turn
    first_messages = captured_payloads[0]["messages"]
    print(f"\nFirst call messages: {first_messages}")
    assert first_messages[-1]["role"] == "user"
    assert first_messages[-1]["content"] == "First message"

    # Second call should include hydrated prior turns plus current user
    second_messages = captured_payloads[1]["messages"]
    print(f"\nSecond call messages: {second_messages}")
    print(f"Number of messages in second call: {len(second_messages)}")

    # Last message is current user
    assert second_messages[-1]["role"] == "user"
    assert second_messages[-1]["content"] == "Second message"

    # Prior two should be the previous user and assistant reply
    assert len(second_messages) >= 3, f"Expected at least 3 messages in second call, got {len(second_messages)}"
    assert second_messages[-3]["role"] == "user"
    assert second_messages[-3]["content"] == "First message"
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["content"] == ASSISTANT_REPLY_1


