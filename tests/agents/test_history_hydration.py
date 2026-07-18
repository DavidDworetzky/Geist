from unittest.mock import patch

from agents.models.tool_calling import ModelEvent, ModelTurn
from app.main import AgentType, agent_cache


ASSISTANT_REPLY_1 = "First reply from assistant"


@patch("agents.online_agent.OnlineAgent.stream_model_turn")
def test_server_side_history_hydration_in_completion(stream_model_turn, online_agent, client):
    """
    Verify that when calling /agent/complete_text/{session_id} twice,
    the second call's payload includes the prior user/assistant turns.
    """
    # Use the fixture agent so the request payload is fully deterministic
    agent_cache[AgentType.GPT4AGENT] = online_agent

    captured_payloads = []

    def capture_messages(messages, _tools, _config):
        captured_payloads.append({"messages": [message.to_openai() for message in messages]})
        yield ModelEvent.text_delta(ASSISTANT_REPLY_1)
        yield ModelEvent.turn_complete(ModelTurn(text=ASSISTANT_REPLY_1, finish_reason="stop"))

    stream_model_turn.side_effect = capture_messages

    session_id = 4321

    # First turn
    r1 = client.post(
        f"/agent/complete_text/{session_id}",
        json={"prompt": "First message", "agent_type": "GPT4AGENT"},
    )
    assert r1.status_code == 200

    # Second turn
    r2 = client.post(
        f"/agent/complete_text/{session_id}",
        json={"prompt": "Second message", "agent_type": "GPT4AGENT"},
    )
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
    assert (
        len(second_messages) >= 3
    ), f"Expected at least 3 messages in second call, got {len(second_messages)}"
    assert second_messages[-3]["role"] == "user"
    assert second_messages[-3]["content"] == "First message"
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["content"] == ASSISTANT_REPLY_1
