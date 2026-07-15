"""
Route-level tests for the LocalAgent-backed completion and tick endpoints.

These cover the behavior formerly exercised through LlamaAgent, using a stub
runner (registered in conftest) so no model weights are loaded.
"""

from unittest.mock import patch

from agents.models.agent_completion import AgentCompletion
from app.main import AgentType, agent_cache, stream_chat_completion
from app.models.completion import CompleteTextParams
from tests.agents.test_online_agent_routes import (
    completions_generator,
    expected_world_context,
)


# by convention, if EXISTS is set for a key, existence is checked but not equivalence
agent_completion = {
    "message": ["Response to: write a haiku about the moon"],
    "id": "EXISTS",
    "chat_id": "EXISTS",
}


def test_legacy_stream_generates_completion_once():
    class LegacyAgent:
        def __init__(self):
            self.complete_calls = 0
            self.stream_calls = 0

        def complete_text(self, **_kwargs):
            self.complete_calls += 1
            return AgentCompletion(
                message=["legacy response"],
                id="legacy-completion",
                chat_id=14,
                run_id="legacy-run",
            )

        def stream_complete_text(self, **_kwargs):
            self.stream_calls += 1
            yield "duplicate response"

    agent = LegacyAgent()
    params = CompleteTextParams(prompt="legacy request", enable_tools=False)

    with patch("app.main.get_active_agent", return_value=agent):
        events = list(stream_chat_completion(params))

    assert agent.complete_calls == 1
    assert agent.stream_calls == 0
    assert any('event: delta\ndata: {"text": "legacy response"}' in event for event in events)
    assert any("event: final\n" in event for event in events)
    assert events[-1] == ('event: done\ndata: {"run_id": "legacy-run", "chat_id": 14}\n\n')


@patch("adapters.log_adapter.LogAdapter.log", autospec=True)
def test_completion(log, local_agent, client):
    log.side_effect = lambda self, output: print(output)
    agent_cache[AgentType.LLAMA] = local_agent
    try:
        payload = {
            "prompt": "write a haiku about the moon",
            "max_tokens": 1024,
            "n": 1,
            "stop": ["string"],
            "temperature": 1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "echo": False,
            "best_of": 0,
            "prompt_tokens": [0],
            "response_format": "text",
            "agent_type": "LLAMA",
        }

        response = client.post("agent/complete_text", json=payload)

        assert response.status_code == 200
        response_payload = response.json()
        for key in agent_completion:
            if agent_completion[key] == "EXISTS":
                assert response_payload[key] is not None
            else:
                assert response_payload[key] == agent_completion[key]
    finally:
        agent_cache[AgentType.LLAMA] = None


@patch("agents.local_agent.LocalAgent.complete_text")
@patch("adapters.log_adapter.LogAdapter.log", autospec=True)
def test_local_tick_with_prompt(log, complete_text, local_agent, client):
    complete_text.side_effect = lambda prompt: completions_generator(prompt=prompt)
    log.side_effect = lambda self, output: print(output)
    agent_cache[AgentType.LLAMA] = local_agent
    try:
        payload = {"prompt": "Write a haiku", "agent_type": "LLAMA"}

        response = client.post("agent/initialize_task_and_tick", json=payload)

        assert response.status_code == 200
        assert response.json() == {
            "world_context": expected_world_context("Write a haiku"),
            "task_context": [],
            "execution_context": [],
        }
        assert log.call_count == 2
    finally:
        agent_cache[AgentType.LLAMA] = None
