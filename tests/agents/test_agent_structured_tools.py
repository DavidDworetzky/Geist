"""Integration tests for structured tool calling in OnlineAgent and LocalAgent."""
import json

import pytest

from adapters.adapter_registry import AdapterWrapper
from adapters.base_adapter import BaseAdapter
from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from agents.architectures import register_runner
from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.exceptions import FunctionCallError
from agents.local_agent import LocalAgent
from agents.models.llama_completion import strings_to_message_dict
from agents.online_agent import OnlineAgent


class CalculatorAdapter(BaseAdapter):
    """Test adapter recording its invocations."""

    def __init__(self):
        self.calls = []

    def enumerate_actions(self):
        return ["add"]

    def add(self, a: int, b: int) -> int:
        """Add two integers."""
        self.calls.append((a, b))
        return a + b


def make_context(adapter):
    settings = AgentSettings(name="test", version="1.0", description="test", max_tokens=64)
    context = AgentContext(
        settings=settings,
        world_context=[],
        task_context=[],
        execution_context=[],
        function_log=[],
        execution_classes=[("CalculatorAdapter", ["add"])],
    )
    context.initialized_classes = [AdapterWrapper(name="CalculatorAdapter", instance=adapter)]
    context._tool_schemas = None
    context._tool_dispatcher = None
    return context


def make_online_agent(adapter):
    context = make_context(adapter)
    return OnlineAgent(context, base_url="https://example.test/v1", model="test-model", api_key="key")


class TestOnlineAgentNativeTools:
    def test_native_tool_loop(self, monkeypatch):
        adapter = CalculatorAdapter()
        agent = make_online_agent(adapter)
        payloads = []

        responses = iter([
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "CalculatorAdapter__add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }],
                    }
                }]
            },
            {"choices": [{"message": {"role": "assistant", "content": "The sum is 5."}}]},
        ])

        def fake_request(payload, **kwargs):
            payloads.append(payload)
            return next(responses)

        monkeypatch.setattr(agent, "_make_request", fake_request)

        completion = agent.complete_with_tools("What is 2 + 3?")

        assert completion.content == "The sum is 5."
        assert completion.used_native_tools
        assert adapter.calls == [(2, 3)]
        assert len(completion.tool_results) == 1
        assert completion.tool_results[0].success
        assert completion.tool_results[0].result == 5

        # The provider saw the reflected schema in the tools payload
        first_tools = payloads[0]["tools"]
        assert first_tools[0]["function"]["name"] == "CalculatorAdapter__add"
        assert first_tools[0]["function"]["parameters"]["required"] == ["a", "b"]

        # The tool result was fed back as a tool message with the call id
        second_messages = payloads[1]["messages"]
        tool_messages = [message for message in second_messages if message.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_1"
        assert json.loads(tool_messages[0]["content"]) == {"success": True, "result": 5}

        # Dispatch was journaled to the agent's function log
        log_entry = json.loads(agent._agent_context.function_log[0])
        assert log_entry["function"] == "add"
        assert log_entry["success"] is True

    def test_malformed_native_tool_call_is_captured(self, monkeypatch):
        adapter = CalculatorAdapter()
        agent = make_online_agent(adapter)

        responses = iter([
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "not_namespaced", "arguments": "{}"},
                        }],
                    }
                }]
            },
            {"choices": [{"message": {"role": "assistant", "content": "giving up"}}]},
        ])
        monkeypatch.setattr(agent, "_make_request", lambda payload, **kwargs: next(responses))

        completion = agent.complete_with_tools("What is 2 + 3?")

        assert completion.content == "giving up"
        assert not completion.tool_results[0].success
        assert adapter.calls == []

    def test_falls_back_to_prompt_tools_when_native_rejected(self, monkeypatch):
        adapter = CalculatorAdapter()
        agent = make_online_agent(adapter)

        def fake_request(payload, **kwargs):
            if "tools" in payload:
                raise Exception("tools parameter not supported")
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": '{"class": "CalculatorAdapter", "function": "add", "parameters": {"a": 4, "b": 6}}',
                    }
                }]
            }

        monkeypatch.setattr(agent, "_make_request", fake_request)

        completion = agent.complete_with_tools("What is 4 + 6?")

        assert not completion.used_native_tools
        assert adapter.calls == [(4, 6)]
        assert completion.tool_results[0].success
        assert completion.tool_results[0].result == 10

    def test_no_tools_degrades_to_plain_completion(self, monkeypatch):
        adapter = CalculatorAdapter()
        agent = make_online_agent(adapter)
        agent._agent_context.initialized_classes = []
        agent._agent_context._tool_schemas = None
        agent._agent_context._tool_dispatcher = None

        monkeypatch.setattr(
            agent,
            "_make_request",
            lambda payload, **kwargs: {
                "choices": [{"message": {"role": "assistant", "content": "plain answer"}}]
            },
        )

        completion = agent.complete_with_tools("Just answer")
        assert completion.content == "plain answer"
        assert completion.tool_results == []


class ScriptedRunner(BaseRunner):
    """Runner returning a scripted sequence of completions."""

    script = []

    def __init__(self):
        self.prompts = []

    def load(self, model_id, device_config=None):
        pass

    def generate(self, prompt, generation_config: GenerationConfig):
        return {}

    def complete(self, system_prompt, user_prompt, generation_config: GenerationConfig):
        self.prompts.append(user_prompt)
        response = self.script[min(len(self.prompts) - 1, len(self.script) - 1)]
        return strings_to_message_dict(user_prompt, response)


register_runner("scripted_test_runner", ScriptedRunner)


class TestLocalAgentPromptTools:
    def make_agent(self, adapter, script):
        ScriptedRunner.script = script
        context = make_context(adapter)
        return LocalAgent(
            context,
            model_id="test-model",
            runner_type="scripted_test_runner",
        )

    def test_schema_grounded_prompt_with_retry(self):
        adapter = CalculatorAdapter()
        agent = self.make_agent(adapter, [
            "Let me think about which tool to use...",
            'Here you go: {"class": "CalculatorAdapter", "function": "add", "parameters": {"a": "7", "b": 8}}',
        ])

        completion = agent.complete_with_tools("What is 7 + 8?")

        assert completion.tool_results[0].success
        # string "7" was coerced to the declared integer type
        assert adapter.calls == [(7, 8)]
        assert completion.tool_results[0].result == 15

        runner = agent.runner
        assert len(runner.prompts) == 2
        # first prompt exposes the reflected schemas (function visibility)
        assert "AVAILABLE_TOOLS" in runner.prompts[0]
        assert "CalculatorAdapter" in runner.prompts[0]
        # retry prompt carries the validation feedback
        assert "previous response was invalid" in runner.prompts[1]

    def test_gives_up_and_reports_after_max_attempts(self):
        adapter = CalculatorAdapter()
        agent = self.make_agent(adapter, ["never json"])

        completion = agent.complete_with_tools("What is 7 + 8?", max_tool_iterations=2)

        assert not completion.tool_results[0].success
        assert "after 2 attempts" in completion.tool_results[0].error
        assert adapter.calls == []

    def test_legacy_helpers_use_robust_pipeline(self):
        adapter = CalculatorAdapter()
        agent = self.make_agent(adapter, ["unused"])

        wrapped = 'prose before ```json\n{"class": "CalculatorAdapter", "function": "add", "parameters": {"a": 1, "b": 2}}\n``` prose after'
        assert agent._is_valid_function_json(wrapped)
        assert agent._take_json_and_call_function(wrapped) == 3

        assert not agent._is_valid_function_json("not a call")
        with pytest.raises(FunctionCallError):
            agent._take_json_and_call_function("not a call")
