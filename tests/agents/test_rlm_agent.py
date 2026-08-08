"""Unit tests for the minimal RLM agent loop, using a scripted fake agent."""

from adapters.adapter_registry import AdapterWrapper
from adapters.base_adapter import BaseAdapter
from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from agents.base_agent import BaseAgent
from agents.rlm_agent import OUTPUT_CHAR_LIMIT, RlmAgent


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


def make_context(adapter=None):
    settings = AgentSettings(name="test", version="1.0", description="test", max_tokens=64)
    context = AgentContext(
        settings=settings,
        world_context=[],
        task_context=[],
        execution_context=[],
        function_log=[],
        execution_classes=[("CalculatorAdapter", ["add"])],
    )
    if adapter is not None:
        context.initialized_classes = [AdapterWrapper(name="CalculatorAdapter", instance=adapter)]
    context._tool_schemas = None
    context._tool_dispatcher = None
    return context


class ScriptedAgent(BaseAgent):
    """Returns canned completions in order and records every prompt it saw."""

    def __init__(self, responses, adapter=None):
        super().__init__(make_context(adapter))
        self.responses = list(responses)
        self.prompts = []

    def complete_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}

    def stream_complete_text(self, prompt, **kwargs):
        raise NotImplementedError

    def complete_audio(self, audio_file, **kwargs):
        raise NotImplementedError

    def connect_realtime_audio(self):
        raise NotImplementedError


def test_plain_text_reply_is_the_final_answer():
    agent = ScriptedAgent(["Paris is the capital of France."])
    result = RlmAgent(agent).run("What is the capital of France?")
    assert result.answer == "Paris is the capital of France."
    assert result.iterations == 1
    assert result.transcript == []


def test_state_persists_across_iterations_and_final_ends_the_session():
    agent = ScriptedAgent(
        [
            "```python\nvalue = sum(range(10))\nprint(value)\n```",
            '```python\nFINAL(f"total={value}")\n```',
        ]
    )
    result = RlmAgent(agent).run("Sum the numbers below ten.")
    assert result.answer == "total=45"
    assert result.iterations == 2
    assert "45" in agent.prompts[1]


def test_context_is_a_repl_variable():
    haystack = ("filler\n" * 500) + "needle=12345\n" + ("filler\n" * 500)
    agent = ScriptedAgent(
        [
            "```python\nline = [l for l in context.splitlines() if 'needle' in l][0]\nprint(line)\n```",
            "```python\nFINAL(line.split('=')[1])\n```",
        ]
    )
    result = RlmAgent(agent).run("Find the needle value.", context=haystack)
    assert result.answer == "12345"
    assert "needle=12345" in agent.prompts[1]
    # The raw context is never inlined into the prompt, only its size.
    assert haystack not in agent.prompts[0]
    assert str(len(haystack)) in agent.prompts[0]


def test_execution_errors_are_fed_back_not_raised():
    agent = ScriptedAgent(
        [
            "```python\nprint(undefined_variable)\n```",
            "```python\nFINAL('recovered')\n```",
        ]
    )
    result = RlmAgent(agent).run("Trip over an error, then recover.")
    assert result.answer == "recovered"
    assert "NameError" in agent.prompts[1]


def test_rlm_query_runs_a_child_session():
    agent = ScriptedAgent(
        [
            "```python\nresult = rlm_query('Summarize the sub context.', context='sub context')\nprint(result)\n```",
            "```python\nFINAL('child-answer')\n```",
            "```python\nFINAL('root saw: ' + result)\n```",
        ]
    )
    result = RlmAgent(agent).run("Delegate a subtask.")
    assert result.answer == "root saw: child-answer"
    # The child ran with its own prompt rendering, against the same model.
    assert "Summarize the sub context." in agent.prompts[1]


def test_rlm_query_degrades_to_llm_query_at_max_depth():
    agent = ScriptedAgent(
        [
            "```python\nresult = rlm_query('Sub question?', context='sub context')\nprint(result)\n```",
            "one-shot answer",
            "```python\nFINAL(result)\n```",
        ]
    )
    result = RlmAgent(agent, max_depth=1).run("Delegate past the depth limit.")
    assert result.answer == "one-shot answer"
    # The fallback inlines the child context into a single flat completion.
    fallback_prompt = agent.prompts[1]
    assert "Sub question?" in fallback_prompt
    assert "sub context" in fallback_prompt


def test_call_tool_dispatches_through_adapters():
    adapter = CalculatorAdapter()
    agent = ScriptedAgent(
        [
            "```python\nprint(call_tool('CalculatorAdapter__add', a=2, b=3))\n```",
            "```python\nFINAL('done')\n```",
        ],
        adapter=adapter,
    )
    result = RlmAgent(agent).run("Add two and three.")
    assert result.answer == "done"
    assert adapter.calls == [(2, 3)]
    assert "5" in agent.prompts[1]


def test_list_tools_reflects_adapter_schemas():
    adapter = CalculatorAdapter()
    agent = ScriptedAgent(
        [
            "```python\nprint([t['name'] for t in list_tools()])\n```",
            "```python\nFINAL('done')\n```",
        ],
        adapter=adapter,
    )
    RlmAgent(agent).run("What tools do you have?")
    assert "CalculatorAdapter__add" in agent.prompts[1]


def test_long_output_is_truncated_before_feedback():
    agent = ScriptedAgent(
        [
            "```python\nprint('x' * 10000)\n```",
            "```python\nFINAL('done')\n```",
        ]
    )
    RlmAgent(agent).run("Print something huge.")
    feedback = agent.prompts[1]
    assert "chars truncated" in feedback
    assert len(feedback) < 10000 + len(agent.prompts[0]) - OUTPUT_CHAR_LIMIT


def test_iteration_budget_exhaustion_returns_none():
    agent = ScriptedAgent(["```python\nprint('still going')\n```"] * 3)
    result = RlmAgent(agent, max_iterations=3).run("Never finish.")
    assert result.answer is None
    assert result.iterations == 3
    assert len(result.transcript) == 3
