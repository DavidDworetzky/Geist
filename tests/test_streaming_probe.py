from agents.architectures.chat_template_tools import provider_tool_name
from tests.streaming_probe import StreamingProbe


def test_reset_clears_observable_state_and_stale_producer_cannot_change_new_run():
    probe = StreamingProbe()
    probe.start()
    old = probe.segments([], [{"name": "tool"}])
    assert next(old) == "STREAM-FIRST"
    probe.reset()
    assert probe.state() == {
        "active": False,
        "stage": 0,
        "closed": False,
        "tools_seen": False,
        "released": [True, True],
        "search_calls": [],
        "tool_result_seen": False,
    }
    probe.start()
    assert list(old) == []
    assert probe.state() == {
        "active": True,
        "stage": 0,
        "closed": False,
        "tools_seen": False,
        "released": [False, False],
        "search_calls": [],
        "tool_result_seen": False,
    }


def test_xml_fixture_completes_goal_instead_of_repeating_search():
    probe = StreamingProbe()
    probe.start("xml_tool")
    probe.closed = True
    tools = [{"function": {"name": provider_tool_name("agent.goal.complete")}}]
    messages = [
        {"role": "tool", "tool_call_id": "observation-1", "content": "Fixture celebrity headline"},
        {"role": "user", "content": "Continue"},
    ]
    result = "".join(probe.segments(messages, tools))
    assert provider_tool_name("agent.goal.complete") in result
    assert "observation-1" in result
    assert "recent celebrity headlines" not in result
