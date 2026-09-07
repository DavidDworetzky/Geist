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
