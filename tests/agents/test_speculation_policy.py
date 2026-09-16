from agents.architectures.llama.speculation_policy import SpeculationPolicy


def test_policy_requires_native_calibration_and_eight_rounds():
    policy = SpeculationPolicy()
    for _ in range(4):
        assert policy.calibrating
        policy.observe_native(0.05)
    assert not policy.calibrating
    assert policy.native_tps == 20
    for _ in range(7):
        policy.observe_round(2, 0.18)
        assert not policy.fallback
    policy.observe_round(2, 0.18)
    assert policy.fallback


def test_policy_keeps_fast_speculation_and_later_fallback_is_sticky():
    policy = SpeculationPolicy()
    for seconds in [0.05, 0.05, 0.05, 0.2]:
        policy.observe_native(seconds)
    for _ in range(8):
        policy.observe_round(6, 0.18)
    assert not policy.fallback
    for _ in range(4):
        policy.observe_round(2, 0.18)
    assert not policy.fallback
    for _ in range(100):
        policy.observe_round(2, 0.18)
    assert policy.fallback
    for _ in range(4):
        policy.observe_round(8, 0.01)
    assert policy.fallback
