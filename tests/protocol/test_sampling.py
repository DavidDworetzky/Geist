from protocol.sampling import sample_indices


def test_sampling_is_deterministic_for_same_inputs():
    first = sample_indices("nonce", "block", population=100, k=10)
    second = sample_indices("nonce", "block", population=100, k=10)
    assert first == second


def test_sampling_changes_with_beacon_nonce():
    assert sample_indices("nonce-a", "block", 100, 10) != sample_indices("nonce-b", "block", 100, 10)


def test_sampling_changes_with_block_id():
    assert sample_indices("nonce", "block-a", 100, 10) != sample_indices("nonce", "block-b", 100, 10)


def test_sampled_indices_are_distinct_and_in_range():
    indices = sample_indices("nonce", "block", population=17, k=17)
    assert sorted(indices) == list(range(17))


def test_sample_size_clamped_to_population():
    assert len(sample_indices("nonce", "block", population=3, k=10)) == 3
