import pytest

from agents.architectures.llama.ngram_draft import NgramDraft


def test_lookup_does_not_copy_the_current_suffix_from_itself():
    lookup = NgramDraft(list(range(20)))
    assert lookup.propose(32) == []


def test_lookup_extends_a_known_span_without_changing_it():
    lookup = NgramDraft(list(range(100)))
    lookup.extend([99, 98, *range(8)])
    assert lookup.propose(32) == list(range(8, 40))


def test_repeated_ngrams_keep_a_long_source_not_only_recent_matches():
    lookup = NgramDraft([0] * 100)
    assert lookup.propose(64, minimum=16) == [0] * 64
    assert all(len(positions) <= 4 for positions in lookup.positions.values())


def test_incremental_index_matches_single_extension():
    tokens = list(range(50)) + list(range(20))
    incremental = NgramDraft([])
    for token in tokens:
        incremental.extend([token])
    assert incremental.propose(20) == NgramDraft(tokens).propose(20)


def test_short_history_and_limits():
    lookup = NgramDraft([1, 2])
    assert lookup.propose(10) == []
    assert lookup.propose(0) == []
    with pytest.raises(ValueError):
        NgramDraft([], match_length=0)


def test_continuation_survives_pruned_repetitive_ngram_positions():
    repeated = list(range(40))
    source = [token for row in range(20) for token in [100 + row, *repeated]]
    lookup = NgramDraft(source)
    lookup.extend([100, *repeated[:8]])
    assert lookup.propose(40) == source[9:49]
    lookup.extend(source[9:49])
    assert lookup.propose(80) == source[49:129]
    # A changed generated token invalidates this alignment until a real suffix
    # match is available again; no unchecked source cursor may be emitted.
    lookup.extend([999] * 8)
    assert lookup.propose(80, minimum=16) == []
