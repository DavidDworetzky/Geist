"""Bounded-history lookup proposals; the target must verify every proposed token."""

from collections import deque


class NgramDraft:
    def __init__(self, tokens: list[int], match_length: int = 8):
        if match_length < 1:
            raise ValueError("Match length must be positive")
        self.match_length = match_length
        self.tokens: list[int] = []
        self.positions: dict[tuple[int, ...], deque[int]] = {}
        self.indexed_end = match_length - 1
        self.last_source: tuple[int, int] | None = None
        self.extend(tokens)

    def extend(self, tokens: list[int]) -> None:
        self.tokens.extend(tokens)
        # Only index spans with at least one known continuation token. Never
        # index the current suffix as its own source.
        for end in range(self.indexed_end + 1, len(self.tokens)):
            key = tuple(self.tokens[end - self.match_length : end])
            positions = self.positions.setdefault(key, deque())
            if len(positions) == 4:
                del positions[1]
            positions.append(end)
            self.indexed_end = end

    def propose(self, limit: int, minimum: int = 1) -> list[int]:
        if limit <= 0 or len(self.tokens) < self.match_length:
            return []
        key = tuple(self.tokens[-self.match_length :])
        candidates = list(reversed(self.positions.get(key, ())))
        continuation = None
        if self.last_source is not None:
            history_length, source_start = self.last_source
            continuation = source_start + len(self.tokens) - history_length
            if self.match_length <= continuation < len(self.tokens):
                candidates.insert(0, continuation)
        best = []
        best_score = (0, 0, 0)
        best_end = None
        for end in candidates:
            if tuple(self.tokens[end - self.match_length : end]) != key:
                continue
            matched = self.match_length
            while (
                matched < min(32, end)
                and self.tokens[end - matched - 1] == self.tokens[-matched - 1]
            ):
                matched += 1
            proposal = self.tokens[end : end + limit]
            if len(proposal) < minimum:
                continue
            score = (matched, int(end == continuation), len(proposal))
            if score > best_score:
                best, best_score, best_end = proposal, score, end
        if best_end is not None:
            self.last_source = (len(self.tokens), best_end)
        return best
