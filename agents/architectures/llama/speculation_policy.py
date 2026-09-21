"""Per-request break-even check; all calibration tokens are real output."""

import statistics


class SpeculationPolicy:
    version = "cumulative-eight-v2"

    def __init__(self) -> None:
        self.native_seconds: list[float] = []
        self.round_count = 0
        self.emitted = 0
        self.speculative_seconds = 0.0
        self.fallback = False

    @property
    def calibrating(self) -> bool:
        return len(self.native_seconds) < 4

    @property
    def native_tps(self) -> float:
        return 1 / statistics.median(self.native_seconds) if self.native_seconds else 0.0

    def observe_native(self, seconds: float) -> None:
        if self.calibrating:
            self.native_seconds.append(max(seconds, 1e-9))

    def observe_round(self, emitted: int, seconds: float) -> None:
        self.round_count += 1
        self.emitted += emitted
        self.speculative_seconds += max(seconds, 1e-9)
        if self.calibrating or self.round_count < 8:
            return
        speculative_tps = self.emitted / self.speculative_seconds
        # Use cumulative evidence and a wide deadband: a short difficult span
        # must not discard a large advantage earned earlier in the request.
        self.fallback |= speculative_tps < self.native_tps * 0.85
