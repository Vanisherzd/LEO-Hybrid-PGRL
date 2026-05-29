"""Dataclass schema for PGRL predictor output — communication-oriented metrics."""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PGRLOutput:
    """PGRL predictor output mapped to uplink-control quantities.

    All timing values in seconds; Doppler in Hz or Hz/s.
    Used to drive adaptive guard-band, Doppler pre-comp, and TX timing selection.
    """

    # Pass timing relative to predicted pass start [s]
    pass_start_s: float
    pass_peak_s: float
    pass_end_s: float

    # Doppler prediction
    doppler_hz: float          # Doppler shift at pass peak [Hz]
    doppler_rate_hz_s: float   # First derivative d|f_D|/dt at peak [Hz/s]

    # 1-sigma uncertainty estimates — drive adaptive guard band
    timing_sigma_s: float      # Timing uncertainty [s]
    doppler_sigma_hz: float    # Doppler uncertainty [Hz]

    # Link quality
    link_score: float         # 0–1, predicted P(success) for this TX window

    def recommended_guard_time_s(
        self,
        base_guard_s: float = 0.5,
        k: float = 3.0,
        max_guard_s: float = 30.0,
    ) -> float:
        """Default guard-band using k-sigma timing uncertainty."""
        guard = base_guard_s + k * self.timing_sigma_s
        return min(guard, max_guard_s)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def pass_duration_s(self) -> float:
        return self.pass_end_s - self.pass_start_s

    def __repr__(self) -> str:
        return (
            f"PGRLOutput(timing_sigma={self.timing_sigma_s:.4f}s, "
            f"doppler={self.doppler_hz:.1f}Hz, link={self.link_score:.2f})"
        )