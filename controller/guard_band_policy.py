"""Adaptive guard-band policy driven by PGRL timing uncertainty."""
from __future__ import annotations

import math

import numpy as np


def adaptive_guard_time(
    base_guard_s: float,
    timing_sigma_s: float,
    k: float = 3.0,
    max_guard_s: float = 30.0,
) -> float:
    """Compute adaptive guard time using k-sigma timing uncertainty.

    G = G_0 + k · σ_t

    Args:
        base_guard_s:    Fixed protocol guard [s]
        timing_sigma_s:  1-sigma PGRL timing uncertainty [s]
        k:               Coverage factor (3-sigma ≈ 99.7 %)
        max_guard_s:     Hard ceiling

    Returns:
        Adaptive guard time [s]
    """
    return float(min(base_guard_s + k * timing_sigma_s, max_guard_s))


def guard_overhead_fraction(adaptive_guard_s: float, pass_duration_s: float) -> float:
    """Guard overhead as fraction of total pass opportunity time."""
    if pass_duration_s <= 0:
        return 0.0
    return float(adaptive_guard_s / pass_duration_s)


def missed_opportunity_probability(timing_sigma_s: float, guard_s: float) -> float:
    """Approximate P(missed opportunity) = P(|δ| > guard) for δ ~ N(0, σ²).

    = 2 · Q(guard / σ) ≈ erfc(guard / (σ√2))
    """
    if guard_s <= 0 or timing_sigma_s <= 0:
        return 1.0
    x = guard_s / timing_sigma_s
    return float(1.0 - math.erf(x / math.sqrt(2.0)))