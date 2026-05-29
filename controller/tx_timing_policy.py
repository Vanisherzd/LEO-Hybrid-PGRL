"""Transmission timing selection based on PGRL link scores and cost terms."""
from __future__ import annotations

import numpy as np
from typing import Optional

from .doppler_precomp import compensated_tx_frequency


def select_tx_time(
    candidate_times_s: np.ndarray,
    link_scores: np.ndarray,
    doppler_rates_hz_s: np.ndarray,
    idle_costs: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> float:
    """Select optimal TX time from candidate windows.

    t* = argmax_t [ L(t) − α·|ḟ_D(t)| − β·E_idle(t) ]

    Args:
        candidate_times_s:  Candidate TX start times [s]
        link_scores:        Link-success probabilities (0–1) per candidate
        doppler_rates_hz_s: Doppler rate magnitude per candidate [Hz/s]
        idle_costs:         Idle energy cost per candidate [J]
        alpha:              Doppler rate weight
        beta:               Idle-cost weight

    Returns:
        Best TX start time [s]
    """
    candidate_times_s = np.asarray(candidate_times_s, dtype=float)
    link_scores        = np.asarray(link_scores, dtype=float)
    doppler_rates      = np.asarray(doppler_rates_hz_s, dtype=float)
    idle_costs         = np.asarray(idle_costs, dtype=float) if idle_costs is not None else np.zeros_like(candidate_times_s)

    utility = link_scores - alpha * np.abs(doppler_rates) - beta * idle_costs
    return float(candidate_times_s[np.argmax(utility)])