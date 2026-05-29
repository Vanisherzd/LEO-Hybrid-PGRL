"""PGRL-to-LR-FHSS uplink controller — communication-oriented adaptation layer."""
from .pgrl_output_schema import PGRLOutput
from .guard_band_policy import adaptive_guard_time
from .doppler_precomp import compensated_tx_frequency
from .tx_timing_policy import select_tx_time
from .energy_model import (
    EnergyBudget,
    receiver_on_energy,
    tx_energy,
    total_opportunity_energy,
)

__all__ = [
    "PGRLOutput",
    "adaptive_guard_time",
    "compensated_tx_frequency",
    "select_tx_time",
    "EnergyBudget",
    "receiver_on_energy",
    "tx_energy",
    "total_opportunity_energy",
]