"""Simple energy model for D2S IoT uplink energy accounting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyBudget:
    """Energy accounting for a sequence of TX opportunities."""
    total_energy_j: float = 0.0
    n_opportunities: int = 0
    n_successful: int = 0

    @property
    def energy_per_opportunity_j(self) -> float:
        if self.n_opportunities <= 0:
            return float("inf")
        return self.total_energy_j / self.n_opportunities

    @property
    def energy_per_success_j(self) -> float:
        if self.n_successful <= 0:
            return float("inf")
        return self.total_energy_j / self.n_successful

    @property
    def success_rate(self) -> float:
        if self.n_opportunities <= 0:
            return 0.0
        return self.n_successful / self.n_opportunities


def receiver_on_energy(current_ma: float, voltage_v: float, on_time_s: float) -> float:
    """Receiver ON energy [J] = I [A] × V [V] × t [s]."""
    return (current_ma / 1000.0) * voltage_v * on_time_s


def tx_energy(power_dbm: float, tx_duration_s: float) -> float:
    """TX energy [J] = 10^{(P_dBm−30)/10} × t [s]."""
    power_w = 10 ** ((power_dbm - 30.0) / 10.0)
    return power_w * tx_duration_s


def total_opportunity_energy(
    guard_s: float,
    rx_on_s: float,
    tx_s: float,
    rx_current_ma: float = 10.0,
    tx_power_dbm: float = 14.0,
    voltage_v: float = 3.3,
) -> dict[str, float]:
    """Energy breakdown for one TX opportunity [J]."""
    guard_j = receiver_on_energy(rx_current_ma, voltage_v, guard_s)
    rx_j    = receiver_on_energy(rx_current_ma, voltage_v, rx_on_s)
    tx_j    = tx_energy(tx_power_dbm, tx_s)
    return {"guard_j": guard_j, "rx_j": rx_j, "tx_j": tx_j, "total_j": guard_j + rx_j + tx_j}