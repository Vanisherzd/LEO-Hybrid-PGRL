"""
Base Controller — Abstract Interface for PGRL Training
========================================================

Defines the contract that all controller implementations must satisfy:
  - reset()          : Initialize/reset environment and agent state
  - step(action)     : Execute action, return (obs, reward, done, info)
  - update_policy()  : Trigger policy policy gradient update
  - get_doppler()    : Return predicted Δf for pre-compensation
  - get_timing()     : Return predicted Δt for slot synchronization

Designed for compatibility with both simulation-only and HWIL (USRP B210)
validation workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ControllerState:
    """Snapshot of controller state for logging and replay."""

    step_idx: int
    timestamp_s: float
    doppler_hz: float
    timing_error_ms: float
    reward: float
    collision: bool
    slot_utilization: float
    energy_j: float
    info: Dict[str, Any] = field(default_factory=dict)


class BaseController(ABC):
    """
    Abstract base class for all PGRL controllers.

    Subclasses must implement the core RL loop: reset, step, update_policy.
    Supports both online policy and offline policy training modes.
    """

    def __init__(
        self,
        num_slots: int = 64,
        slot_duration_ms: float = 49.0,
        carrier_freq_hz: float = 436.5e6,
        device_id: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            num_slots:       Number of control slots per superframe
            slot_duration_ms: Duration of each LR-FHSS hop/slot in ms
            carrier_freq_hz:  Carrier frequency in Hz (S-band default)
            device_id:        USRP device identifier for HWIL (None = simulation)
            seed:             Random seed for reproducibility
        """
        self.num_slots = num_slots
        self.slot_duration_ms = slot_duration_ms
        self.carrier_freq_hz = carrier_freq_hz
        self.device_id = device_id
        self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self._current_state: Optional[ControllerState] = None

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the controller and environment for a new episode.

        Returns:
            obs: Initial observation dict with keys 'doppler_hz', 'timing_s',
                 'orbital_elements', 'slot_history'
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one RL step.

        Args:
            action: Dict with keys:
              - 'guard_us':     guard-band duration in microseconds
              - 'freq_bin':     assigned FH frequency bin index [0, num_slots)
              - 'tx_power_dbm': transmit power in dBm

        Returns:
            obs:       Next observation
            reward:    Scalar reward for this step
            done:      Episode termination flag
            info:      Auxiliary diagnostics dict
        """
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, **kwargs) -> Dict[str, float]:
        """
        Perform one policy policy gradient update.

        Returns:
            metrics: Dict of training metrics (policy_loss, value_loss, entropy, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def get_doppler_prediction(self, t_future_s: float) -> float:
        """
        Return predicted Doppler shift at t_future_s from current epoch.

        Args:
            t_future_s: Time offset in seconds from now

        Returns:
            doppler_hz: Predicted Doppler frequency in Hz
        """
        raise NotImplementedError

    @abstractmethod
    def get_timing_prediction(self, t_future_s: float) -> float:
        """
        Return predicted timing error at t_future_s.

        Args:
            t_future_s: Time offset in seconds from now

        Returns:
            timing_error_ms: Predicted timing error in milliseconds
        """
        raise NotImplementedError

    def get_state(self) -> Optional[ControllerState]:
        """Return the most recent controller state snapshot."""
        return self._current_state

    def save_checkpoint(self, path: str) -> None:
        """Persist controller checkpoint to disk."""
        raise NotImplementedError(f"Subclass {type(self).__name__} must implement save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Restore controller from checkpoint."""
        raise NotImplementedError(f"Subclass {type(self).__name__} must implement load_checkpoint")