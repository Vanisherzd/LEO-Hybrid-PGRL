"""Experiment-config loader + validation for the conducted-hardware runner.

Modes:
  dryrun     : no hardware; mock TX + mock RX. Can self-test PER path.
  iq_only    : conducted IQ capture replay; requires iq.metadata_path.
               Reports PER UNAVAILABLE (signal detection only).
  decoded_rx : conducted decoded RX log; requires rx.rx_log. Computes PER from
               decoded payloads.

Hardware modes (iq_only, decoded_rx) require a conducted/shielded safety
acknowledgement in the config and the runner's --armed flag.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation import _yaml  # noqa: E402

VALID_MODES = ("dryrun", "iq_only", "decoded_rx")
HARDWARE_MODES = ("iq_only", "decoded_rx")


class ConfigError(ValueError):
    pass


@dataclass
class ExperimentConfig:
    run_id: str
    mode: str
    n_packets: int = 100
    seed: int = 42
    tx: Dict[str, Any] = field(default_factory=dict)
    rx: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    mock: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_hardware_mode(self) -> bool:
        return self.mode in HARDWARE_MODES

    def validate(self) -> None:
        if not self.run_id:
            raise ConfigError("run_id is required")
        if self.mode not in VALID_MODES:
            raise ConfigError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if not isinstance(self.n_packets, int) or self.n_packets <= 0:
            raise ConfigError("n_packets must be a positive int")

        if self.mode == "dryrun":
            tb = self.tx.get("backend", "mock")
            rb = self.rx.get("backend", "mock")
            if tb != "mock" or rb != "mock":
                raise ConfigError("dryrun requires tx.backend=mock and rx.backend=mock")
            return

        # hardware modes
        ack = bool(self.safety.get("conducted_or_shielded"))
        if not ack:
            raise ConfigError(
                f"mode {self.mode} requires safety.conducted_or_shielded: true "
                "(conducted/shielded setup only; OTA is not allowed)")
        if self.mode == "iq_only":
            if not self.rx.get("iq_metadata_path"):
                raise ConfigError("iq_only requires rx.iq_metadata_path")
        elif self.mode == "decoded_rx":
            if not self.rx.get("rx_log_path"):
                raise ConfigError("decoded_rx requires rx.rx_log_path")


def load_config(path: str) -> ExperimentConfig:
    if not os.path.exists(path):
        raise ConfigError(f"config not found: {path}")
    with open(path) as fh:
        data = _yaml.loads(fh.read()) or {}
    cfg = ExperimentConfig(
        run_id=data.get("run_id") or "",
        mode=data.get("mode") or "",
        n_packets=int(data.get("n_packets", 100)),
        seed=int(data.get("seed", 42)),
        tx=data.get("tx") or {},
        rx=data.get("rx") or {},
        safety=data.get("safety") or {},
        mock=data.get("mock") or {},
        raw=data,
    )
    cfg.validate()
    return cfg
