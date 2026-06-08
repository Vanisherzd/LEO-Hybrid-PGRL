"""Typed records for packet-delivery validation.

All records are plain dataclasses with ``to_dict`` for CSV/JSON serialization.
No hardware or numpy dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

REQUIRED_SUMMARY_FIELDS = [
    "run_id", "n_tx", "n_rx", "n_crc_ok", "n_payload_match",
    "packet_delivery_ratio", "packet_error_rate", "crc_error_rate",
    "duplicate_rx_count", "missing_seq_count", "false_positive_count",
    "median_latency_ms", "notes",
]


@dataclass
class TxPacketRecord:
    run_id: str
    seq: int
    payload_hex: str
    payload_len: int
    tx_timestamp_utc: Optional[str] = None
    tx_backend: str = "mock"
    tx_power_dbm: Optional[float] = None
    center_frequency_hz: Optional[float] = None
    lr_fhss_config: Optional[str] = None
    uart_status: str = "unknown"
    raw_log_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RxPacketRecord:
    run_id: str
    seq: Optional[int]
    payload_hex: Optional[str]
    payload_len: Optional[int]
    rx_timestamp_utc: Optional[str] = None
    rx_backend: str = "mock"
    crc_ok: Optional[bool] = None
    rssi_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    cfo_hz: Optional[float] = None
    # decode_status in {"decoded", "crc_fail", "not_decoded"}
    decode_status: str = "not_decoded"
    raw_log_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IqCaptureRecord:
    run_id: str
    capture_id: str
    center_frequency_hz: Optional[float] = None
    sample_rate_sps: Optional[float] = None
    gain_db: Optional[float] = None
    duration_s: Optional[float] = None
    on_off_delta_db: Optional[float] = None
    candidate_score: Optional[float] = None
    validation_status: str = "unknown"
    raw_iq_path: Optional[str] = None
    metadata_path: Optional[str] = None
    sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PacketValidationSummary:
    run_id: str
    n_tx: int = 0
    n_rx: int = 0
    n_crc_ok: int = 0
    n_payload_match: int = 0
    packet_delivery_ratio: Optional[float] = None
    packet_error_rate: Optional[float] = None
    crc_error_rate: Optional[float] = None
    duplicate_rx_count: int = 0
    missing_seq_count: int = 0
    false_positive_count: int = 0
    median_latency_ms: Optional[float] = None
    notes: str = ""
    # provenance / mode flags (not part of the required metric set)
    decoding_available: bool = False
    tx_backend: Optional[str] = None
    rx_backend: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
