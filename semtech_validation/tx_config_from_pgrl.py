"""Convert PGRLOutput into a Semtech LR1121 / LR11xx TX configuration JSON."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running standalone
FILE_DIR = Path(__file__).parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from controller.pgrl_output_schema import PGRLOutput
from controller.doppler_precomp import compensated_tx_frequency


def make_semtech_tx_config(
    pgrl: PGRLOutput,
    base_freq_hz: float,
    tx_power_dbm: int = 14,
    payload_len: int = 16,
    sf: int = 12,
    coding_rate: int = 4,
    precomp_ratio: float = 1.0,
    num_hops: int = 84,
    hop_length_ms: float = 6.0,
) -> dict:
    """Generate a Semtech LR1121 TX configuration from PGRL predictor output.

    Args:
        pgrl:             PGRLOutput instance
        base_freq_hz:     Nominal carrier frequency [Hz]
        tx_power_dbm:     TX output power [dBm]
        payload_len:      Payload length [bytes]
        sf:               Spreading factor (LR-FHSS: typically 12)
        coding_rate:      Coding rate denominator (4 = 4/5)
        precomp_ratio:    Doppler pre-compensation fraction (1.0 = full)
        num_hops:         Number of frequency hops per LR-FHSS packet
        hop_length_ms:    Duration per hop [ms]

    Returns:
        Dict suitable for serialising to JSON and sending to LR1121 driver
    """
    tx_freq = compensated_tx_frequency(base_freq_hz, pgrl.doppler_hz, precomp_ratio)
    guard_s = pgrl.recommended_guard_time_s()

    return {
        "center_frequency_hz": int(round(tx_freq)),
        "tx_power_dbm": tx_power_dbm,
        "payload_len": payload_len,
        "spreading_factor": sf,
        "coding_rate": coding_rate,
        "guard_time_s": round(guard_s, 6),
        "tx_time_s": round(pgrl.pass_peak_s, 4),
        "doppler_precomp_ratio": precomp_ratio,
        "lrfhss_mode": True,
        "num_hops": num_hops,
        "hop_length_ms": hop_length_ms,
        "header_disabled": False,
        "invert_iq": False,
        "metadata": pgrl.to_dict(),
    }


def save_config(config: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_example() -> PGRLOutput:
    """Representative PGRLOutput for a 400 km LEO pass."""
    return PGRLOutput(
        pass_start_s=0.0,
        pass_peak_s=120.0,
        pass_end_s=240.0,
        doppler_hz=1250.0,
        doppler_rate_hz_s=35.0,
        timing_sigma_s=0.016,
        doppler_sigma_hz=300.0,
        link_score=0.92,
    )


if __name__ == "__main__":
    pgrl = load_example()
    cfg = make_semtech_tx_config(pgrl, base_freq_hz=915_000_000, tx_power_dbm=14)
    out_path = FILE_DIR / "lr1121_tx_config_example.json"
    save_config(cfg, str(out_path))
    print(f"Config → {out_path}")
    print(json.dumps(cfg, indent=2))